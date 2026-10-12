use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::{
    canonical_json, decode_canonical_json, BeginManifestV1, HandoffError, IncrementalVerifierV1,
    LayerHeaderV1, Result, SealManifestV1, TensorRoleV1, ValidationLimits, VerifiedPlaneV1,
    VerifiedSealV1,
};

const MANIFEST_MAX_BYTES: usize = 1024 * 1024;

/// Compatibility name retained for sealed-bundle callers.
pub type VerifiedPlane = VerifiedPlaneV1;

/// A sealed bundle that was reopened and re-verified streaming: every
/// plane file was hashed and its bytes dropped before the next plane was
/// read, so peak memory stays at one bounded frame instead of the whole
/// bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBundle {
    path: PathBuf,
    begin: BeginManifestV1,
    seal: SealManifestV1,
}

impl VerifiedBundle {
    pub fn open(path: impl AsRef<Path>, limits: &ValidationLimits) -> Result<Self> {
        let opened = open_verified(path, limits, false)?;
        Ok(Self {
            path: opened.path,
            begin: opened.begin,
            seal: opened.seal,
        })
    }

    /// Identical verification to [`VerifiedBundle::open`], but retains
    /// every authenticated plane for callers that need the bytes.
    pub fn open_materialized(
        path: impl AsRef<Path>,
        limits: &ValidationLimits,
    ) -> Result<MaterializedVerifiedBundle> {
        let opened = open_verified(path, limits, true)?;
        Ok(MaterializedVerifiedBundle {
            path: opened.path,
            begin: opened.begin,
            planes: opened
                .planes
                .expect("materialized open retains every verified plane"),
            seal: opened.seal,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn begin(&self) -> &BeginManifestV1 {
        &self.begin
    }

    pub fn seal(&self) -> &SealManifestV1 {
        &self.seal
    }
}

/// A sealed bundle reopened with every verified plane retained in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedVerifiedBundle {
    path: PathBuf,
    begin: BeginManifestV1,
    planes: Vec<VerifiedPlane>,
    seal: SealManifestV1,
}

impl MaterializedVerifiedBundle {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn begin(&self) -> &BeginManifestV1 {
        &self.begin
    }

    pub fn planes(&self) -> &[VerifiedPlane] {
        &self.planes
    }

    pub fn seal(&self) -> &SealManifestV1 {
        &self.seal
    }

    /// F1 (import path): authenticate this reopened bundle under the armed
    /// tenant key. The reopen already re-verified begin, every plane, and
    /// the seal integrity; this recomputes the keyed tag over the
    /// authenticated begin + headers + core. A consumer that loads a
    /// sealed bundle from anywhere but the authenticated receiver (local
    /// file, cache hop) calls this before installing any state so a
    /// forged artifact without the key fails closed.
    pub fn authenticate_hmac(&self, key: &crate::MacKey) -> Result<()> {
        let headers: Vec<LayerHeaderV1> = self
            .planes
            .iter()
            .map(|plane| plane.header.clone())
            .collect();
        self.seal.authenticate_hmac(&self.begin, &headers, key)
    }
}

struct OpenedBundle {
    path: PathBuf,
    begin: BeginManifestV1,
    planes: Option<Vec<VerifiedPlane>>,
    seal: SealManifestV1,
}

/// Reopen one sealed bundle and re-verify every byte: both manifests, each
/// plane in sequence against the incremental verifier, the exact layers
/// directory entry set, the terminal seal, and the READY marker. With
/// `materialize` the verified planes are retained; otherwise each plane's
/// bytes are dropped as soon as its hash authenticated.
fn open_verified(
    path: impl AsRef<Path>,
    limits: &ValidationLimits,
    materialize: bool,
) -> Result<OpenedBundle> {
    let requested = path.as_ref();
    reject_symlink(requested, "bundle")?;
    let path = fs::canonicalize(requested)?;
    if !path.is_dir() {
        return Err(HandoffError::Validation(
            "bundle path is not a directory".into(),
        ));
    }
    let begin_path = path.join("begin.json");
    let seal_path = path.join("seal.json");
    let ready_path = path.join("READY");
    let layers_path = path.join("layers");
    for (candidate, label) in [
        (&begin_path, "begin manifest"),
        (&seal_path, "seal manifest"),
        (&ready_path, "READY marker"),
        (&layers_path, "layers directory"),
    ] {
        reject_symlink(candidate, label)?;
    }
    let begin_bytes = read_regular_bounded(&begin_path, MANIFEST_MAX_BYTES)?;
    let begin: BeginManifestV1 = decode_canonical_json(&begin_bytes, MANIFEST_MAX_BYTES)?;
    begin.validate(limits)?;
    let seal_bytes = read_regular_bounded(&seal_path, MANIFEST_MAX_BYTES)?;
    let seal: SealManifestV1 = decode_canonical_json(&seal_bytes, MANIFEST_MAX_BYTES)?;

    let frame_count = usize::try_from(begin.expected_layer_frames)
        .map_err(|_| HandoffError::Validation("frame count exceeds usize".into()))?;
    let mut planes = materialize.then(|| Vec::with_capacity(frame_count));
    let mut verifier = IncrementalVerifierV1::new(begin.clone(), limits.clone())?;
    let mut expected_entries = BTreeSet::new();
    // v2: the declared layout table defines which (layer, role) sits at
    // each sequence; the flat walk is built once and indexed per frame.
    let layout_walk = begin.is_v2().then(|| begin.layout_walk_v2());
    for sequence in 0..begin.expected_layer_frames {
        // v1: the walk is ascending layers, K then V. File stems stay
        // layer-keyed either way.
        let (layer, role) = if let Some(walk) = &layout_walk {
            let &(_, layer, role) = walk.get(sequence as usize).ok_or_else(|| {
                HandoffError::Validation(format!(
                    "layer frame {sequence} is outside the declared layout table"
                ))
            })?;
            (layer, role)
        } else {
            (
                sequence / 2,
                if sequence % 2 == 0 {
                    TensorRoleV1::Key
                } else {
                    TensorRoleV1::Value
                },
            )
        };
        let stem = format!("{layer:05}-{}", role.suffix());
        let header_name = format!("{stem}.json");
        let payload_name = format!("{stem}.{}", begin.plane_payload_extension(role));
        expected_entries.insert(header_name.clone());
        expected_entries.insert(payload_name.clone());
        let header_path = layers_path.join(header_name);
        let payload_path = layers_path.join(payload_name);
        reject_symlink(&header_path, "layer header")?;
        reject_symlink(&payload_path, "layer payload")?;
        let header_bytes = read_regular_bounded(&header_path, MANIFEST_MAX_BYTES)?;
        let header: LayerHeaderV1 = decode_canonical_json(&header_bytes, MANIFEST_MAX_BYTES)?;
        let payload_limit = usize::try_from(limits.max_frame_bytes).unwrap_or(usize::MAX);
        let bytes = read_regular_bounded(&payload_path, payload_limit)?;
        let plane = verifier.verify_plane(header, bytes)?;
        if let Some(planes) = &mut planes {
            planes.push(plane);
        }
    }
    let actual_entries = fs::read_dir(&layers_path)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<BTreeSet<_>>>()?;
    if actual_entries != expected_entries {
        return Err(HandoffError::Validation(
            "layers directory contains missing or unexpected entries".into(),
        ));
    }
    verifier.verify_seal(seal.clone())?;
    let ready = read_regular_bounded(&ready_path, 65)?;
    let expected_ready = format!("{}\n", seal.artifact_sha256);
    if ready != expected_ready.as_bytes() {
        return Err(HandoffError::Validation(
            "READY marker does not match the sealed artifact".into(),
        ));
    }
    Ok(OpenedBundle {
        path,
        begin,
        planes,
        seal,
    })
}

pub struct BundleStager {
    begin: BeginManifestV1,
    committed: bool,
    final_path: PathBuf,
    prepared: Option<SealManifestV1>,
    verifier: IncrementalVerifierV1,
    staging_path: PathBuf,
    aborted: bool,
}

impl BundleStager {
    pub fn create(
        final_path: impl AsRef<Path>,
        begin: BeginManifestV1,
        limits: ValidationLimits,
    ) -> Result<Self> {
        begin.validate(&limits)?;
        let verifier = IncrementalVerifierV1::new(begin.clone(), limits)?;
        let requested = final_path.as_ref();
        if requested.file_name().is_none() {
            return Err(HandoffError::Validation(
                "bundle output must name one child directory".into(),
            ));
        }
        let parent = requested.parent().unwrap_or_else(|| Path::new("."));
        reject_symlink(parent, "bundle parent")?;
        let parent = fs::canonicalize(parent)?;
        let final_path = parent.join(requested.file_name().expect("checked above"));
        match final_path.symlink_metadata() {
            Ok(_) => {
                return Err(HandoffError::Validation(format!(
                    "bundle output already exists: {}",
                    final_path.display()
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let name = final_path
            .file_name()
            .expect("checked above")
            .to_string_lossy();
        let staging_path = parent.join(format!(".{name}.{}.partial", &begin.transfer_id[..16]));
        fs::create_dir(&staging_path)?;
        fs::create_dir(staging_path.join("layers"))?;
        write_new_sync(&staging_path.join("begin.json"), &canonical_json(&begin)?)?;
        Ok(Self {
            begin,
            committed: false,
            final_path,
            prepared: None,
            verifier,
            staging_path,
            aborted: false,
        })
    }

    pub fn begin(&self) -> &BeginManifestV1 {
        &self.begin
    }

    pub fn staging_path(&self) -> &Path {
        &self.staging_path
    }

    pub(crate) fn verifier(&self) -> &IncrementalVerifierV1 {
        &self.verifier
    }

    pub(crate) fn verifier_mut(&mut self) -> &mut IncrementalVerifierV1 {
        &mut self.verifier
    }

    /// Stage a plane authenticated by this stager's own verifier without
    /// replaying the hash. Used by the streaming coordinator, which shares
    /// the verifier, so the anti-mixing guarantee holds by construction.
    pub(crate) fn stage_shared_verified(&mut self, plane: &VerifiedPlaneV1) -> Result<()> {
        self.write_verified(plane)
    }

    pub fn ingest(&mut self, header: LayerHeaderV1, payload: &[u8]) -> Result<()> {
        let verified = self.verifier.verify_plane(header, payload.to_vec())?;
        self.write_verified(&verified)
    }

    /// Persist an already-authenticated plane. The stager independently
    /// replays the same verifier before writing so witnesses cannot be mixed
    /// across BEGIN declarations or stream cursors.
    pub fn stage_verified(&mut self, plane: &VerifiedPlaneV1) -> Result<()> {
        let locally_verified = self
            .verifier
            .verify_plane(plane.header.clone(), plane.bytes.clone())?;
        if locally_verified != *plane {
            return Err(HandoffError::Validation(
                "verified plane changed before internal staging".into(),
            ));
        }
        self.write_verified(plane)
    }

    fn write_verified(&mut self, plane: &VerifiedPlaneV1) -> Result<()> {
        if self.committed || self.prepared.is_some() || self.aborted {
            return Err(HandoffError::Validation(
                "cannot stage a plane outside the receiving state".into(),
            ));
        }
        // Each plane file is fsynced as soon as it is written so the
        // durability work overlaps the network receive; the publish barrier
        // then owes only READY, the directory entries, and the atomic
        // rename.
        let header = &plane.header;
        let stem = format!("{:05}-{}", header.layer, header.role.suffix());
        let layers = self.staging_path.join("layers");
        write_new_sync(
            &layers.join(format!("{stem}.json")),
            &canonical_json(header)?,
        )?;
        write_new_sync(
            &layers.join(format!(
                "{stem}.{}",
                self.begin.plane_payload_extension(header.role)
            )),
            &plane.bytes,
        )?;
        Ok(())
    }

    pub fn staged_header_path(&self, header: &LayerHeaderV1) -> PathBuf {
        let stem = format!("{:05}-{}", header.layer, header.role.suffix());
        self.staging_path
            .join("layers")
            .join(format!("{stem}.json"))
    }

    pub fn staged_payload_path(&self, header: &LayerHeaderV1) -> PathBuf {
        let stem = format!("{:05}-{}", header.layer, header.role.suffix());
        self.staging_path.join("layers").join(format!(
            "{stem}.{}",
            self.begin.plane_payload_extension(header.role)
        ))
    }

    /// Durably write the terminal manifest without creating `READY` or making
    /// the bundle visible at its final path.
    pub fn prepare_seal(&mut self, verified: &VerifiedSealV1) -> Result<()> {
        if self.committed || self.prepared.is_some() || self.aborted {
            return Err(HandoffError::Validation(
                "bundle cannot prepare another terminal seal".into(),
            ));
        }
        let local = self.verifier.verify_seal(verified.manifest().clone())?;
        if &local != verified {
            return Err(HandoffError::Validation(
                "terminal seal witness belongs to a different verified stream".into(),
            ));
        }
        self.write_prepared_seal(verified.manifest())
    }

    /// Prepare a seal witnessed by this stager's own shared verifier. The
    /// coordinator produced the witness from this very verifier, so the
    /// anti-mixing check holds by construction; re-verifying would trip the
    /// single-seal guard.
    pub(crate) fn prepare_shared_seal(&mut self, verified: &VerifiedSealV1) -> Result<()> {
        if self.committed || self.prepared.is_some() || self.aborted {
            return Err(HandoffError::Validation(
                "bundle cannot prepare another terminal seal".into(),
            ));
        }
        self.write_prepared_seal(verified.manifest())
    }

    fn write_prepared_seal(&mut self, seal: &SealManifestV1) -> Result<()> {
        write_new_sync(&self.staging_path.join("seal.json"), &canonical_json(seal)?)?;
        File::open(self.staging_path.join("layers"))?.sync_all()?;
        File::open(&self.staging_path)?.sync_all()?;
        self.prepared = Some(seal.clone());
        Ok(())
    }

    /// Write `READY`, fsync, and publish the prepared directory with a
    /// no-replace atomic rename.
    pub fn publish(&mut self) -> Result<PathBuf> {
        if self.committed || self.aborted {
            return Err(HandoffError::Validation(
                "bundle cannot be published in its current state".into(),
            ));
        }
        let seal = self
            .prepared
            .as_ref()
            .ok_or_else(|| HandoffError::Validation("bundle seal was not prepared".into()))?;
        // Plane and seal files were already fsynced when written; the
        // barrier owes only READY, the directory entries, and the rename.
        write_new_sync(
            &self.staging_path.join("READY"),
            format!("{}\n", seal.artifact_sha256).as_bytes(),
        )?;
        File::open(self.staging_path.join("layers"))?.sync_all()?;
        File::open(&self.staging_path)?.sync_all()?;
        let parent_path = self
            .final_path
            .parent()
            .expect("canonical final path has a parent");
        let parent = File::open(parent_path)?;
        rustix::fs::renameat_with(
            &parent,
            self.staging_path
                .file_name()
                .expect("staging path has a file name"),
            &parent,
            self.final_path
                .file_name()
                .expect("final path has a file name"),
            rustix::fs::RenameFlags::NOREPLACE,
        )
        .map_err(std::io::Error::from)?;
        parent.sync_all()?;
        self.committed = true;
        Ok(self.final_path.clone())
    }

    /// Compatibility wrapper for the original seal-and-publish operation.
    pub fn seal(&mut self, seal: SealManifestV1) -> Result<PathBuf> {
        if self.committed || self.prepared.is_some() || self.aborted {
            return Err(HandoffError::Validation(
                "bundle cannot be sealed in its current state".into(),
            ));
        }
        let verified = self.verifier.verify_seal(seal)?;
        self.write_prepared_seal(verified.manifest())?;
        self.publish()
    }

    /// Remove only this transfer-owned, never-published staging directory.
    pub fn abort(&mut self) -> Result<()> {
        if self.committed || self.aborted || !self.staging_path.exists() {
            return Ok(());
        }
        fs::remove_dir_all(&self.staging_path)?;
        self.aborted = true;
        Ok(())
    }
}

fn write_new_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn reject_symlink(path: &Path, label: &str) -> Result<()> {
    let metadata = path.symlink_metadata()?;
    if metadata.file_type().is_symlink() {
        return Err(HandoffError::Validation(format!(
            "{label} must not be a symlink: {}",
            path.display()
        )));
    }
    Ok(())
}

fn read_regular_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let metadata = path.symlink_metadata()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(HandoffError::Validation(format!(
            "expected a regular non-symlink file: {}",
            path.display()
        )));
    }
    let length = usize::try_from(metadata.len())
        .map_err(|_| HandoffError::Validation("file length exceeds usize".into()))?;
    if length == 0 || length > max_bytes {
        return Err(HandoffError::Validation(format!(
            "file {} length {length} is outside 1..={max_bytes}",
            path.display()
        )));
    }
    let mut bytes = Vec::with_capacity(length);
    File::open(path)?
        .take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() != length {
        return Err(HandoffError::Validation(format!(
            "file {} changed while being read",
            path.display()
        )));
    }
    Ok(bytes)
}
