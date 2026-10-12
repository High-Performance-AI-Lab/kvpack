use super::*;

pub(super) const MIN_AUDIT_SEGMENT_BYTES: usize = 64 * 1024;
const MAX_AUDIT_SEGMENT_BYTES: usize = 4 * 1024 * 1024;
const MAX_AUDIT_SEGMENTS: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditDirectoryPolicy {
    maximum_segment_bytes: usize,
    maximum_segments: usize,
    retention: Duration,
}

impl AuditDirectoryPolicy {
    pub fn new(
        maximum_segment_bytes: usize,
        maximum_segments: usize,
        retention: Duration,
    ) -> Result<Self, StoreError> {
        if !(MIN_AUDIT_SEGMENT_BYTES..=MAX_AUDIT_SEGMENT_BYTES).contains(&maximum_segment_bytes)
            || maximum_segments == 0
            || maximum_segments > MAX_AUDIT_SEGMENTS
            || retention < Duration::from_secs(60 * 60)
            || retention > Duration::from_secs(365 * 24 * 60 * 60)
        {
            return Err(StoreError::State(
                "audit directory policy is outside fixed bounds",
            ));
        }
        Ok(Self {
            maximum_segment_bytes,
            maximum_segments,
            retention,
        })
    }

    pub fn production_v1() -> Self {
        Self::new(1024 * 1024, 4_096, Duration::from_secs(30 * 24 * 60 * 60))
            .expect("production audit directory policy is valid")
    }
}

#[derive(Debug)]
pub struct AuditDirectoryExporter {
    root: PathBuf,
    policy: AuditDirectoryPolicy,
}

impl AuditDirectoryExporter {
    pub fn new(root: PathBuf, policy: AuditDirectoryPolicy) -> Result<Self, StoreError> {
        create_private_dir(&root)?;
        Ok(Self { root, policy })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl AuditExporter for AuditDirectoryExporter {
    fn export(&self, batch: &AuditBatch) -> Result<(), StoreError> {
        if batch.records.is_empty() || batch.records.len() > MAX_AUDIT_BATCH_RECORDS {
            return Err(StoreError::State(
                "audit export batch is empty or exceeds its bound",
            ));
        }
        let bytes = encode_batch(batch)?;
        if bytes.len() > self.policy.maximum_segment_bytes {
            return Err(StoreError::Quota(
                "audit segment exceeds its configured byte bound",
            ));
        }
        let digest: Id32 = Sha256::digest(&bytes).into();
        let filename = format!(
            "audit-{:020}-{:020}-{}.jsonl",
            batch.first_sequence(),
            batch.last_sequence(),
            hex(&digest)
        );
        let target = self.root.join(filename);
        publish_segment(&self.root, &target, &bytes)?;
        prune_segments(&self.root, &target, self.policy)?;
        Ok(())
    }
}

fn encode_batch(batch: &AuditBatch) -> Result<Vec<u8>, StoreError> {
    let mut output = String::with_capacity(batch.records.len().saturating_mul(192));
    for record in &batch.records {
        writeln!(
            output,
            "{{\"schema\":{},\"sequence\":{},\"event\":\"{}\",\"object\":\"{}\",\"id\":\"{}\",\"generation\":{},\"occurred_unix_ns\":{}}}",
            AUDIT_SCHEMA_VERSION,
            record.sequence,
            record.event.as_str(),
            record.object.as_str(),
            hex(&record.object_id),
            record.generation,
            record.occurred_unix_ns
        )?;
    }
    Ok(output.into_bytes())
}

fn publish_segment(root: &Path, target: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    if target.exists() {
        return verify_segment(target, bytes);
    }
    let mut random = [0u8; 16];
    getrandom::fill(&mut random)
        .map_err(|_| StoreError::State("audit segment filename entropy unavailable"))?;
    let partial = root.join(format!(".audit-{}.partial", hex(&random)));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    let mut file = options
        .open(&partial)
        .map_err(io_error("create private audit segment partial"))?;
    file.write_all(bytes)
        .map_err(io_error("write private audit segment partial"))?;
    file.sync_all()
        .map_err(io_error("fsync private audit segment partial"))?;
    drop(file);
    match fs::hard_link(&partial, target) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_segment(target, bytes)?;
        }
        Err(source) => {
            return Err(StoreError::Io {
                op: "publish immutable audit segment",
                source,
            });
        }
    }
    fs::remove_file(&partial).map_err(io_error("unlink audit segment partial"))?;
    fs::set_permissions(target, fs::Permissions::from_mode(0o600))
        .map_err(io_error("set private audit segment permissions"))?;
    fsync_dir(root)
}

fn verify_segment(path: &Path, expected: &[u8]) -> Result<(), StoreError> {
    let mut file = fs::File::open(path).map_err(io_error("open immutable audit segment"))?;
    let metadata = file
        .metadata()
        .map_err(io_error("inspect immutable audit segment"))?;
    if !metadata.is_file() || metadata.len() != expected.len() as u64 {
        return Err(StoreError::Integrity(
            "immutable audit segment bounds changed",
        ));
    }
    let mut existing = Vec::with_capacity(expected.len());
    file.read_to_end(&mut existing)
        .map_err(io_error("read immutable audit segment"))?;
    if existing != expected {
        return Err(StoreError::Integrity(
            "immutable audit segment bytes changed",
        ));
    }
    Ok(())
}

fn prune_segments(
    root: &Path,
    current: &Path,
    policy: AuditDirectoryPolicy,
) -> Result<(), StoreError> {
    let now = SystemTime::now();
    let mut segments = Vec::new();
    for entry in fs::read_dir(root).map_err(io_error("list audit segments"))? {
        let entry = entry.map_err(io_error("read audit segment entry"))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("audit-") || !name.ends_with(".jsonl") {
            continue;
        }
        if segments.len() == MAX_AUDIT_SEGMENTS {
            return Err(StoreError::Quota(
                "audit segment directory exceeds its fixed bound",
            ));
        }
        let file_type = entry
            .file_type()
            .map_err(io_error("inspect audit segment type"))?;
        if file_type.is_symlink() || !file_type.is_file() {
            return Err(StoreError::Integrity(
                "audit segment entry is not a regular file",
            ));
        }
        let metadata = entry
            .metadata()
            .map_err(io_error("inspect audit segment"))?;
        segments.push((
            entry.path(),
            metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        ));
    }
    segments.sort_by(|left, right| left.0.file_name().cmp(&right.0.file_name()));
    let mut changed = false;
    for (path, modified) in &segments {
        if path != current
            && now.duration_since(*modified).unwrap_or_default() > policy.retention
            && path.exists()
        {
            fs::remove_file(path).map_err(io_error("prune expired audit segment"))?;
            changed = true;
        }
    }
    let mut remaining = segments
        .into_iter()
        .filter(|(path, _)| path.exists())
        .collect::<Vec<_>>();
    while remaining.len() > policy.maximum_segments {
        let index = remaining
            .iter()
            .position(|(path, _)| path != current)
            .ok_or(StoreError::State(
                "audit rotation cannot retain current segment",
            ))?;
        let (path, _) = remaining.remove(index);
        fs::remove_file(path).map_err(io_error("rotate oldest audit segment"))?;
        changed = true;
    }
    if changed {
        fsync_dir(root)?;
    }
    Ok(())
}
