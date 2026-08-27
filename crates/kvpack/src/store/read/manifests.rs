use super::*;

/// Bound on the decoded-manifest LRU. Entries are inserted once — after full
/// authentication and validation — and never need refresh, because manifest
/// decodes are immutable by content identity.
const MANIFEST_CACHE_CAPACITY: usize = 32;

/// Small LRU of decoded manifests keyed by manifest content identity. A hit
/// skips the decode/canonical re-encode passes, but every fetch still
/// re-reads the object, re-authenticates its bytes (framing, header digest,
/// HMAC) against the store key, and re-validates the cached decode against
/// the CALLER's ValidationContext — the entry was validated under the first
/// caller's bounds, which a stricter later caller must not inherit.
pub(crate) struct ManifestLru {
    entries: HashMap<Id32, Arc<CutManifest>>,
    order: VecDeque<Id32>,
}

impl ManifestLru {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, manifest_id: &Id32) -> Option<Arc<CutManifest>> {
        let entry = self.entries.get(manifest_id).cloned()?;
        if let Some(position) = self.order.iter().position(|id| id == manifest_id) {
            self.order.remove(position);
        }
        self.order.push_back(*manifest_id);
        Some(entry)
    }

    fn put(&mut self, manifest_id: Id32, manifest: Arc<CutManifest>) {
        if self.entries.contains_key(&manifest_id) {
            return;
        }
        while self.entries.len() >= MANIFEST_CACHE_CAPACITY {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.entries.insert(manifest_id, manifest);
        self.order.push_back(manifest_id);
    }
}

impl LocalStore {
    /// Authenticate a manifest before exposing the minimum final shadow size
    /// needed by admission. The client cannot reduce this value.
    pub fn authenticated_resource_minimum(
        &self,
        manifest_id: &Id32,
        context: &ValidationContext,
    ) -> Result<u64, StoreError> {
        let bytes = self.read_authenticated_manifest_object(manifest_id, context)?;
        let header = inspect_pack_header(&bytes)?;
        let keys = self.schedule(header.key_epoch)?;
        Ok(decode_authenticated_pack(&bytes, &keys, context)?
            .realized_schema
            .complete_restored_bytes)
    }

    /// Return a stored manifest object only after authenticating and validating
    /// it against this tenant. Intended for path-free gateway transmission.
    pub fn read_authenticated_manifest_object(
        &self,
        manifest_id: &Id32,
        context: &ValidationContext,
    ) -> Result<Vec<u8>, StoreError> {
        Ok(self.authenticated_manifest_cached(manifest_id, context)?.0)
    }

    /// Authenticate a manifest once, then serve subsequent fetches with the
    /// cached decode while still re-reading and re-authenticating the object
    /// bytes (catalog gate, framing, header digest, HMAC) and re-validating
    /// the decode against the caller's ValidationContext on every call:
    /// on-disk deletion or tampering fails closed exactly like the uncached
    /// path, but the decode/canonical re-encode passes run once.
    pub(crate) fn authenticated_manifest_cached(
        &self,
        manifest_id: &Id32,
        context: &ValidationContext,
    ) -> Result<(Vec<u8>, Arc<CutManifest>), StoreError> {
        self.manifest_catalog_file_bytes(manifest_id)?;
        let bytes = self.read_manifest_bytes(manifest_id)?;
        let header = inspect_pack_header(&bytes)?;
        if header.tenant_namespace != self.tenant_namespace {
            return Err(StoreError::Authentication(
                "manifest belongs to another tenant",
            ));
        }
        let keys = self.schedule(header.key_epoch)?;
        if let Some(manifest) = self
            .manifest_cache
            .lock()
            .map_err(|_| StoreError::State("manifest cache mutex poisoned"))?
            .get(manifest_id)
        {
            let verified = kvpack_core::verify_authenticated_pack(&bytes, &keys)?;
            if verified.manifest_id != *manifest_id {
                return Err(StoreError::Authentication(
                    "manifest catalog identity mismatch",
                ));
            }
            // The cache entry was validated under the FIRST caller's
            // ValidationContext; a later, stricter caller (smaller
            // max_restored_bytes etc.) must not inherit that pass, so the
            // cached decode is re-validated against THIS caller's context on
            // every hit (audit N3). This is cheap relative to the byte
            // re-authentication above and keeps the fail-closed posture.
            validate_manifest(&manifest, context)?;
            self.require_manifest_readable_epochs(&manifest)?;
            return Ok((bytes, manifest));
        }
        let manifest = decode_authenticated_pack(&bytes, &keys, context)?;
        // `decode_authenticated_pack` already proved
        // `manifest_id(verified canonical body) == header.manifest_id`, and
        // the header itself is digest- and HMAC-authenticated, so the header
        // identity *is* the manifest identity; no re-encode pass is needed.
        if header.manifest_id != *manifest_id {
            return Err(StoreError::Authentication(
                "manifest catalog identity mismatch",
            ));
        }
        self.require_manifest_readable_epochs(&manifest)?;
        let manifest = Arc::new(manifest);
        self.manifest_cache
            .lock()
            .map_err(|_| StoreError::State("manifest cache mutex poisoned"))?
            .put(*manifest_id, Arc::clone(&manifest));
        Ok((bytes, manifest))
    }

    fn require_manifest_readable_epochs(&self, manifest: &CutManifest) -> Result<(), StoreError> {
        if manifest
            .states
            .iter()
            .flat_map(|state| &state.chunks)
            .any(|chunk| {
                chunk.key_epoch < self.minimum_readable_key_epoch()
                    || chunk.key_epoch > self.key_epoch()
            })
        {
            return Err(StoreError::NotFound);
        }
        Ok(())
    }

    fn manifest_catalog_file_bytes(&self, manifest_id: &Id32) -> Result<u64, StoreError> {
        let connection = self.lock_catalog()?;
        let bytes: Option<u64> = connection.query_row("SELECT file_bytes FROM manifests WHERE tenant=?1 AND manifest_id=?2 AND key_epoch>=?3 AND key_epoch<=?4 AND NOT EXISTS(SELECT 1 FROM tombstones WHERE tenant=?1 AND object_kind='manifest' AND object_id=?2)", params![self.tenant_namespace.as_slice(), manifest_id.as_slice(), self.minimum_readable_key_epoch(), self.key_epoch()], |row| row.get(0)).optional()?;
        bytes.ok_or(StoreError::NotFound)
    }

    pub(crate) fn read_manifest_bytes(&self, manifest_id: &Id32) -> Result<Vec<u8>, StoreError> {
        let bytes = self.manifest_catalog_file_bytes(manifest_id)?;
        let maximum = PACK_HEADER_BYTES + MAX_MANIFEST_BYTES + 16 + PACK_FOOTER_BYTES;
        if bytes as usize > maximum {
            return Err(StoreError::Authentication(
                "catalog manifest size exceeds bound",
            ));
        }
        let path = self.manifest_path(manifest_id);
        let mut file = match fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(StoreError::NotFound);
            }
            Err(source) => {
                return Err(StoreError::Io {
                    op: "open manifest object",
                    source,
                });
            }
        };
        let mut result = Vec::with_capacity(bytes as usize);
        (&mut file)
            .take(bytes + 1)
            .read_to_end(&mut result)
            .map_err(io_error("read manifest object"))?;
        if result.len() as u64 != bytes {
            return Err(StoreError::Authentication("manifest object length changed"));
        }
        Ok(result)
    }
}
