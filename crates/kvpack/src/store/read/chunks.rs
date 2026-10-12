use super::*;

const PUBLICATION_LEASE_REFRESH_MARGIN_NS: u64 = 30 * 1_000_000_000;

/// Public metadata for one chunk in canonical global manifest order. The
/// object itself remains behind [`AuthenticatedPublicationSource`], so callers
/// never need catalog rows or filesystem paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthenticatedPublicationChunk {
    ordinal: u64,
    object_key: Id32,
    object_bytes: u32,
}

impl AuthenticatedPublicationChunk {
    pub const fn ordinal(self) -> u64 {
        self.ordinal
    }

    pub const fn object_key(self) -> Id32 {
        self.object_key
    }

    pub const fn object_bytes(self) -> u32 {
        self.object_bytes
    }
}

#[derive(Clone)]
struct PublicationChunkEntry {
    public: AuthenticatedPublicationChunk,
    state_key: StateKey,
    reference: ChunkRef,
    span: ChunkSpan,
}

/// One authenticated local manifest and its exact ordered chunk stream.
///
/// Construction acquires a durable source lease over the complete closure.
/// Reads renew that lease near expiry, authenticate each pinned chunk, and
/// return only bytes bound by the manifest. Dropping or explicitly releasing
/// the source releases the lease; a process crash leaves it uncertain for the
/// existing restart reconciliation path.
pub struct AuthenticatedPublicationSource {
    store: Arc<LocalStore>,
    manifest_id: Id32,
    manifest_object: Arc<[u8]>,
    tenant_namespace: Id32,
    family: RepresentationFamilyId,
    chunks: Vec<PublicationChunkEntry>,
    expected_bytes: u64,
    lease_id: Id32,
    owner_id: Id32,
    owner_incarnation: Id32,
    authority_term: u64,
    lease_expires_ns: u64,
    released: bool,
}

impl std::fmt::Debug for AuthenticatedPublicationSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AuthenticatedPublicationSource")
            .field("manifest_id", &hex(&self.manifest_id))
            .field("chunk_count", &self.chunks.len())
            .field("expected_bytes", &self.expected_bytes)
            .field("released", &self.released)
            .finish()
    }
}

impl AuthenticatedPublicationSource {
    pub const fn manifest_id(&self) -> Id32 {
        self.manifest_id
    }

    pub fn manifest_object(&self) -> &[u8] {
        &self.manifest_object
    }

    /// Clone the immutable manifest allocation without copying its bytes.
    pub fn shared_manifest_object(&self) -> Arc<[u8]> {
        Arc::clone(&self.manifest_object)
    }

    pub fn chunk_count(&self) -> u64 {
        self.chunks.len() as u64
    }

    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    pub fn chunk(&self, ordinal: u64) -> Option<AuthenticatedPublicationChunk> {
        usize::try_from(ordinal)
            .ok()
            .and_then(|ordinal| self.chunks.get(ordinal))
            .map(|entry| entry.public)
    }

    /// Renew the durable source lease. Long-running publishers may call this
    /// after a slow network write; ordinary chunk reads also renew near expiry.
    pub fn renew(&mut self) -> Result<(), StoreError> {
        if self.released {
            return Err(StoreError::State("publication source was released"));
        }
        let status = self.store.reattach_source_lease(
            &self.lease_id,
            &self.owner_id,
            &self.owner_incarnation,
            self.authority_term,
            MAX_SOURCE_LEASE_NS,
        )?;
        self.lease_expires_ns = status.expires_ns;
        Ok(())
    }

    /// Read and authenticate one exact global manifest ordinal.
    pub fn read_chunk_object(
        &mut self,
        ordinal: u64,
    ) -> Result<(AuthenticatedPublicationChunk, Vec<u8>), StoreError> {
        if self.released {
            return Err(StoreError::State("publication source was released"));
        }
        if self.lease_expires_ns <= now_ns().saturating_add(PUBLICATION_LEASE_REFRESH_MARGIN_NS) {
            self.renew()?;
        }
        let index = usize::try_from(ordinal)
            .map_err(|_| StoreError::Expectation("publication chunk ordinal exceeds usize"))?;
        let entry = self.chunks.get(index).ok_or(StoreError::Expectation(
            "publication chunk ordinal is outside the manifest",
        ))?;
        self.store.require_source_lease_object(
            &self.lease_id,
            &self.owner_id,
            InventoryObjectKind::Chunk,
            &entry.reference.object_key,
        )?;
        let bytes = read_validated_chunk_object(
            &self.store,
            &self.tenant_namespace,
            &self.family,
            &entry.state_key,
            &entry.reference,
            &entry.span,
        )?;
        Ok((entry.public, bytes))
    }

    /// Read, authenticate, and decode one exact global manifest ordinal,
    /// returning the plaintext state bytes. Same lease renewal and closure
    /// binding as [`Self::read_chunk_object`]; only the decoding differs.
    pub fn read_chunk_plaintext(&mut self, ordinal: u64) -> Result<Vec<u8>, StoreError> {
        if self.released {
            return Err(StoreError::State("publication source was released"));
        }
        if self.lease_expires_ns <= now_ns().saturating_add(PUBLICATION_LEASE_REFRESH_MARGIN_NS) {
            self.renew()?;
        }
        let index = usize::try_from(ordinal)
            .map_err(|_| StoreError::Expectation("publication chunk ordinal exceeds usize"))?;
        let entry = self.chunks.get(index).ok_or(StoreError::Expectation(
            "publication chunk ordinal is outside the manifest",
        ))?;
        self.store.require_source_lease_object(
            &self.lease_id,
            &self.owner_id,
            InventoryObjectKind::Chunk,
            &entry.reference.object_key,
        )?;
        read_validated_chunk_plaintext(
            &self.store,
            &self.tenant_namespace,
            &self.family,
            &entry.state_key,
            &entry.reference,
            &entry.span,
        )
    }

    /// Release the complete source closure. This operation is idempotent.
    pub fn release(&mut self) -> Result<(), StoreError> {
        if self.released {
            return Ok(());
        }
        self.store
            .release_source_lease(&self.lease_id, &self.owner_id, self.authority_term)?;
        self.released = true;
        Ok(())
    }
}

impl Drop for AuthenticatedPublicationSource {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

impl LocalStore {
    /// Open a path-free, lease-protected source for publishing one exact local
    /// manifest and its chunks to another immutable store.
    pub fn authenticated_publication_source(
        self: &Arc<Self>,
        manifest_id: &Id32,
        context: &ValidationContext,
    ) -> Result<AuthenticatedPublicationSource, StoreError> {
        let manifest_object = self.read_authenticated_manifest_object(manifest_id, context)?;
        let header = inspect_pack_header(&manifest_object)?;
        let keys = self.schedule(header.key_epoch)?;
        let manifest = decode_authenticated_pack(&manifest_object, &keys, context)?;
        let mut expected_bytes = u64::try_from(manifest_object.len())
            .map_err(|_| StoreError::Expectation("publication manifest length exceeds u64"))?;
        let mut chunks = Vec::new();
        for (state, schema) in manifest.states.iter().zip(&manifest.realized_schema.states) {
            for (reference, span) in state.chunks.iter().zip(&schema.chunk_spans) {
                let ordinal = u64::try_from(chunks.len())
                    .map_err(|_| StoreError::Expectation("publication chunk count exceeds u64"))?;
                expected_bytes = expected_bytes
                    .checked_add(u64::from(reference.object_bytes))
                    .ok_or(StoreError::Expectation(
                        "publication expected byte count overflows",
                    ))?;
                chunks.push(PublicationChunkEntry {
                    public: AuthenticatedPublicationChunk {
                        ordinal,
                        object_key: reference.object_key,
                        object_bytes: reference.object_bytes,
                    },
                    state_key: state.key.clone(),
                    reference: reference.clone(),
                    span: *span,
                });
            }
        }

        let lease_id = random_nonzero_id("publication source lease entropy failed")?;
        let owner_id = process_identity();
        if owner_id == [0; 32] {
            return Err(StoreError::State(
                "publication source process identity is zero",
            ));
        }
        let owner_incarnation = random_nonzero_id("publication source incarnation entropy failed")?;
        let authority_term = self.catalog_epoch();
        let status = self.acquire_source_lease(
            &lease_id,
            manifest_id,
            &owner_id,
            &owner_incarnation,
            authority_term,
            MAX_SOURCE_LEASE_NS,
            context,
        )?;
        if status.state != SourceLeaseState::Active {
            return Err(StoreError::State("publication source lease is not active"));
        }
        Ok(AuthenticatedPublicationSource {
            store: Arc::clone(self),
            manifest_id: *manifest_id,
            manifest_object: Arc::from(manifest_object),
            tenant_namespace: manifest.tenant_namespace,
            family: manifest.family,
            chunks,
            expected_bytes,
            lease_id,
            owner_id,
            owner_incarnation,
            authority_term,
            lease_expires_ns: status.expires_ns,
            released: false,
        })
    }

    /// Resolve an object through an authenticated manifest, pin its FD, and
    /// verify the complete chunk before returning stored bytes for HTTP/2.
    pub fn read_authenticated_chunk_object(
        self: &Arc<Self>,
        manifest_id: &Id32,
        object_key: &Id32,
        expected_ordinal: u64,
        context: &ValidationContext,
    ) -> Result<Vec<u8>, StoreError> {
        let (_pack, manifest) = self.authenticated_manifest_cached(manifest_id, context)?;
        let entry = manifest
            .states
            .iter()
            .zip(&manifest.realized_schema.states)
            .flat_map(|(state, schema)| {
                state
                    .chunks
                    .iter()
                    .zip(&schema.chunk_spans)
                    .map(move |(chunk, span)| (&state.key, chunk, span))
            })
            .enumerate()
            .find(|(ordinal, (_, chunk, _))| {
                *ordinal as u64 == expected_ordinal && &chunk.object_key == object_key
            })
            .map(|(_, entry)| entry)
            .ok_or(StoreError::NotFound)?;
        let (state_key, reference, span) = entry;
        read_validated_chunk_object(
            self,
            &manifest.tenant_namespace,
            &manifest.family,
            state_key,
            reference,
            span,
        )
    }
}

fn read_validated_chunk_object(
    store: &Arc<LocalStore>,
    tenant_namespace: &Id32,
    family: &RepresentationFamilyId,
    state_key: &StateKey,
    reference: &ChunkRef,
    span: &ChunkSpan,
) -> Result<Vec<u8>, StoreError> {
    let bytes = store.pin_chunk(reference)?.read_all()?;
    let chunk_keys = store.schedule(reference.key_epoch)?;
    decode_chunk(
        &bytes,
        reference,
        span,
        tenant_namespace,
        family,
        state_key,
        &chunk_keys,
    )?;
    Ok(bytes)
}

/// Plaintext sibling of `read_validated_chunk_object`: same pin, schedule,
/// and authenticated decode, returning the decoded state bytes.
fn read_validated_chunk_plaintext(
    store: &Arc<LocalStore>,
    tenant_namespace: &Id32,
    family: &RepresentationFamilyId,
    state_key: &StateKey,
    reference: &ChunkRef,
    span: &ChunkSpan,
) -> Result<Vec<u8>, StoreError> {
    let bytes = store.pin_chunk(reference)?.read_all()?;
    let chunk_keys = store.schedule(reference.key_epoch)?;
    Ok(decode_chunk(
        &bytes,
        reference,
        span,
        tenant_namespace,
        family,
        state_key,
        &chunk_keys,
    )?)
}
