use super::*;

pub struct PinnedChunk {
    store: Arc<LocalStore>,
    pin_id: Option<Id32>,
    file: fs::File,
    expected_bytes: usize,
    object_key: Id32,
}

pub(crate) struct RetainedPin {
    store: Arc<LocalStore>,
    pin_id: Option<Id32>,
    bytes: u64,
}

impl PinnedChunk {
    pub fn read_all(mut self) -> Result<Vec<u8>, StoreError> {
        self.read_bytes()
    }

    pub(crate) fn read_all_retained(mut self) -> Result<(Vec<u8>, RetainedPin), StoreError> {
        let bytes = self.read_bytes()?;
        let pin_id = self
            .pin_id
            .take()
            .ok_or(StoreError::State("pinned chunk was already transferred"))?;
        Ok((
            bytes,
            RetainedPin {
                store: Arc::clone(&self.store),
                pin_id: Some(pin_id),
                bytes: self.expected_bytes as u64,
            },
        ))
    }

    pub(crate) fn into_retained_file(mut self) -> Result<(fs::File, RetainedPin), StoreError> {
        let file = self
            .file
            .try_clone()
            .map_err(io_error("duplicate pinned chunk descriptor"))?;
        let pin_id = self
            .pin_id
            .take()
            .ok_or(StoreError::State("pinned chunk was already transferred"))?;
        Ok((
            file,
            RetainedPin {
                store: Arc::clone(&self.store),
                pin_id: Some(pin_id),
                bytes: self.expected_bytes as u64,
            },
        ))
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, StoreError> {
        if let Some(bytes) = direct::maybe_read_chunk_bypass(
            &self.store.chunk_path(&self.object_key),
            self.expected_bytes,
            direct::direct_read_enabled(),
        ) {
            return Ok(bytes);
        }
        let mut bytes = Vec::with_capacity(self.expected_bytes);
        (&mut self.file)
            .take(self.expected_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(io_error("read pinned chunk"))?;
        if bytes.len() != self.expected_bytes {
            return Err(StoreError::Authentication("pinned chunk length changed"));
        }
        Ok(bytes)
    }
}

impl Drop for PinnedChunk {
    fn drop(&mut self) {
        release_pin(&self.store, self.pin_id.take());
    }
}

impl RetainedPin {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn id(&self) -> Result<Id32, StoreError> {
        self.pin_id
            .ok_or(StoreError::State("retained source pin has no identity"))
    }

    pub(crate) fn take_id(&mut self) -> Option<Id32> {
        self.pin_id.take()
    }
}

/// Release a set of retained pins plus not-yet-consumed pin identities in one
/// catalog transaction instead of one autocommit DELETE per pin.  Best
/// effort, exactly like dropping the pins individually.
pub(crate) fn release_restore_pin_batch(
    store: &LocalStore,
    pins: &mut [RetainedPin],
    pending_ids: &[Id32],
) {
    let mut pin_ids: Vec<Id32> = pins.iter_mut().filter_map(RetainedPin::take_id).collect();
    pin_ids.extend_from_slice(pending_ids);
    let _ = store.release_retained_source_pins(&pin_ids);
}

impl Drop for RetainedPin {
    fn drop(&mut self) {
        release_pin(&self.store, self.pin_id.take());
    }
}

fn release_pin(store: &LocalStore, pin_id: Option<Id32>) {
    let Some(pin_id) = pin_id else {
        return;
    };
    if let Ok(connection) = store.lock_catalog() {
        let _ = connection.execute(
            "DELETE FROM pins WHERE tenant=?1 AND pin_id=?2",
            params![store.tenant_namespace.as_slice(), pin_id.as_slice()],
        );
    }
}

impl LocalStore {
    /// Release source-pin identities previously returned by a prepared scatter
    /// transfer. This is idempotent so an agent can replace descriptors after
    /// restart without leaving stale durable pins behind.
    pub fn release_retained_source_pins(&self, pin_ids: &[Id32]) -> Result<u64, StoreError> {
        if pin_ids.len() > 1_000_000 {
            return Err(StoreError::Expectation(
                "retained source pin release exceeds the bounded count",
            ));
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction()?;
        let mut released = 0u64;
        for pin_id in pin_ids {
            released = released
                .checked_add(transaction.execute(
                    "DELETE FROM pins WHERE tenant=?1 AND pin_id=?2",
                    params![self.tenant_namespace.as_slice(), pin_id.as_slice()],
                )? as u64)
                .ok_or(StoreError::State("released source pin count overflow"))?;
        }
        transaction.commit()?;
        Ok(released)
    }

    pub(crate) fn pin_chunk(
        self: &Arc<Self>,
        reference: &ChunkRef,
    ) -> Result<PinnedChunk, StoreError> {
        let mut pin_id = [0u8; 32];
        getrandom::fill(&mut pin_id).map_err(|_| StoreError::State("pin entropy failed"))?;
        self.pin_chunk_with_id(reference, pin_id)
    }

    pub(crate) fn pin_chunk_with_id(
        self: &Arc<Self>,
        reference: &ChunkRef,
        pin_id: Id32,
    ) -> Result<PinnedChunk, StoreError> {
        if pin_id == [0; 32] {
            return Err(StoreError::Expectation(
                "source pin identity must be nonzero",
            ));
        }
        self.pin_chunk_rows(std::slice::from_ref(reference), &[pin_id])?;
        let file = match self.open_chunk_file(reference) {
            Ok(file) => file,
            Err(error) => {
                let _ = self.release_retained_source_pins(&[pin_id]);
                return Err(error);
            }
        };
        let pinned = PinnedChunk {
            store: Arc::clone(self),
            pin_id: Some(pin_id),
            file,
            expected_bytes: reference.object_bytes as usize,
            object_key: reference.object_key,
        };
        let metadata = pinned
            .file
            .metadata()
            .map_err(io_error("inspect pinned chunk"))?;
        if !metadata.is_file() || metadata.len() != reference.object_bytes as u64 {
            return Err(StoreError::Authentication("pinned chunk metadata mismatch"));
        }
        self.record_chunk_access(
            reference.object_key,
            reference.object_bytes as u64,
            reference.plaintext_bytes as u64,
        )?;
        Ok(pinned)
    }

    /// Insert one pins row per reference in a single catalog transaction.
    fn pin_chunk_rows(&self, references: &[ChunkRef], pin_ids: &[Id32]) -> Result<(), StoreError> {
        let owner_start = process_identity();
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for (reference, pin_id) in references.iter().zip(pin_ids) {
            // Mirror find_chunk's tombstone predicate: a tombstoned chunk
            // must never acquire a pin, even in the narrow window where its
            // catalog row is still AVAILABLE (audit N10).
            let exists: Option<u64> = transaction.query_row("SELECT object_bytes FROM chunks WHERE tenant=?1 AND object_key=?2 AND object_digest=?3 AND location_state='AVAILABLE' AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=chunks.tenant AND t.object_kind='chunk' AND t.object_id=chunks.object_key)", params![self.tenant_namespace.as_slice(), reference.object_key.as_slice(), reference.object_digest.as_slice()], |row| row.get(0)).optional()?;
            if exists != Some(reference.object_bytes as u64) {
                return Err(StoreError::Authentication(
                    "chunk catalog reference mismatch",
                ));
            }
            transaction.execute("INSERT INTO pins(tenant,pin_id,object_key,owner_pid,owner_start,created_ns) VALUES(?1,?2,?3,?4,?5,?6)", params![self.tenant_namespace.as_slice(), pin_id.as_slice(), reference.object_key.as_slice(), std::process::id(), owner_start.as_slice(), now_ns()])?;
        }
        transaction.commit()?;
        Ok(())
    }

    /// Acquire one freshly-generated source pin per reference in a single
    /// catalog transaction.  The caller reads each chunk with
    /// `read_pinned_chunk_object` and owns every returned identity.
    pub(crate) fn pin_chunks_batch(
        self: &Arc<Self>,
        references: &[ChunkRef],
    ) -> Result<Vec<Id32>, StoreError> {
        let mut pin_ids = Vec::with_capacity(references.len());
        for _ in references {
            pin_ids.push(random_nonzero_id("pin entropy failed")?);
        }
        self.pin_chunk_rows(references, &pin_ids)?;
        Ok(pin_ids)
    }

    /// Acquire caller-identified source pins (deterministic scatter transfer
    /// identities) in a single catalog transaction, then open every chunk FD.
    /// Any failure releases the complete batch in one transaction.
    pub(crate) fn pin_chunks_with_ids(
        self: &Arc<Self>,
        entries: &[(ChunkRef, Id32)],
    ) -> Result<Vec<PinnedChunk>, StoreError> {
        if entries.iter().any(|(_, pin_id)| *pin_id == [0; 32]) {
            return Err(StoreError::Expectation(
                "source pin identity must be nonzero",
            ));
        }
        let references: Vec<ChunkRef> = entries
            .iter()
            .map(|(reference, _)| reference.clone())
            .collect();
        let pin_ids: Vec<Id32> = entries.iter().map(|(_, pin_id)| *pin_id).collect();
        let result = (|| {
            self.pin_chunk_rows(&references, &pin_ids)?;
            let mut pinned = Vec::with_capacity(entries.len());
            for (reference, pin_id) in entries {
                let file = self.open_chunk_file(reference)?;
                pinned.push(PinnedChunk {
                    store: Arc::clone(self),
                    pin_id: Some(*pin_id),
                    file,
                    expected_bytes: reference.object_bytes as usize,
                    object_key: reference.object_key,
                });
            }
            for (chunk, (reference, _)) in pinned.iter().zip(entries) {
                let metadata = chunk
                    .file
                    .metadata()
                    .map_err(io_error("inspect pinned chunk"))?;
                if !metadata.is_file() || metadata.len() != reference.object_bytes as u64 {
                    return Err(StoreError::Authentication("pinned chunk metadata mismatch"));
                }
                self.record_chunk_access(
                    reference.object_key,
                    reference.object_bytes as u64,
                    reference.plaintext_bytes as u64,
                )?;
            }
            Ok(pinned)
        })();
        if result.is_err() {
            let _ = self.release_retained_source_pins(&pin_ids);
        }
        result
    }

    /// Read a chunk whose pin row already exists (acquired through
    /// `pin_chunks_batch`) without a catalog round trip.  On any failure the
    /// pin row is released again.
    pub(crate) fn read_pinned_chunk_object(
        self: &Arc<Self>,
        reference: &ChunkRef,
        pin_id: Id32,
    ) -> Result<(Vec<u8>, RetainedPin), StoreError> {
        let file = match self.open_chunk_file(reference) {
            Ok(file) => file,
            Err(error) => {
                let _ = self.release_retained_source_pins(&[pin_id]);
                return Err(error);
            }
        };
        let pinned = PinnedChunk {
            store: Arc::clone(self),
            pin_id: Some(pin_id),
            file,
            expected_bytes: reference.object_bytes as usize,
            object_key: reference.object_key,
        };
        let metadata = pinned
            .file
            .metadata()
            .map_err(io_error("inspect pinned chunk"))?;
        if !metadata.is_file() || metadata.len() != reference.object_bytes as u64 {
            return Err(StoreError::Authentication("pinned chunk metadata mismatch"));
        }
        self.record_chunk_access(
            reference.object_key,
            reference.object_bytes as u64,
            reference.plaintext_bytes as u64,
        )?;
        pinned.read_all_retained()
    }

    fn open_chunk_file(&self, reference: &ChunkRef) -> Result<fs::File, StoreError> {
        let path = self.chunk_path(&reference.object_key);
        match rustix::fs::open(
            &path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        ) {
            Ok(fd) => Ok(fs::File::from(fd)),
            Err(errno) if errno == rustix::io::Errno::NOENT => Err(StoreError::NotFound),
            Err(errno) => Err(StoreError::Io {
                op: "open pinned chunk",
                source: std::io::Error::from(errno),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use kvpack_core::{
        CacheKind, Codec, DType, FamilyState, Layout, RepresentationFamilyId, RepresentationMode,
        SemanticModelId, StateKey, StaticDimension, TokenAxisRule,
    };

    use crate::{
        ExportCutPolicy, ExportDeclaration, ExportSession, ExportStateDeclaration, StoreConfig,
        WritePolicy,
    };

    use super::*;

    fn id(value: u8) -> Id32 {
        [value; 32]
    }

    fn semantic() -> SemanticModelId {
        SemanticModelId {
            weights_config: id(1),
            adapters: id(2),
            tokenizer_template: id(3),
            position_semantics: id(4),
            qualified_math: id(5),
        }
    }

    fn family() -> RepresentationFamilyId {
        RepresentationFamilyId {
            engine_cache_abi: id(6),
            mode: RepresentationMode::Native,
            page_size_tokens: 256,
            topology: id(7),
            shard_map: id(8),
            states: vec![FamilyState {
                key: StateKey::new(0, "k"),
                cache_kind: CacheKind::OrdinaryKv,
                dtype: DType::U8,
                codec: Codec::Raw,
                codec_version: 1,
                layout: Layout::Contiguous,
                token_axis_rule: TokenAxisRule::Direct,
                token_axis: 0,
                elements_per_token: 4,
                dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(4)],
                dependencies: vec![],
            }],
        }
    }

    fn publish_one_chunk(store: &Arc<LocalStore>) -> Id32 {
        let declaration = ExportDeclaration {
            semantic_model: semantic(),
            input_tokens: (0..256u32).collect(),
            auxiliary_inputs: Vec::new(),
            family: family(),
            states: vec![ExportStateDeclaration {
                key: StateKey::new(0, "k"),
                strides: vec![4, 1],
                atomic_group: 1,
            }],
        };
        let policy = WritePolicy::exact_qualified(id(50), semantic(), &family()).unwrap();
        let mut session = ExportSession::begin(
            Arc::clone(store),
            declaration,
            ExportCutPolicy::production_v1(),
            policy,
        )
        .unwrap();
        session
            .next_state(StateKey::new(0, "k"))
            .unwrap()
            .write_source(&mut Cursor::new(vec![7u8; 256 * 4]))
            .unwrap();
        session.commit().unwrap().exact_final.manifest_id
    }

    fn only_chunk_reference(store: &LocalStore) -> ChunkRef {
        let connection = store.lock_catalog().unwrap();
        connection
            .query_row(
                "SELECT chunk_id,object_key,object_digest,key_epoch,plaintext_bytes,object_bytes FROM chunks",
                [],
                |row| {
                    Ok(ChunkRef {
                        chunk_id: row.get::<_, Vec<u8>>(0)?.try_into().unwrap(),
                        object_key: row.get::<_, Vec<u8>>(1)?.try_into().unwrap(),
                        object_digest: row.get::<_, Vec<u8>>(2)?.try_into().unwrap(),
                        key_epoch: row.get(3)?,
                        plaintext_bytes: row.get(4)?,
                        object_bytes: row.get(5)?,
                    })
                },
            )
            .unwrap()
    }

    #[test]
    fn tombstoned_chunk_never_acquires_a_pin() {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        crate::create_store_key_random(&key_path, temp.path()).unwrap();
        let store = Arc::new(
            LocalStore::open(
                StoreConfig {
                    object_root: temp.path().join("objects"),
                    catalog_path: temp.path().join("catalog/catalog.sqlite"),
                    operator_tenant_id: b"pin-tombstone-tenant".to_vec(),
                    key_epoch: 1,
                    minimum_readable_key_epoch: 1,
                    catalog_epoch: 1,
                    quota_bytes: 1 << 30,
                    staging_quota_bytes: 1 << 30,
                    endurance_bytes_per_five_minutes: 1 << 30,
                },
                crate::load_store_key(&key_path, temp.path()).unwrap(),
            )
            .unwrap(),
        );
        publish_one_chunk(&store);
        let reference = only_chunk_reference(&store);

        // Baseline: an available, untombstoned chunk pins cleanly.
        let pinned = store.pin_chunk(&reference).unwrap();
        drop(pinned);

        // Tombstone the chunk (the same predicate find_chunk applies); the
        // pin path must now refuse it even though the row still reads
        // AVAILABLE (audit N10).
        store
            .lock_catalog()
            .unwrap()
            .execute(
                "INSERT INTO tombstones(tenant,object_kind,object_id,catalog_epoch,created_ns) VALUES(?1,'chunk',?2,?3,0)",
                params![
                    store.tenant_namespace.as_slice(),
                    reference.object_key.as_slice(),
                    store.catalog_epoch()
                ],
            )
            .unwrap();
        assert!(matches!(
            store.pin_chunk(&reference),
            Err(StoreError::Authentication(
                "chunk catalog reference mismatch"
            ))
        ));
    }
}
