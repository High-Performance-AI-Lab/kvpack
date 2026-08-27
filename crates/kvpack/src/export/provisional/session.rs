use super::*;

#[derive(Debug, Clone)]
pub(super) struct ProvisionalChunk {
    pub(super) stored: StoredStateChunk,
    pub(super) staged: ProvisionalStagedChunk,
}

pub struct ProvisionalExportSession {
    pub(super) store: Arc<LocalStore>,
    pub(super) declaration: ProvisionalExportDeclaration,
    pub(super) policy: WritePolicy,
    pub(super) keys: KeySchedule,
    pub(super) session_token: u64,
    pub(super) provenance: ProvisionalProvenance,
    pub(super) completed: Vec<StoredState>,
    pub(super) provisional_chunks: Vec<ProvisionalChunk>,
    pub(super) next_state: usize,
    pub(super) next_ordinal: u64,
    pub(super) replay: bool,
    pub(super) poisoned: bool,
    pub(super) terminal: bool,
    pub(super) origin: Instant,
    pub(super) begin_duration_ns: u64,
    pub(super) encryption_duration_ns: u64,
    pub(super) staging_duration_ns: u64,
    pub(super) staged_bytes: u64,
    pub(super) deduplicated_bytes: u64,
    pub(super) staged_chunk_count: u64,
    pub(super) deduplicated_chunk_count: u64,
    pub(super) write_intervals: Vec<ProvisionalIoIntervalV1>,
    pub(super) encryption_intervals: Vec<ProvisionalIoIntervalV1>,
}

impl ProvisionalExportSession {
    pub fn begin(
        store: Arc<LocalStore>,
        declaration: ProvisionalExportDeclaration,
        cut_policy: ExportCutPolicy,
        policy: WritePolicy,
    ) -> Result<Self, StoreError> {
        let origin = Instant::now();
        if cut_policy != ExportCutPolicy::production_v1() {
            return Err(StoreError::Expectation(
                "request cannot widen the immutable production cut policy",
            ));
        }
        if declaration.cached_token_count == 0
            || declaration.sealed_prompt_token_ids_sha256 == [0; 32]
            || declaration.source_declaration_digest == [0; 32]
            || policy.idempotency_key != declaration.source_declaration_digest
        {
            return Err(StoreError::Expectation(
                "provisional export declaration or BEGIN-derived idempotency is invalid",
            ));
        }
        let cached = usize::try_from(declaration.cached_token_count)
            .map_err(|_| StoreError::State("cached token count exceeds usize"))?;
        let placeholder = ExportDeclaration {
            semantic_model: declaration.semantic_model,
            input_tokens: vec![0; cached],
            auxiliary_inputs: declaration.auxiliary_inputs.clone(),
            family: declaration.family.clone(),
            states: declaration.states.clone(),
        };
        validate_export_declaration(&placeholder, &policy)?;
        let keys = store.schedule(store.key_epoch())?;
        let (_, placeholder_nodes) = kvpack_core::derive_input_cut(
            keys.prefix_key(),
            &store.tenant_namespace(),
            &declaration.semantic_model,
            &declaration.family,
            &placeholder.input_tokens,
            &declaration.auxiliary_inputs,
        )?;
        let expected_bytes = estimate_export_reservation(&placeholder, &placeholder_nodes)?;
        let intent_digest = provisional_intent_digest(&store, &declaration, &policy)?;
        let quiesced = {
            let stat = store.stat()?;
            stat.active_grants == 0
                && stat.active_leases == 0
                && stat.active_restores == 0
                && stat.active_source_leases == 0
                && stat.active_uploads == 0
        };
        let provenance = ProvisionalProvenance {
            source_wall_clock_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .min(u64::MAX as u128) as u64,
            clock_offset_ns: None,
            quiesced,
        };
        let (state, session_token, provenance) = store.begin_provisional_upload(
            &declaration.source_declaration_digest,
            UploadReservation {
                expected_bytes,
                publication_generation: policy.publication_generation,
                intent_digest,
                retention: policy.retention.with_physical_bytes(expected_bytes)?,
            },
            provenance,
        )?;
        Ok(Self {
            store,
            declaration,
            policy,
            keys,
            session_token,
            provenance,
            completed: Vec::new(),
            provisional_chunks: Vec::new(),
            next_state: 0,
            next_ordinal: 0,
            replay: state == UploadState::Published,
            poisoned: false,
            terminal: false,
            origin,
            begin_duration_ns: duration_ns(origin.elapsed()),
            encryption_duration_ns: 0,
            staging_duration_ns: 0,
            staged_bytes: 0,
            deduplicated_bytes: 0,
            staged_chunk_count: 0,
            deduplicated_chunk_count: 0,
            write_intervals: Vec::new(),
            encryption_intervals: Vec::new(),
        })
    }

    pub fn state_bounds(&self, key: &StateKey) -> Result<ExportStateBounds, StoreError> {
        let index = self
            .declaration
            .states
            .iter()
            .position(|state| &state.key == key)
            .ok_or(StoreError::Expectation(
                "state is not in the declared provisional inventory",
            ))?;
        let bytes_per_token = family_bytes_per_token(&self.declaration.family.states[index])?;
        let token_count = u64::from(self.declaration.cached_token_count);
        Ok(ExportStateBounds {
            token_count,
            bytes_per_token,
            plaintext_bytes: token_count
                .checked_mul(bytes_per_token)
                .ok_or(StoreError::State("provisional state source bound overflow"))?,
        })
    }

    /// Nanoseconds elapsed on the same monotonic clock used by the receipt's
    /// encryption and write intervals. Qualification workers use one sample
    /// to translate those bounded intervals into their own trial-relative
    /// timeline; no raw monotonic epoch is exposed or serialized.
    pub fn elapsed_ns(&self) -> u64 {
        elapsed_ns(self.origin)
    }

    /// Provenance captured at begin (also persisted on the upload row and
    /// hashed into the seal digest).
    pub fn provenance(&self) -> ProvisionalProvenance {
        self.provenance
    }

    /// Record the producer/consumer clock offset. Only valid before the
    /// first staged chunk so the offset is stable for resume and replay.
    pub fn record_clock_offset_ns(&mut self, clock_offset_ns: u64) -> Result<(), StoreError> {
        if self.poisoned || self.terminal || self.next_ordinal != 0 {
            return Err(StoreError::State(
                "provisional clock offset must be recorded before staging",
            ));
        }
        self.store.record_provisional_clock_offset(
            &self.declaration.source_declaration_digest,
            self.session_token,
            clock_offset_ns,
        )?;
        self.provenance.clock_offset_ns = Some(clock_offset_ns);
        Ok(())
    }

    pub fn cancel(mut self) -> Result<(), StoreError> {
        self.store.cancel_provisional_upload(
            &self.declaration.source_declaration_digest,
            self.session_token,
        )?;
        self.terminal = true;
        Ok(())
    }
}

impl Drop for ProvisionalExportSession {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = self.store.cancel_provisional_upload(
                &self.declaration.source_declaration_digest,
                self.session_token,
            );
        }
    }
}
