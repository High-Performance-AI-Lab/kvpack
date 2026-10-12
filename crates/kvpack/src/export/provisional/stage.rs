use super::*;

impl ProvisionalExportSession {
    /// Encrypt and stage one complete state in canonical family order. The
    /// supplied digest is the authenticated SHA-256 from its live layer frame.
    pub fn stage_state(
        &mut self,
        expected_state_key: StateKey,
        authenticated_state_sha256: Id32,
        source: &mut impl Read,
    ) -> Result<ProvisionalStateReceipt, StoreError> {
        let result = self.stage_state_inner(expected_state_key, authenticated_state_sha256, source);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn stage_state_inner(
        &mut self,
        expected_state_key: StateKey,
        authenticated_state_sha256: Id32,
        source: &mut impl Read,
    ) -> Result<ProvisionalStateReceipt, StoreError> {
        let started = Instant::now();
        if self.poisoned || self.terminal {
            return Err(StoreError::Poisoned(
                "provisional export session is not writable",
            ));
        }
        let declaration = self
            .declaration
            .states
            .get(self.next_state)
            .ok_or(StoreError::State(
                "all provisional export states were already staged",
            ))?
            .clone();
        if declaration.key != expected_state_key {
            self.poisoned = true;
            return Err(StoreError::State(
                "provisional state does not match canonical K/V family order",
            ));
        }
        let bounds = self.state_bounds(&expected_state_key)?;
        let maximum_chunk_tokens = (MAX_CHUNK_PLAINTEXT as u64 / bounds.bytes_per_token).max(1);
        let mut plaintext_offset = 0u64;
        let mut chunks = Vec::new();
        let state_ordinal_start = self.next_ordinal;
        let staged_before = self.staged_bytes;
        let deduplicated_before = self.deduplicated_bytes;
        let mut state_hash = Sha256::new();
        // Chunk bytes still stage and fsync per file, but the chunk-ledger
        // rows and durable cursor advance commit in ONE catalog transaction
        // per batch (per state, or per CHUNK_PUT_BATCH_BYTES staged).
        let mode = self.store.begin_provisional_stage_batch(
            &self.declaration.source_declaration_digest,
            self.session_token,
        )?;
        let mut pending_ledger: Vec<(u64, ChunkRef)> = Vec::new();
        let mut pending_ledger_bytes = 0u64;
        while plaintext_offset < bounds.plaintext_bytes {
            let token_start = plaintext_offset / bounds.bytes_per_token;
            let chunk_bytes = next_chunk_bytes(
                token_start,
                bounds.token_count,
                bounds.bytes_per_token,
                maximum_chunk_tokens,
            )?;
            let mut plaintext = vec![0u8; chunk_bytes];
            read_exact_state(source, &mut plaintext)?;
            state_hash.update(&plaintext);
            let token_count = plaintext.len() as u64 / bounds.bytes_per_token;
            let span = ChunkSpan {
                token_start,
                token_count,
                plaintext_offset,
                plaintext_bytes: u32::try_from(plaintext.len())
                    .map_err(|_| StoreError::State("provisional chunk plaintext exceeds u32"))?,
            };
            let content_id = chunk_id(
                self.keys.chunk_identity_key(),
                &self.store.tenant_namespace(),
                &self.declaration.family,
                &declaration.key,
                &span,
                &plaintext,
            )?;
            let encryption_start = elapsed_ns(self.origin);
            let encryption_clock = Instant::now();
            let object = encode_chunk_with_content_id(
                &plaintext,
                &ChunkEncoding {
                    tenant_namespace: self.store.tenant_namespace(),
                    family: &self.declaration.family,
                    state_key: &declaration.key,
                    span,
                    key_epoch: self.store.key_epoch(),
                    encrypt: self.policy.encrypt_chunks,
                    stats_sidecar: None,
                },
                &self.keys,
                Some(&content_id),
            )?;
            let encryption_ns = duration_ns(encryption_clock.elapsed());
            self.encryption_duration_ns = self
                .encryption_duration_ns
                .checked_add(encryption_ns)
                .ok_or(StoreError::State("provisional encryption timing overflow"))?;
            self.encryption_intervals.push(ProvisionalIoIntervalV1 {
                start_ns: encryption_start,
                end_ns: elapsed_ns(self.origin),
                bytes: object.bytes.len() as u64,
            });
            if object.chunk_id != content_id {
                self.poisoned = true;
                return Err(StoreError::Authentication(
                    "provisional encoded chunk identity changed",
                ));
            }
            let write_start = elapsed_ns(self.origin);
            let write_clock = Instant::now();
            let staged = self.store.stage_provisional_chunk_file(
                &self.declaration.source_declaration_digest,
                &mode,
                self.next_ordinal,
                &object,
            )?;
            let write_ns = duration_ns(write_clock.elapsed());
            self.staging_duration_ns = self
                .staging_duration_ns
                .checked_add(write_ns)
                .ok_or(StoreError::State("provisional staging timing overflow"))?;
            self.write_intervals.push(ProvisionalIoIntervalV1 {
                start_ns: write_start,
                end_ns: elapsed_ns(self.origin),
                bytes: staged.staged_bytes,
            });
            self.staged_bytes = self
                .staged_bytes
                .checked_add(staged.staged_bytes)
                .ok_or(StoreError::State("provisional staged byte total overflow"))?;
            self.deduplicated_bytes = self
                .deduplicated_bytes
                .checked_add(staged.deduplicated_bytes)
                .ok_or(StoreError::State(
                    "provisional deduplicated byte total overflow",
                ))?;
            self.staged_chunk_count += u64::from(staged.staged_bytes != 0);
            self.deduplicated_chunk_count += u64::from(staged.deduplicated_bytes != 0);
            let stored = StoredStateChunk {
                reference: staged.reference.clone(),
                span,
            };
            if let ProvisionalStageMode::Receiving { cursor } = mode {
                if self.next_ordinal >= cursor {
                    pending_ledger.push((self.next_ordinal, staged.reference.clone()));
                    pending_ledger_bytes = pending_ledger_bytes
                        .checked_add(staged.staged_bytes)
                        .ok_or(StoreError::State("provisional ledger batch byte overflow"))?;
                }
            }
            chunks.push(stored.clone());
            self.provisional_chunks
                .push(ProvisionalChunk { stored, staged });
            self.next_ordinal = self
                .next_ordinal
                .checked_add(1)
                .ok_or(StoreError::State("provisional chunk ordinal overflow"))?;
            plaintext_offset = plaintext_offset
                .checked_add(plaintext.len() as u64)
                .ok_or(StoreError::State("provisional state size overflow"))?;
            if pending_ledger_bytes >= CHUNK_PUT_BATCH_BYTES {
                self.store.commit_provisional_stage_batch(
                    &self.declaration.source_declaration_digest,
                    self.session_token,
                    &pending_ledger,
                )?;
                pending_ledger.clear();
                pending_ledger_bytes = 0;
            }
        }
        self.store.commit_provisional_stage_batch(
            &self.declaration.source_declaration_digest,
            self.session_token,
            &pending_ledger,
        )?;
        ensure_source_ended(source)?;
        let actual_sha256: Id32 = state_hash.finalize().into();
        if actual_sha256 != authenticated_state_sha256 {
            self.poisoned = true;
            return Err(StoreError::Authentication(
                "provisional state source disagrees with its authenticated layer SHA-256",
            ));
        }
        if chunks.is_empty() || chunks.len() > MAX_CHUNKS_PER_STATE {
            self.poisoned = true;
            return Err(StoreError::State(
                "provisional state chunk inventory is empty or exceeds its bound",
            ));
        }
        if !self.replay {
            self.store
                .sync_provisional_state(&self.declaration.source_declaration_digest)?;
        }
        self.completed.push(StoredState {
            key: declaration.key.clone(),
            chunks,
        });
        self.next_state += 1;
        Ok(ProvisionalStateReceipt {
            state: declaration.key,
            duration_ns: duration_ns(started.elapsed()),
            plaintext_bytes: bounds.plaintext_bytes,
            chunk_count: self.next_ordinal - state_ordinal_start,
            staged_bytes: self.staged_bytes - staged_before,
            deduplicated_bytes: self.deduplicated_bytes - deduplicated_before,
        })
    }
}
