use super::*;

#[derive(Debug, Clone)]
pub(super) struct StoredStateChunk {
    pub(super) reference: ChunkRef,
    pub(super) span: ChunkSpan,
}

#[derive(Debug, Clone)]
pub(super) struct StoredState {
    pub(super) key: StateKey,
    pub(super) chunks: Vec<StoredStateChunk>,
}

pub struct ExportSession {
    pub(super) store: Arc<LocalStore>,
    pub(super) semantic_model: SemanticModelId,
    pub(super) family: RepresentationFamilyId,
    pub(super) declarations: Vec<ExportStateDeclaration>,
    prefix_nodes: Vec<PrefixNode>,
    pub(super) auxiliary_root: Id32,
    pub(super) final_token_count: u64,
    pub(super) policy: WritePolicy,
    keys: KeySchedule,
    pub(super) completed: Vec<StoredState>,
    pub(super) next_state: usize,
    replay: bool,
    pub(super) poisoned: bool,
    committed: bool,
}

/// Outcome of staging one export chunk: an already-durable dedup winner, or a
/// freshly encoded object queued for the next batched put.
pub(super) enum ExportStagedChunk {
    Deduplicated(ChunkRef),
    Encoded(ChunkObject),
}

impl ExportSession {
    pub fn begin(
        store: Arc<LocalStore>,
        declaration: ExportDeclaration,
        cut_policy: ExportCutPolicy,
        policy: WritePolicy,
    ) -> Result<Self, StoreError> {
        if cut_policy != ExportCutPolicy::production_v1() {
            return Err(StoreError::Expectation(
                "request cannot widen the immutable production cut policy",
            ));
        }
        validate_export_declaration(&declaration, &policy)?;
        let keys = store.schedule(store.key_epoch())?;
        let (final_cut, prefix_nodes) = kvpack_core::derive_input_cut(
            keys.prefix_key(),
            &store.tenant_namespace(),
            &declaration.semantic_model,
            &declaration.family,
            &declaration.input_tokens,
            &declaration.auxiliary_inputs,
        )?;
        if prefix_nodes.is_empty()
            || prefix_nodes.last().map(|node| node.id) != Some(final_cut.token_root)
            || prefix_nodes.last().map(|node| node.token_count) != Some(final_cut.token_count)
            || prefix_nodes.len() > MAX_CHUNKS_PER_STATE
        {
            return Err(StoreError::State(
                "derived cut inventory is empty, inconsistent, or exceeds its bound",
            ));
        }
        let expected_bytes = estimate_export_reservation(&declaration, &prefix_nodes)?;
        let intent_digest = export_intent_digest(&store, &declaration, final_cut, &policy)?;
        let upload_state = store.reserve_upload(
            &policy.idempotency_key,
            UploadReservation {
                expected_bytes,
                publication_generation: policy.publication_generation,
                intent_digest,
                retention: policy.retention.with_physical_bytes(expected_bytes)?,
            },
        )?;
        let replay = upload_state == UploadState::Published;
        if !replay {
            store.mark_receiving(&policy.idempotency_key)?;
        }
        Ok(Self {
            store,
            semantic_model: declaration.semantic_model,
            family: declaration.family,
            declarations: declaration.states,
            prefix_nodes,
            auxiliary_root: final_cut.auxiliary_input_root,
            final_token_count: final_cut.token_count,
            policy,
            keys,
            completed: Vec::new(),
            next_state: 0,
            replay,
            poisoned: false,
            committed: false,
        })
    }

    pub fn state_bounds(&self, key: &StateKey) -> Result<ExportStateBounds, StoreError> {
        let index = self
            .declarations
            .iter()
            .position(|state| &state.key == key)
            .ok_or(StoreError::Expectation(
                "state is not in the declared export inventory",
            ))?;
        self.bounds_at(index)
    }

    fn bounds_at(&self, index: usize) -> Result<ExportStateBounds, StoreError> {
        let bytes_per_token = family_bytes_per_token(&self.family.states[index])?;
        Ok(ExportStateBounds {
            token_count: self.final_token_count,
            bytes_per_token,
            plaintext_bytes: self
                .final_token_count
                .checked_mul(bytes_per_token)
                .ok_or(StoreError::State("state source bound overflow"))?,
        })
    }

    pub fn next_state(
        &mut self,
        expected_state_key: StateKey,
    ) -> Result<ExportStateWriter<'_>, StoreError> {
        if self.poisoned {
            return Err(StoreError::Poisoned("export session is poisoned"));
        }
        let declaration = self
            .declarations
            .get(self.next_state)
            .ok_or(StoreError::State(
                "all declared export states were already written",
            ))?
            .clone();
        if declaration.key != expected_state_key {
            self.poisoned = true;
            return Err(StoreError::State(
                "next export state does not match canonical family order",
            ));
        }
        let bounds = self.bounds_at(self.next_state)?;
        let maximum_chunk_tokens = (MAX_CHUNK_PLAINTEXT as u64 / bounds.bytes_per_token).max(1);
        let first_chunk_bytes = next_chunk_bytes(
            0,
            bounds.token_count,
            bounds.bytes_per_token,
            maximum_chunk_tokens,
        )?;
        Ok(ExportStateWriter {
            session: self,
            declaration,
            buffer: Vec::with_capacity(first_chunk_bytes),
            chunks: Vec::new(),
            pending: Vec::new(),
            pending_bytes: 0,
            flushed_bytes: 0,
            expected_bytes: bounds.plaintext_bytes,
            bytes_per_token: bounds.bytes_per_token,
            maximum_chunk_tokens,
            finished: false,
        })
    }

    pub(super) fn store_chunk(
        &self,
        state_key: &StateKey,
        plaintext: &[u8],
        span: ChunkSpan,
    ) -> Result<ExportStagedChunk, StoreError> {
        let content_id = chunk_id(
            self.keys.chunk_identity_key(),
            &self.store.tenant_namespace(),
            &self.family,
            state_key,
            &span,
            plaintext,
        )?;
        match self.store.find_chunk(&content_id, self.store.key_epoch())? {
            Some(existing) => {
                if existing.chunk_id != content_id
                    || existing.plaintext_bytes as usize != plaintext.len()
                {
                    return Err(StoreError::Authentication(
                        "deduplicated export chunk metadata mismatch",
                    ));
                }
                Ok(ExportStagedChunk::Deduplicated(existing))
            }
            None if self.replay => Err(StoreError::Authentication(
                "published export replay contains changed or unavailable chunk bytes",
            )),
            None => {
                let object = encode_chunk_with_content_id(
                    plaintext,
                    &ChunkEncoding {
                        tenant_namespace: self.store.tenant_namespace(),
                        family: &self.family,
                        state_key,
                        span,
                        key_epoch: self.store.key_epoch(),
                        encrypt: self.policy.encrypt_chunks,
                        stats_sidecar: None,
                    },
                    &self.keys,
                    Some(&content_id),
                )?;
                Ok(ExportStagedChunk::Encoded(object))
            }
        }
    }

    pub fn commit(mut self) -> Result<PublishedCutSet, StoreError> {
        if self.poisoned {
            return Err(StoreError::Poisoned("export session is poisoned"));
        }
        if self.next_state != self.declarations.len() {
            self.poisoned = true;
            return Err(StoreError::State(
                "not every declared export state was written",
            ));
        }

        self.store.sync_export_partial_cleanup()?;

        let mut manifests = Vec::with_capacity(self.prefix_nodes.len());
        let mut encoded = Vec::with_capacity(self.prefix_nodes.len());
        let mut previous = None;
        for node in &self.prefix_nodes {
            let kind = match previous {
                Some((parent, parent_cut, depth)) if depth < MAX_DELTA_DEPTH => {
                    ManifestKind::Delta {
                        parent,
                        parent_cut,
                        depth: depth + 1,
                    }
                }
                None | Some(_) => ManifestKind::Full,
            };
            let manifest = self.manifest_for_cut(*node, kind)?;
            let object = if self.replay {
                let manifest_id = kvpack_core::manifest_id(&manifest.encode_canonical()?);
                EncodedPack {
                    bytes: self.store.read_authenticated_manifest_object(
                        &manifest_id,
                        &self.policy.validation_context(),
                    )?,
                    manifest_id,
                }
            } else {
                encode_authenticated_pack(
                    &manifest,
                    &self.keys,
                    self.policy.encrypt_manifest,
                    &self.policy.validation_context(),
                )?
            };
            previous = Some((
                object.manifest_id,
                manifest.input_cut,
                manifest.realized_schema.kind.depth(),
            ));
            manifests.push(manifest);
            encoded.push(object);
        }
        let pending: Vec<_> = encoded
            .iter()
            .zip(&manifests)
            .zip(&self.prefix_nodes)
            .enumerate()
            .map(|(index, ((encoded, manifest), node))| PendingManifest {
                encoded,
                manifest,
                prefix_node: *node,
                exact_final: index + 1 == self.prefix_nodes.len(),
            })
            .collect();
        let published = self
            .store
            .publish_manifest_batch(&self.policy.idempotency_key, &pending)?;
        let cuts: Vec<_> = published
            .into_iter()
            .zip(manifests)
            .map(|(published, manifest)| published_cut(published, manifest))
            .collect();
        let exact_final = cuts
            .last()
            .cloned()
            .ok_or(StoreError::State("export produced no final cut"))?;
        let checkpoints = cuts
            .into_iter()
            .zip(&self.prefix_nodes)
            .filter(|(_, node)| node.reusable)
            .map(|(cut, _)| cut)
            .collect();
        self.committed = true;
        Ok(PublishedCutSet {
            checkpoints,
            exact_final,
        })
    }

    pub fn cancel(mut self) -> Result<(), StoreError> {
        self.store.abort_upload(&self.policy.idempotency_key)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for ExportSession {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.store.abort_upload(&self.policy.idempotency_key);
        }
    }
}

pub(super) fn published_cut(published: PublishedArtifact, manifest: CutManifest) -> PublishedCut {
    PublishedCut {
        manifest_id: published.manifest_id,
        input_cut: manifest.input_cut,
        realized_schema: manifest.realized_schema,
        restored_bytes: published.restored_bytes,
        publication_generation: published.publication_generation,
    }
}
