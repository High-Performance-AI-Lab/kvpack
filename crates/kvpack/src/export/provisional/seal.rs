use super::*;

impl ProvisionalExportSession {
    pub fn seal_and_publish(
        mut self,
        seal: ProvisionalExportSeal,
    ) -> Result<ProvisionalExportReceipt, StoreError> {
        let mut promoted = Vec::new();
        let result = self.seal_and_publish_inner(seal, &mut promoted);
        match result {
            Ok(receipt) => Ok(receipt),
            Err(error) => {
                self.poisoned = true;
                let _ = self.store.cleanup_provisional_promotion(
                    &self.declaration.source_declaration_digest,
                    &promoted,
                );
                let _ = self.store.cancel_provisional_upload(
                    &self.declaration.source_declaration_digest,
                    self.session_token,
                );
                self.terminal = true;
                Err(error)
            }
        }
    }

    fn seal_and_publish_inner(
        &mut self,
        seal: ProvisionalExportSeal,
        promoted: &mut Vec<ProvisionalPromotedChunk>,
    ) -> Result<ProvisionalExportReceipt, StoreError> {
        if self.poisoned || self.terminal {
            return Err(StoreError::Poisoned(
                "provisional export session cannot be sealed",
            ));
        }
        if self.next_state != self.declaration.states.len()
            || self.completed.len() != self.declaration.family.states.len()
        {
            return Err(StoreError::State(
                "not every provisional state was staged before seal",
            ));
        }
        if seal.artifact_digest == [0; 32]
            || seal.prompt_token_ids.len()
                != usize::try_from(self.declaration.cached_token_count)
                    .map_err(|_| StoreError::State("cached token count exceeds usize"))?
                    .saturating_add(1)
            || portable_prefill_token_ids_sha256(&seal.prompt_token_ids)
                != self.declaration.sealed_prompt_token_ids_sha256
        {
            return Err(StoreError::Authentication(
                "provisional seal has the wrong artifact, prompt count, or token commitment",
            ));
        }
        let cached = usize::try_from(self.declaration.cached_token_count)
            .map_err(|_| StoreError::State("cached token count exceeds usize"))?;
        let boundary_token_id = seal.prompt_token_ids[cached];
        let (final_cut, prefix_nodes) = kvpack_core::derive_input_cut(
            self.keys.prefix_key(),
            &self.store.tenant_namespace(),
            &self.declaration.semantic_model,
            &self.declaration.family,
            &seal.prompt_token_ids[..cached],
            &self.declaration.auxiliary_inputs,
        )?;
        if prefix_nodes.is_empty()
            || prefix_nodes.last().map(|node| node.id) != Some(final_cut.token_root)
            || prefix_nodes.last().map(|node| node.token_count) != Some(final_cut.token_count)
        {
            return Err(StoreError::Authentication(
                "provisional seal derived an inconsistent durable input cut",
            ));
        }
        let references = self
            .provisional_chunks
            .iter()
            .map(|chunk| chunk.stored.reference.clone())
            .collect::<Vec<_>>();
        self.store
            .verify_provisional_ledger(&self.declaration.source_declaration_digest, &references)?;

        let promotion_started = Instant::now();
        let mut promoted_bytes = 0u64;
        let mut promoted_chunk_count = 0u64;
        for chunk in &mut self.provisional_chunks {
            let bytes = if let Some(path) = chunk.staged.staged_path.as_ref() {
                read_exact_object(path, chunk.stored.reference.object_bytes as usize)?
            } else {
                read_exact_object(
                    &self.store.chunk_path(&chunk.stored.reference.object_key),
                    chunk.stored.reference.object_bytes as usize,
                )?
            };
            // Re-hash the re-read bytes: a size-only check would catalog
            // bit-rot between stage-fsync and promote under a good digest.
            let rehashed: Id32 = Sha256::digest(&bytes).into();
            if rehashed != chunk.stored.reference.object_digest {
                self.poisoned = true;
                return Err(StoreError::Authentication(
                    "provisional staged object digest mismatch before promotion",
                ));
            }
            let object = ChunkObject {
                chunk_id: chunk.stored.reference.chunk_id,
                object_key: chunk.stored.reference.object_key,
                object_digest: chunk.stored.reference.object_digest,
                plaintext_bytes: chunk.stored.reference.plaintext_bytes,
                bytes,
            };
            let (reference, ownership) = self.store.promote_provisional_chunk(
                &self.declaration.source_declaration_digest,
                &chunk.staged,
                &object,
                self.policy.retention,
            )?;
            if ownership.created_target {
                promoted_bytes = promoted_bytes
                    .checked_add(object.bytes.len() as u64)
                    .ok_or(StoreError::State("promoted byte total overflow"))?;
                promoted_chunk_count = promoted_chunk_count
                    .checked_add(1)
                    .ok_or(StoreError::State("promoted chunk count overflow"))?;
            } else if chunk.staged.staged_path.is_some() {
                self.deduplicated_bytes = self
                    .deduplicated_bytes
                    .checked_add(object.bytes.len() as u64)
                    .ok_or(StoreError::State("deduplicated byte total overflow"))?;
                self.deduplicated_chunk_count = self
                    .deduplicated_chunk_count
                    .checked_add(1)
                    .ok_or(StoreError::State("deduplicated chunk count overflow"))?;
            }
            chunk.stored.reference = reference.clone();
            chunk.staged.reference = reference;
            promoted.push(ownership);
        }
        let promotion_duration_ns = duration_ns(promotion_started.elapsed());
        // Copy any canonical references resolved by a concurrent dedup winner
        // back into the state inventories used by manifest construction.
        let mut flat = self.provisional_chunks.iter();
        for state in &mut self.completed {
            for chunk in &mut state.chunks {
                let promoted_chunk = flat.next().ok_or(StoreError::State(
                    "promoted provisional inventory became incomplete",
                ))?;
                chunk.reference = promoted_chunk.stored.reference.clone();
            }
        }
        if flat.next().is_some() {
            return Err(StoreError::State(
                "promoted provisional inventory has trailing chunks",
            ));
        }

        let mut manifests = Vec::with_capacity(prefix_nodes.len());
        let mut encoded = Vec::with_capacity(prefix_nodes.len());
        let mut previous = None;
        for node in &prefix_nodes {
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
            let manifest = manifest_for_cut_parts(
                ManifestParts {
                    store: &self.store,
                    semantic_model: self.declaration.semantic_model,
                    family: &self.declaration.family,
                    declarations: &self.declaration.states,
                    completed: &self.completed,
                    auxiliary_root: final_cut.auxiliary_input_root,
                },
                *node,
                kind,
            )?;
            let object = if self.replay {
                let manifest_id = manifest_id(&manifest.encode_canonical()?);
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
        let final_manifest = encoded
            .last()
            .ok_or(StoreError::State("provisional export produced no manifest"))?
            .manifest_id;
        let seal_digest = provisional_seal_digest(
            &self.declaration.source_declaration_digest,
            &seal.artifact_digest,
            &final_cut,
            &final_manifest,
            boundary_token_id,
            &self.provenance,
            &self.provisional_chunks,
        );
        self.store.seal_provisional_upload(
            &self.declaration.source_declaration_digest,
            self.session_token,
            self.next_ordinal,
            &seal_digest,
            boundary_token_id,
        )?;
        let pending: Vec<_> = encoded
            .iter()
            .zip(&manifests)
            .zip(&prefix_nodes)
            .enumerate()
            .map(|(index, ((encoded, manifest), node))| PendingManifest {
                encoded,
                manifest,
                prefix_node: *node,
                exact_final: index + 1 == prefix_nodes.len(),
            })
            .collect();
        let publication_started = Instant::now();
        let published = self
            .store
            .publish_manifest_batch(&self.declaration.source_declaration_digest, &pending)?;
        let publication_duration_ns = duration_ns(publication_started.elapsed());
        let cuts = published
            .into_iter()
            .zip(manifests)
            .map(|(published, manifest)| published_cut(published, manifest))
            .collect::<Vec<_>>();
        let exact_final = cuts
            .last()
            .cloned()
            .ok_or(StoreError::State("provisional export has no final cut"))?;
        let checkpoints = cuts
            .into_iter()
            .zip(&prefix_nodes)
            .filter(|(_, node)| node.reusable)
            .map(|(cut, _)| cut)
            .collect();
        let authenticated = self.store.read_authenticated_manifest_object(
            &exact_final.manifest_id,
            &self.policy.validation_context(),
        )?;
        let header = kvpack_core::inspect_pack_header(&authenticated)?;
        if header.manifest_id != exact_final.manifest_id {
            return Err(StoreError::Authentication(
                "reopened provisional final manifest identity changed",
            ));
        }
        self.store
            .finish_provisional_upload_dir(&self.declaration.source_declaration_digest)?;
        self.store
            .finish_provisional_ledger(&self.declaration.source_declaration_digest)?;
        self.terminal = true;
        Ok(ProvisionalExportReceipt {
            begin_duration_ns: self.begin_duration_ns,
            encryption_duration_ns: self.encryption_duration_ns,
            staging_duration_ns: self.staging_duration_ns,
            promotion_duration_ns,
            publication_duration_ns,
            total_duration_ns: duration_ns(self.origin.elapsed()),
            staged_bytes: self.staged_bytes,
            deduplicated_bytes: self.deduplicated_bytes,
            promoted_bytes,
            staged_chunk_count: self.staged_chunk_count,
            deduplicated_chunk_count: self.deduplicated_chunk_count,
            promoted_chunk_count,
            chunk_count: self.next_ordinal,
            boundary_token_id,
            provenance: self.provenance,
            write_intervals: std::mem::take(&mut self.write_intervals),
            encryption_intervals: std::mem::take(&mut self.encryption_intervals),
            published: PublishedCutSet {
                checkpoints,
                exact_final,
            },
        })
    }
}
