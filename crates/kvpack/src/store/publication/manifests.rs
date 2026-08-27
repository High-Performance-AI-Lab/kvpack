use super::*;

impl LocalStore {
    pub(crate) fn publish_manifest(
        &self,
        idempotency: &Id32,
        encoded: &EncodedPack,
        manifest: &CutManifest,
        prefix_nodes: &[PrefixNode],
    ) -> Result<PublishedArtifact, StoreError> {
        if prefix_nodes.len() != 1 {
            return Err(StoreError::State(
                "one manifest publication requires exactly its own prefix node",
            ));
        }
        let pending = [PendingManifest {
            encoded,
            manifest,
            prefix_node: prefix_nodes[0],
            exact_final: true,
        }];
        self.publish_manifest_batch(idempotency, &pending)?
            .pop()
            .ok_or(StoreError::State("manifest publication returned no result"))
    }

    pub(crate) fn publish_manifest_batch(
        &self,
        idempotency: &Id32,
        publications: &[PendingManifest<'_>],
    ) -> Result<Vec<PublishedArtifact>, StoreError> {
        let started = std::time::Instant::now();
        let started_unix_ns = now_ns();
        let context = TraceContext::new(ServiceComponent::Store).ok();
        let result = self.publish_manifest_batch_inner(idempotency, publications);
        let outcome = match &result {
            Ok(_) => SpanOutcome::Ok,
            Err(StoreError::Cancelled) => SpanOutcome::Cancelled,
            Err(StoreError::Integrity(_) | StoreError::Authentication(_)) => {
                SpanOutcome::IntegrityError
            }
            Err(StoreError::Busy | StoreError::Quota(_) | StoreError::Endurance(_)) => {
                SpanOutcome::Rejected
            }
            Err(_) => SpanOutcome::Unavailable,
        };
        let _ = self
            .telemetry
            .observe_latency(TracePhase::Publication, started.elapsed());
        if let Some(context) = context {
            let _ = self.telemetry.record_span(
                &context,
                None,
                TracePhase::Publication,
                outcome,
                started_unix_ns,
                now_ns().max(started_unix_ns),
            );
        }
        result
    }

    fn publish_manifest_batch_inner(
        &self,
        idempotency: &Id32,
        publications: &[PendingManifest<'_>],
    ) -> Result<Vec<PublishedArtifact>, StoreError> {
        let (semantic, family) = validate_and_write_manifest_batch(self, publications)?;

        durability_fault(self, DurabilityFaultPoint::CatalogBegin)?;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, reserved, publication_generation): (String, u64, u64) = transaction.query_row(
            "SELECT state,reserved_bytes,generation FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let mut audit_events = Vec::with_capacity(publications.len().saturating_add(1));
        audit_events.push(AuditEventKey::new(
            AuditEventKind::Verified,
            AuditObjectKind::Upload,
            *idempotency,
            publication_generation,
        ));
        audit_events.extend(publications.iter().map(|publication| {
            AuditEventKey::new(
                AuditEventKind::Published,
                AuditObjectKind::Manifest,
                publication.encoded.manifest_id,
                publication_generation,
            )
        }));
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        if state == "PUBLISHED" {
            let existing: Vec<u8> = transaction.query_row(
                "SELECT manifest_id FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| row.get(0),
            )?;
            if vec_id(existing)? != publications.last().unwrap().encoded.manifest_id {
                return Err(StoreError::Authentication(
                    "published idempotency result changed",
                ));
            }
            for publication in publications {
                let row: Option<(Vec<u8>, u64, i64)> = transaction
                    .query_row(
                        "SELECT manifest_id,token_count,exact_final FROM prefix_checkpoints WHERE tenant=?1 AND prefix_id=?2 AND semantic_id=?3 AND family_id=?4",
                        params![
                            self.tenant_namespace.as_slice(),
                            publication.prefix_node.id.as_slice(),
                            semantic.as_slice(),
                            family.as_slice()
                        ],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                if row
                    .as_ref()
                    .is_none_or(|(manifest_id, token_count, exact_final)| {
                        manifest_id.as_slice() != publication.encoded.manifest_id
                            || *token_count != publication.prefix_node.token_count
                            || (*exact_final != 0) != publication.exact_final
                    })
                {
                    return Err(StoreError::Authentication(
                        "published cut set no longer matches its idempotent result",
                    ));
                }
            }
            let result = publications
                .iter()
                .map(|publication| PublishedArtifact {
                    manifest_id: publication.encoded.manifest_id,
                    tenant_namespace: self.tenant_namespace,
                    restored_bytes: publication.manifest.realized_schema.complete_restored_bytes,
                    publication_generation,
                })
                .collect();
            let enqueued = audit::append_events(
                &transaction,
                &self.tenant_namespace,
                &audit_events,
                now_ns(),
            )?;
            transaction.commit()?;
            let _ = self
                .telemetry
                .record_audit_count(AuditOutcome::Enqueued, enqueued);
            return Ok(result);
        }
        if state != "RECEIVING" && state != "RESERVED" && state != "VERIFIED" {
            return Err(StoreError::State("upload cannot enter chunk verification"));
        }

        type CatalogChunk = (Id32, Id32, u64, u64);
        let mut chunk_rows: BTreeMap<Id32, CatalogChunk> = BTreeMap::new();
        let mut refcount_increments: BTreeMap<Id32, u64> = BTreeMap::new();
        let mut new_manifest_bytes = 0u64;
        let mut new_chunk_bytes = 0u64;
        for publication in publications {
            let manifest = publication.manifest;
            let encoded = publication.encoded;
            let parent = manifest.realized_schema.kind.parent();
            let parent_depth = manifest.realized_schema.kind.depth();
            let existing: Option<StoredManifestRow> =
                transaction
                    .query_row(
                        "SELECT key_epoch,file_bytes,restored_bytes,semantic_id,family_id,token_count,parent_id,parent_depth FROM manifests WHERE tenant=?1 AND manifest_id=?2",
                        params![
                            self.tenant_namespace.as_slice(),
                            encoded.manifest_id.as_slice()
                        ],
                        |row| {
                            Ok((
                                row.get(0)?,
                                row.get(1)?,
                                row.get(2)?,
                                row.get(3)?,
                                row.get(4)?,
                                row.get(5)?,
                                row.get(6)?,
                                row.get(7)?,
                            ))
                        },
                    )
                    .optional()?;
            if let Some(existing) = existing {
                if existing.0 != manifest.key_epoch
                    || existing.1 != encoded.bytes.len() as u64
                    || existing.2 != manifest.realized_schema.complete_restored_bytes
                    || existing.3.as_slice() != semantic
                    || existing.4.as_slice() != family
                    || existing.5 != manifest.input_cut.token_count
                    || existing.6.as_deref() != parent.as_ref().map(|value| value.as_slice())
                    || existing.7 != parent_depth as u64
                {
                    return Err(StoreError::Authentication(
                        "existing manifest catalog row disagrees with immutable bytes",
                    ));
                }
                let path = self.manifest_path(&encoded.manifest_id);
                transaction.execute("INSERT OR REPLACE INTO locations(tenant,object_kind,object_id,tier,state,locator) VALUES(?1,'manifest',?2,'local','AVAILABLE',?3)", params![self.tenant_namespace.as_slice(), encoded.manifest_id.as_slice(), path.to_string_lossy()])?;
                continue;
            }

            let mut manifest_objects = BTreeSet::new();
            for state_manifest in &manifest.states {
                for (ordinal, chunk) in state_manifest.chunks.iter().enumerate() {
                    let row = if let Some(row) = chunk_rows.get(&chunk.object_key) {
                        *row
                    } else {
                        let raw: (Vec<u8>, Vec<u8>, u64, u64) = transaction.query_row("SELECT chunk_id,object_digest,object_bytes,refcount FROM chunks WHERE tenant=?1 AND object_key=?2 AND location_state='AVAILABLE'", params![self.tenant_namespace.as_slice(), chunk.object_key.as_slice()], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)))?;
                        let row = (vec_id(raw.0)?, vec_id(raw.1)?, raw.2, raw.3);
                        chunk_rows.insert(chunk.object_key, row);
                        row
                    };
                    if row.0 != chunk.chunk_id
                        || row.1 != chunk.object_digest
                        || row.2 != chunk.object_bytes as u64
                    {
                        return Err(StoreError::Authentication(
                            "manifest references a mismatched chunk catalog row",
                        ));
                    }
                    manifest_objects.insert(chunk.object_key);
                    transaction.execute("INSERT INTO manifest_chunks(tenant,manifest_id,state_layer,state_name,ordinal,object_key) VALUES(?1,?2,?3,?4,?5,?6)", params![self.tenant_namespace.as_slice(), encoded.manifest_id.as_slice(), state_manifest.key.layer, state_manifest.key.state_name, ordinal as u64, chunk.object_key.as_slice()])?;
                }
            }
            for object_key in manifest_objects {
                *refcount_increments.entry(object_key).or_default() += 1;
            }
            new_manifest_bytes = new_manifest_bytes
                .checked_add(encoded.bytes.len() as u64)
                .ok_or(StoreError::Quota("durable byte count overflow"))?;
            transaction.execute("INSERT INTO manifests(tenant,manifest_id,key_epoch,catalog_epoch,file_bytes,restored_bytes,semantic_id,family_id,token_count,parent_id,parent_depth,published_ns,generation) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)", params![self.tenant_namespace.as_slice(), encoded.manifest_id.as_slice(), manifest.key_epoch, self.catalog_epoch(), encoded.bytes.len() as u64, manifest.realized_schema.complete_restored_bytes, semantic.as_slice(), family.as_slice(), manifest.input_cut.token_count, parent.map(|id| id.as_slice()), parent_depth, now_ns(), publication_generation])?;
            let path = self.manifest_path(&encoded.manifest_id);
            transaction.execute("INSERT OR REPLACE INTO locations(tenant,object_kind,object_id,tier,state,locator) VALUES(?1,'manifest',?2,'local','AVAILABLE',?3)", params![self.tenant_namespace.as_slice(), encoded.manifest_id.as_slice(), path.to_string_lossy()])?;
        }
        for (object_key, increment) in &refcount_increments {
            let row = chunk_rows[object_key];
            if row.3 == 0 {
                new_chunk_bytes = new_chunk_bytes
                    .checked_add(row.2)
                    .ok_or(StoreError::Quota("durable byte count overflow"))?;
            }
            transaction.execute(
                "UPDATE chunks SET refcount=refcount+?3 WHERE tenant=?1 AND object_key=?2",
                params![
                    self.tenant_namespace.as_slice(),
                    object_key.as_slice(),
                    increment
                ],
            )?;
        }
        if state != "VERIFIED" {
            transition(
                &transaction,
                &self.tenant_namespace,
                idempotency,
                "RECEIVING",
                "VERIFIED",
            )
            .or_else(|error| {
                if state == "RESERVED" {
                    transition(
                        &transaction,
                        &self.tenant_namespace,
                        idempotency,
                        "RESERVED",
                        "VERIFIED",
                    )
                } else {
                    Err(error)
                }
            })?;
        }
        let actual_bytes = new_chunk_bytes
            .checked_add(new_manifest_bytes)
            .ok_or(StoreError::Quota("durable byte count overflow"))?;
        if actual_bytes > reserved {
            return Err(StoreError::Quota(
                "actual durable publication exceeds reservation",
            ));
        }
        for publication in publications {
            transaction.execute("INSERT INTO prefix_checkpoints(tenant,prefix_id,semantic_id,family_id,token_count,manifest_id,exact_final) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(tenant,prefix_id,semantic_id,family_id) DO UPDATE SET manifest_id=excluded.manifest_id,token_count=excluded.token_count,exact_final=excluded.exact_final", params![self.tenant_namespace.as_slice(), publication.prefix_node.id.as_slice(), semantic.as_slice(), family.as_slice(), publication.prefix_node.token_count, publication.encoded.manifest_id.as_slice(), i64::from(publication.exact_final)])?;
        }
        transaction.execute("UPDATE tenants SET reserved_bytes=MAX(0,reserved_bytes-?2),durable_bytes=durable_bytes+?3 WHERE namespace=?1", params![self.tenant_namespace.as_slice(), reserved, actual_bytes])?;
        transaction.execute(
            "UPDATE write_tickets SET consumed_bytes=?3 WHERE tenant=?1 AND ticket_id=?2",
            params![
                self.tenant_namespace.as_slice(),
                idempotency.as_slice(),
                actual_bytes
            ],
        )?;
        let final_manifest_id = publications.last().unwrap().encoded.manifest_id;
        transaction.execute("UPDATE uploads SET state='PUBLISHED',manifest_id=?3,updated_ns=?4 WHERE tenant=?1 AND idempotency_key=?2 AND state='VERIFIED'", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), final_manifest_id.as_slice(), now_ns()])?;
        let enqueued = audit::append_events(
            &transaction,
            &self.tenant_namespace,
            &audit_events,
            now_ns(),
        )?;
        durability_fault(self, DurabilityFaultPoint::CatalogCommit)?;
        transaction.commit()?;
        let _ = self.telemetry.record_lifecycle(CacheLifecycle::Verified);
        let _ = self.telemetry.record_lifecycle(CacheLifecycle::Published);
        let _ = self
            .telemetry
            .add_bytes(ByteCounter::DurableWritten, new_manifest_bytes);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, enqueued);
        Ok(publications
            .iter()
            .map(|publication| PublishedArtifact {
                manifest_id: publication.encoded.manifest_id,
                tenant_namespace: self.tenant_namespace,
                restored_bytes: publication.manifest.realized_schema.complete_restored_bytes,
                publication_generation,
            })
            .collect())
    }
}
