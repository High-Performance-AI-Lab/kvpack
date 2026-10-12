use super::*;

impl LocalStore {
    pub(crate) fn published_upload(
        &self,
        idempotency: &Id32,
    ) -> Result<Option<PublishedArtifact>, StoreError> {
        let connection = self.lock_catalog()?;
        let row: Option<(Vec<u8>, u64, u64)> = connection
            .query_row(
                "SELECT u.manifest_id,m.restored_bytes,u.generation FROM uploads u JOIN manifests m ON m.tenant=u.tenant AND m.manifest_id=u.manifest_id WHERE u.tenant=?1 AND u.idempotency_key=?2 AND u.state='PUBLISHED'",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        row.map(|(manifest_id, restored_bytes, publication_generation)| {
            Ok(PublishedArtifact {
                manifest_id: vec_id(manifest_id)?,
                tenant_namespace: self.tenant_namespace,
                restored_bytes,
                publication_generation,
            })
        })
        .transpose()
    }

    pub(crate) fn reserve_upload(
        &self,
        idempotency: &Id32,
        reservation: UploadReservation,
    ) -> Result<UploadState, StoreError> {
        let UploadReservation {
            expected_bytes,
            publication_generation,
            intent_digest,
            retention,
        } = reservation;
        self.reap_expired_provisional_uploads()?;
        if idempotency.iter().all(|byte| *byte == 0)
            || expected_bytes == 0
            || expected_bytes > i64::MAX as u64
            || publication_generation == 0
            || publication_generation > i64::MAX as u64
            || intent_digest.iter().all(|byte| *byte == 0)
        {
            return Err(StoreError::State("invalid write reservation"));
        }
        let retention = retention.validate()?;
        if retention.physical_bytes != expected_bytes {
            return Err(StoreError::State(
                "write reservation retention bytes disagree with its bound",
            ));
        }
        let mut reinit_pending = false;
        {
            let connection = self.lock_catalog()?;
            let existing: Option<(String, u64, u64, Vec<u8>)> = connection
                .query_row(
                    "SELECT state,expected_bytes,generation,intent_digest FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                    params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()?;
            if let Some((state, expected, generation, digest)) = existing {
                if expected != expected_bytes {
                    return Err(StoreError::Expectation(
                        "idempotency key was reused with different write bounds",
                    ));
                }
                if generation != publication_generation {
                    return Err(StoreError::Expectation(
                        "idempotency key was reused with a different publication generation",
                    ));
                }
                if vec_id(digest)? != intent_digest {
                    return Err(StoreError::Expectation(
                        "idempotency key was reused with a different immutable declaration",
                    ));
                }
                // RE_INIT: an ABORTED row with a byte-identical declaration
                // falls through and is re-reserved under the write
                // transaction below (docs/UPLOAD_REINIT_DESIGN.md).
                if state != "ABORTED" {
                    return UploadState::parse(&state);
                }
                reinit_pending = true;
            } else {
                let window_start = now_ns().saturating_sub(300_000_000_000);
                let recent: u64 = connection.query_row("SELECT COALESCE(SUM(reserved_bytes),0) FROM write_tickets WHERE tenant=?1 AND bucket_start_ns>=?2", params![self.tenant_namespace.as_slice(), window_start], |row| row.get(0))?;
                if recent.saturating_add(expected_bytes)
                    > self.config.endurance_bytes_per_five_minutes
                {
                    return Err(StoreError::Endurance(
                        "five-minute endurance budget exhausted",
                    ));
                }
            }
        }
        self.enforce_quarantine_cap()?;
        let stat = self.stat()?;
        if stat.reserved_bytes.saturating_add(expected_bytes) > stat.staging_quota_bytes {
            return Err(StoreError::Quota(
                "tenant staging quota reservation refused",
            ));
        }
        let projected = stat
            .durable_bytes
            .saturating_add(stat.reserved_bytes)
            .saturating_add(expected_bytes);
        if projected > stat.quota_bytes {
            return Err(StoreError::Quota(
                "tenant durable quota reservation refused",
            ));
        }
        if (projected as u128).saturating_mul(100) >= (stat.quota_bytes as u128).saturating_mul(85)
        {
            let utilization_millis = (projected as u128)
                .saturating_mul(1000)
                .checked_div(stat.quota_bytes as u128)
                .unwrap_or(1000)
                .min(1000) as u16;
            let decision = self
                .policy
                .lock()
                .map_err(|_| StoreError::State("TinyLFU policy mutex poisoned"))?
                .automatic_admission_decision(idempotency, retention, utilization_millis)?;
            match decision {
                AdmissionDecision::RejectLowerFrequency => {
                    return Err(StoreError::Quota(
                        "TinyLFU rejected lower-value cache admission",
                    ));
                }
                AdmissionDecision::RejectPromotionStopped => {
                    return Err(StoreError::Quota(
                        "cache promotion stopped at 95% utilization",
                    ));
                }
                AdmissionDecision::Admit | AdmissionDecision::AdmitOverVictim(_) => {}
            }
            let report = self.maintain_capacity_with_headroom(
                stat.quota_bytes,
                UtilizationPolicy::default(),
                1024,
                expected_bytes,
            )?;
            let utilization = (report.after_bytes as u128).saturating_mul(100);
            let capacity = (stat.quota_bytes as u128).saturating_mul(100);
            if utilization >= capacity.saturating_mul(95) / 100 {
                return Err(StoreError::Quota(
                    "cache promotion stopped at 95% utilization",
                ));
            }
            if report.blocked && utilization >= capacity.saturating_mul(92) / 100 {
                return Err(StoreError::Quota(
                    "cache is read-only under emergency utilization",
                ));
            }
        }
        if reinit_pending {
            // RE_INIT restarts from a clean cursor: discard any staged
            // leftovers the abort path did not clean up (a plain abort, as
            // opposed to reconcile/reap/cancel, leaves the upload directory
            // and the chunk ledger behind). Both operations require the
            // terminal ABORTED state and fail closed on a race.
            self.finish_provisional_upload_dir(idempotency)?;
            self.clear_provisional_ledger(idempotency, false)?;
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, u64, u64, Vec<u8>)> = transaction
            .query_row(
                "SELECT state,expected_bytes,generation,intent_digest FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        let mut reinit_aborted = false;
        if let Some((state, expected, generation, digest)) = existing {
            if expected != expected_bytes {
                return Err(StoreError::Expectation(
                    "idempotency key was reused with different write bounds",
                ));
            }
            if generation != publication_generation {
                return Err(StoreError::Expectation(
                    "idempotency key was reused with a different publication generation",
                ));
            }
            if vec_id(digest)? != intent_digest {
                return Err(StoreError::Expectation(
                    "idempotency key was reused with a different immutable declaration",
                ));
            }
            if state != "ABORTED" {
                return UploadState::parse(&state);
            }
            reinit_aborted = true;
        }
        let (durable, reserved, quota, staging_quota): (u64, u64, u64, u64) = transaction.query_row(
            "SELECT durable_bytes,reserved_bytes,quota_bytes,staging_quota_bytes FROM tenants WHERE namespace=?1",
            [self.tenant_namespace.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        if reserved.saturating_add(expected_bytes) > staging_quota {
            return Err(StoreError::Quota(
                "tenant staging quota reservation refused",
            ));
        }
        if durable
            .saturating_add(reserved)
            .saturating_add(expected_bytes)
            > quota
        {
            return Err(StoreError::Quota(
                "tenant durable quota reservation refused",
            ));
        }
        let window_start = now_ns().saturating_sub(300_000_000_000);
        let recent: u64 = transaction.query_row("SELECT COALESCE(SUM(reserved_bytes),0) FROM write_tickets WHERE tenant=?1 AND bucket_start_ns>=?2", params![self.tenant_namespace.as_slice(), window_start], |row| row.get(0))?;
        if recent.saturating_add(expected_bytes) > self.config.endurance_bytes_per_five_minutes {
            return Err(StoreError::Endurance(
                "five-minute endurance budget exhausted",
            ));
        }
        let audit_events = [AuditEventKey::new(
            AuditEventKind::Reserved,
            AuditObjectKind::Upload,
            *idempotency,
            publication_generation,
        )];
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        let timestamp = now_ns();
        if reinit_aborted {
            // RE_INIT (docs/UPLOAD_REINIT_DESIGN.md): the staged bytes are
            // already gone, so the reservation restarts from a clean cursor.
            // The compare-and-swap on state='ABORTED' fences a concurrent
            // abort/re-init; the loser fails closed.
            let zero_digest = [0u8; 32];
            let changed = transaction.execute("UPDATE uploads SET state='RESERVED',reserved_bytes=?3,abort_reason=NULL,next_chunk_ordinal=0,seal_digest=?4,session_token=0,lease_expires_ns=0,boundary_token_id=NULL,manifest_id=NULL,updated_ns=?5 WHERE tenant=?1 AND idempotency_key=?2 AND state='ABORTED'", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), expected_bytes, zero_digest.as_slice(), timestamp])?;
            if changed != 1 {
                return Err(StoreError::State("upload re-init raced an abort"));
            }
        } else {
            transaction.execute("INSERT INTO uploads(tenant,idempotency_key,state,reserved_bytes,expected_bytes,generation,intent_digest,created_ns,updated_ns) VALUES(?1,?2,'INIT',?3,?3,?4,?5,?6,?6)", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), expected_bytes, publication_generation, intent_digest.as_slice(), timestamp])?;
            transaction.execute("UPDATE uploads SET state='RESERVED' WHERE tenant=?1 AND idempotency_key=?2 AND state='INIT'", params![self.tenant_namespace.as_slice(), idempotency.as_slice()])?;
        }
        transaction.execute(
            "UPDATE tenants SET reserved_bytes=reserved_bytes+?2 WHERE namespace=?1",
            params![self.tenant_namespace.as_slice(), expected_bytes],
        )?;
        transaction.execute("INSERT INTO write_tickets(tenant,ticket_id,bucket_start_ns,reserved_bytes) VALUES(?1,?2,?3,?4) ON CONFLICT(tenant,ticket_id) DO UPDATE SET bucket_start_ns=excluded.bucket_start_ns,reserved_bytes=excluded.reserved_bytes", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), timestamp, expected_bytes])?;
        let enqueued = audit::append_events(
            &transaction,
            &self.tenant_namespace,
            &audit_events,
            timestamp,
        )?;
        transaction.commit()?;
        let _ = self.telemetry.record_lifecycle(CacheLifecycle::Reserved);
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, enqueued);
        Ok(UploadState::Reserved)
    }

    pub(crate) fn mark_receiving(&self, idempotency: &Id32) -> Result<(), StoreError> {
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, generation): (String, u64) = transaction
            .query_row(
                "SELECT state,generation FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .ok_or(StoreError::NotFound)?;
        if state != "RESERVED" && state != "RECEIVING" {
            return Err(StoreError::State("upload is not reserved"));
        }
        let audit_events = [AuditEventKey::new(
            AuditEventKind::Receiving,
            AuditObjectKind::Upload,
            *idempotency,
            generation,
        )];
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        let changed = transaction.execute("UPDATE uploads SET state='RECEIVING',updated_ns=?3 WHERE tenant=?1 AND idempotency_key=?2 AND state='RESERVED'", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), now_ns()])?;
        let enqueued = audit::append_events(
            &transaction,
            &self.tenant_namespace,
            &audit_events,
            now_ns(),
        )?;
        transaction.commit()?;
        if changed == 1 {
            let _ = self.telemetry.record_lifecycle(CacheLifecycle::Receiving);
        }
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, enqueued);
        Ok(())
    }

    pub(crate) fn abort_upload(&self, idempotency: &Id32) -> Result<(), StoreError> {
        let mut transitioned = false;
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let upload: Option<(String, u64, u64)> = transaction
            .query_row(
                "SELECT state,reserved_bytes,generation FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let mut enqueued = 0u64;
        if let Some((state, bytes, generation)) = upload {
            if matches!(state.as_str(), "PUBLISHED" | "QUARANTINED") {
                transaction.commit()?;
                return Ok(());
            }
            let audit_events = [AuditEventKey::new(
                AuditEventKind::Aborted,
                AuditObjectKind::Upload,
                *idempotency,
                generation,
            )];
            if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
                == AuditCapacity::Backpressured
            {
                transaction.commit()?;
                let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
                return Err(StoreError::Busy);
            }
            if state != "ABORTED" {
                let changed = transaction.execute("UPDATE uploads SET state='ABORTED',reserved_bytes=0,abort_reason='cancelled',updated_ns=?3 WHERE tenant=?1 AND idempotency_key=?2 AND state IN ('INIT','RESERVED','RECEIVING','VERIFIED')", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), now_ns()])?;
                if changed != 1 {
                    return Err(StoreError::State("upload cannot transition to aborted"));
                }
                transitioned = true;
                transaction.execute(
                    "UPDATE tenants SET reserved_bytes=MAX(0,reserved_bytes-?2) WHERE namespace=?1",
                    params![self.tenant_namespace.as_slice(), bytes],
                )?;
            }
            enqueued = audit::append_events(
                &transaction,
                &self.tenant_namespace,
                &audit_events,
                now_ns(),
            )?;
        }
        transaction.commit()?;
        if transitioned {
            let _ = self.telemetry.record_lifecycle(CacheLifecycle::Aborted);
        }
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, enqueued);
        Ok(())
    }

    pub(crate) fn quarantine_upload(
        &self,
        idempotency: &Id32,
        reason: &'static str,
        file: Option<&QuarantinedUploadFile>,
    ) -> Result<(), StoreError> {
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let upload: Option<(String, u64, u64)> = transaction
            .query_row(
                "SELECT state,reserved_bytes,generation FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let Some((state, reserved_bytes, generation)) = upload else {
            return Err(StoreError::NotFound);
        };
        if matches!(state.as_str(), "PUBLISHED" | "ABORTED") {
            return Err(StoreError::State(
                "terminal upload cannot transition to quarantined",
            ));
        }
        let audit_events = [AuditEventKey::new(
            AuditEventKind::Quarantined,
            AuditObjectKind::Upload,
            *idempotency,
            generation,
        )];
        if audit::preflight_events(&transaction, &self.tenant_namespace, &audit_events)?
            == AuditCapacity::Backpressured
        {
            transaction.commit()?;
            let _ = self.telemetry.record_audit(AuditOutcome::Backpressure);
            return Err(StoreError::Busy);
        }
        let transitioned = state != "QUARANTINED";
        if transitioned {
            let changed = transaction.execute("UPDATE uploads SET state='QUARANTINED',reserved_bytes=0,abort_reason=?3,updated_ns=?4 WHERE tenant=?1 AND idempotency_key=?2 AND state IN ('INIT','RESERVED','RECEIVING','VERIFIED')", params![self.tenant_namespace.as_slice(), idempotency.as_slice(), reason, now_ns()])?;
            if changed != 1 {
                return Err(StoreError::State("upload cannot transition to quarantined"));
            }
            transaction.execute(
                "UPDATE tenants SET reserved_bytes=MAX(0,reserved_bytes-?2) WHERE namespace=?1",
                params![self.tenant_namespace.as_slice(), reserved_bytes],
            )?;
            if let Some(file) = file {
                let created = now_ns();
                let expires = created.saturating_add(24 * 60 * 60 * 1_000_000_000);
                transaction.execute("INSERT INTO quarantine_entries(tenant,entry_id,object_kind,object_id,path_token,file_bytes,created_ns,expires_ns,reason) VALUES(?1,?2,'upload_manifest',?3,?4,?5,?6,?7,?8)", params![self.tenant_namespace.as_slice(), file.entry_id.as_slice(), idempotency.as_slice(), file.path_token, file.file_bytes, created, expires, reason])?;
            }
        }
        let enqueued = audit::append_events(
            &transaction,
            &self.tenant_namespace,
            &audit_events,
            now_ns(),
        )?;
        transaction.commit()?;
        drop(connection);
        if transitioned && file.is_some() {
            self.maintain_quarantine_cap()?;
        }
        if transitioned {
            let _ = self.telemetry.record_lifecycle(CacheLifecycle::Quarantined);
        }
        let _ = self
            .telemetry
            .record_audit_count(AuditOutcome::Enqueued, enqueued);
        Ok(())
    }
}
