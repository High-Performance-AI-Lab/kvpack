use super::*;

// M6 fidelity ladder demotion.  Watermark pressure demotes the coldest
// chunk objects exactly one rung per round — 0 resident-fp16 → 1
// rest-quantized → 2 tombstone — and eviction proper only ever collects
// objects already on the tombstone rung, so eviction never skips ahead.
//
// Rung semantics:
// - 0 → 1: catalog annotation only; the authenticated object bytes are
//   untouched (a rest-quantized re-encode would produce a *new* object).
// - 1 → 2: the local bytes are dropped (trash + unlink), the locations
//   rows are removed, and the tenant durable-byte total is released, but
//   the `chunks` row — the chained key (`object_key`/`chunk_id`) and its
//   token-cut metadata — is retained so restore planning can mark the
//   object as a guided-recompute candidate instead of serving bytes.
//
// The demoted row's `location_state` becomes `TOMBSTONED`, which every
// byte-serving query (`location_state='AVAILABLE'`) rejects by default:
// a tombstoned chunk fails closed everywhere it is not explicitly handled.

/// Outcome of one demotion batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DemotionReport {
    /// Objects moved down exactly one rung in this batch.
    pub demoted: u64,
    /// 0 → 1 transitions (catalog annotation only).
    pub quantized: u64,
    /// 1 → 2 transitions (bytes dropped, chained key retained).
    pub tombstoned: u64,
    /// Durable bytes released by 1 → 2 transitions.
    pub freed_bytes: u64,
}

/// Coldest-first victim selection for demotion.  Higher rungs demote first
/// (a 1 → 2 transition is the one that frees bytes); ties fall back to the
/// same policy ordering as garbage collection.  Referenced chunks are
/// eligible: tombstoning a referenced chunk is exactly what turns its
/// restore plan into a guided-recompute candidate.  Pins, leases, source
/// leases, and in-flight uploads protect their objects as usual.
const DEMOTE_BATCH_SQL: &str = "SELECT c.object_key,c.fidelity_rung,c.object_bytes
 FROM chunks c
 LEFT JOIN policy_objects p ON p.tenant=c.tenant AND p.object_key=c.object_key
 WHERE c.tenant=?1 AND c.location_state='AVAILABLE' AND c.fidelity_rung<2
 AND NOT EXISTS(SELECT 1 FROM pins pin WHERE pin.tenant=c.tenant AND pin.object_key=c.object_key)
 AND NOT EXISTS(SELECT 1 FROM leases l WHERE l.tenant=c.tenant AND l.object_kind='chunk' AND l.object_id=c.object_key AND l.state='ACTIVE' AND l.expires_ns>?2)
 AND NOT EXISTS(SELECT 1 FROM source_lease_objects slo JOIN source_leases sl ON sl.tenant=slo.tenant AND sl.lease_id=slo.lease_id WHERE slo.tenant=c.tenant AND slo.object_kind='chunk' AND slo.object_id=c.object_key AND (sl.state='UNCERTAIN' OR (sl.state='ACTIVE' AND sl.expires_ns>?2)))
 AND NOT EXISTS(SELECT 1 FROM uploads u WHERE u.tenant=c.tenant AND u.state IN ('INIT','RESERVED','RECEIVING','VERIFIED'))
 ORDER BY c.fidelity_rung DESC,
 CASE COALESCE(p.segment,c.retention_segment) WHEN 'PROBATIONARY' THEN 0 ELSE 1 END,
 COALESCE(p.score,0),COALESCE(p.last_access_ns,c.last_access_ns),c.object_bytes DESC,c.object_key
 LIMIT ?3";

impl LocalStore {
    /// Current fidelity rung of one catalog chunk object (0 resident-fp16,
    /// 1 rest-quantized, 2 tombstone); `None` when the object is absent.
    pub fn chunk_fidelity_rung(
        &self,
        object_key: &kvpack_core::Id32,
    ) -> Result<Option<u8>, StoreError> {
        let connection = self.lock_catalog()?;
        let rung: Option<u64> = connection
            .query_row(
                "SELECT fidelity_rung FROM chunks WHERE tenant=?1 AND object_key=?2",
                params![self.tenant_namespace.as_slice(), object_key.as_slice()],
                |row| row.get(0),
            )
            .optional()?;
        Ok(rung.map(|rung| rung as u8))
    }

    /// Demote up to `limit` coldest chunk objects exactly one fidelity rung
    /// each.  Eviction never skips ahead: an object on rung 0 is annotated
    /// rest-quantized (bytes kept), and only an object already on rung 1 is
    /// tombstoned (bytes dropped, chained key retained).
    pub fn demote_fidelity_one_rung(&self, limit: usize) -> Result<DemotionReport, StoreError> {
        if limit == 0 {
            return Err(StoreError::State("demotion batch bound must be nonzero"));
        }
        let victims: Vec<(kvpack_core::Id32, u64, u64)> = {
            let connection = self.lock_catalog()?;
            let mut statement = connection.prepare(DEMOTE_BATCH_SQL)?;
            let rows = statement.query_map(
                params![self.tenant_namespace.as_slice(), now_ns(), limit as u64],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, u64>(1)?,
                        row.get::<_, u64>(2)?,
                    ))
                },
            )?;
            rows.collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|(raw, rung, bytes)| Ok((vec_id(raw)?, rung, bytes)))
                .collect::<Result<Vec<_>, StoreError>>()?
        };
        if victims.is_empty() {
            return Ok(DemotionReport::default());
        }
        let mut report = DemotionReport::default();
        {
            let mut connection = self.lock_catalog()?;
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            for (id, rung, object_bytes) in &victims {
                match rung {
                    0 => {
                        // Rung 1 is a catalog annotation: the authenticated
                        // bytes stay exactly as published.
                        let changed = transaction.execute("UPDATE chunks SET fidelity_rung=1 WHERE tenant=?1 AND object_key=?2 AND fidelity_rung=0 AND location_state='AVAILABLE'", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                        if changed == 1 {
                            report.demoted += 1;
                            report.quantized += 1;
                        }
                    }
                    1 => {
                        // Rung 2 drops the bytes but retains the chained key
                        // and the token-cut catalog row.  Durable bytes are
                        // released here; the later row eviction must not
                        // release them again.
                        let changed = transaction.execute("UPDATE chunks SET fidelity_rung=2,location_state='TOMBSTONED' WHERE tenant=?1 AND object_key=?2 AND fidelity_rung=1 AND location_state='AVAILABLE'", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                        if changed == 1 {
                            transaction.execute("DELETE FROM locations WHERE tenant=?1 AND object_kind='chunk' AND object_id=?2", params![self.tenant_namespace.as_slice(), id.as_slice()])?;
                            transaction.execute(
                                "UPDATE tenants SET durable_bytes=MAX(0,durable_bytes-?2) WHERE namespace=?1",
                                params![self.tenant_namespace.as_slice(), object_bytes],
                            )?;
                            report.demoted += 1;
                            report.tombstoned += 1;
                            report.freed_bytes = report
                                .freed_bytes
                                .checked_add(*object_bytes)
                                .ok_or(StoreError::State("demotion freed byte total overflow"))?;
                        }
                    }
                    _ => {
                        return Err(StoreError::State(
                            "catalog fidelity rung is outside the demotable ladder",
                        ));
                    }
                }
            }
            transaction.commit()?;
        }
        // Trash the dropped bytes after the catalog transition commits, the
        // same ordering as chunk eviction: a crash in the window leaves an
        // orphaned object file (a space leak the later row eviction cleans
        // up), never a byte-serving catalog row.
        for (id, rung, _) in &victims {
            if *rung != 1 {
                continue;
            }
            let source = self.chunk_path(id);
            let trash = self
                .config
                .object_root
                .join("trash")
                .join(format!("{}.demoted.trash", hex(id)));
            super::eviction::move_and_unlink(
                &source,
                &trash,
                "move tombstoned chunk to trash",
                "unlink trashed tombstoned chunk",
            )?;
        }
        Ok(report)
    }
}
