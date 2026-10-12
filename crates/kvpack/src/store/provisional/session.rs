//! Provisional upload session lifecycle: begin/resume with a fencing token,
//! seal, clock-offset provenance, lease reaping, and fenced cancel/abort.

use std::fs;
use std::os::unix::fs::DirBuilderExt;

use kvpack_core::Id32;
use rusqlite::{params, OptionalExtension, TransactionBehavior};

use crate::StoreError;

use super::super::{fsync_dir, vec_id, LocalStore, UploadReservation, UploadState};
use super::{
    new_session_token, now_ns, reject_expired_lease, ProvisionalProvenance,
    ProvisionalUploadMetadata, PROVISIONAL_UPLOAD_LEASE_NS,
};

impl LocalStore {
    /// Begin (or resume) a provisional upload. Returns the session fencing
    /// token and the provenance persisted on the upload row. A fresh token is
    /// minted and persisted on every begin, so a resume invalidates any stale
    /// session still holding the previous token.
    pub(crate) fn begin_provisional_upload(
        &self,
        idempotency: &Id32,
        reservation: UploadReservation,
        provenance: ProvisionalProvenance,
    ) -> Result<(UploadState, u64, ProvisionalProvenance), StoreError> {
        let state = self.reserve_upload(idempotency, reservation)?;
        let token = new_session_token()?;
        match state {
            UploadState::Reserved => {
                if let Err(error) = self.mark_receiving(idempotency) {
                    let _ = self.abort_upload(idempotency);
                    return Err(error);
                }
                if let Err(error) = self.adopt_provisional_session(idempotency, token, provenance) {
                    return self.fail_provisional_begin(idempotency, token, error);
                }
            }
            UploadState::Receiving => {
                // Resume: adopting a new token fences out the stale session.
                self.adopt_provisional_session(idempotency, token, provenance)?;
            }
            UploadState::Published => {
                let stored = self.provisional_session_row(idempotency)?;
                return Ok((UploadState::Published, stored.0, stored.1));
            }
            _ => {
                return Err(StoreError::State(
                    "provisional export cannot resume a terminal upload",
                ));
            }
        }
        let directory = self.provisional_upload_path(idempotency);
        let prepared = match fs::symlink_metadata(&directory) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => Ok(()),
            Ok(_) => {
                return self.fail_provisional_begin(
                    idempotency,
                    token,
                    StoreError::State("provisional upload path is not a private directory"),
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let mut builder = fs::DirBuilder::new();
                builder.mode(0o700);
                if let Err(source) = builder.create(&directory) {
                    return self.fail_provisional_begin(
                        idempotency,
                        token,
                        StoreError::Io {
                            op: "create provisional upload directory",
                            source,
                        },
                    );
                }
                fsync_dir(&self.config.object_root.join("uploads"))
            }
            Err(source) => {
                return self.fail_provisional_begin(
                    idempotency,
                    token,
                    StoreError::Io {
                        op: "inspect provisional upload directory",
                        source,
                    },
                );
            }
        };
        if let Err(error) = prepared {
            return self.fail_provisional_begin(idempotency, token, error);
        }
        Ok((
            if state == UploadState::Reserved {
                UploadState::Receiving
            } else {
                state
            },
            token,
            provenance,
        ))
    }

    fn adopt_provisional_session(
        &self,
        idempotency: &Id32,
        token: u64,
        provenance: ProvisionalProvenance,
    ) -> Result<(), StoreError> {
        let now = now_ns();
        let lease = now.saturating_add(PROVISIONAL_UPLOAD_LEASE_NS);
        let connection = self.lock_catalog()?;
        let changed = connection.execute(
            "UPDATE uploads SET session_token=?3,lease_expires_ns=?4,provenance_source_ns=?5,provenance_clock_offset_ns=?6,provenance_quiesced=?7,updated_ns=?8 WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING'",
            params![
                self.tenant_namespace.as_slice(),
                idempotency.as_slice(),
                token as i64,
                lease as i64,
                provenance.source_wall_clock_ns as i64,
                provenance.clock_offset_ns.map(|value| value as i64),
                i64::from(provenance.quiesced),
                now as i64,
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::State("provisional session adoption raced"));
        }
        Ok(())
    }

    fn provisional_session_row(
        &self,
        idempotency: &Id32,
    ) -> Result<(u64, ProvisionalProvenance), StoreError> {
        let connection = self.lock_catalog()?;
        let row: (i64, i64, Option<i64>, i64) = connection.query_row(
            "SELECT session_token,provenance_source_ns,provenance_clock_offset_ns,provenance_quiesced FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )?;
        Ok((
            row.0 as u64,
            ProvisionalProvenance {
                source_wall_clock_ns: row.1 as u64,
                clock_offset_ns: row.2.map(|value| value as u64),
                quiesced: row.3 != 0,
            },
        ))
    }

    fn fail_provisional_begin<T>(
        &self,
        idempotency: &Id32,
        token: u64,
        error: StoreError,
    ) -> Result<T, StoreError> {
        let _ = self.abort_provisional_session(idempotency, token);
        let _ = self.clear_provisional_ledger(idempotency, false);
        Err(error)
    }

    /// Abort only when the presented token still owns the reservation. A
    /// stale session's error or Drop must not kill the live reservation.
    fn abort_provisional_session(&self, idempotency: &Id32, token: u64) -> Result<(), StoreError> {
        let row: Option<(String, i64)> = {
            let connection = self.lock_catalog()?;
            connection
                .query_row(
                    "SELECT state,session_token FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                    params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?
        };
        if let Some((state, stored)) = row {
            if matches!(
                state.as_str(),
                "INIT" | "RESERVED" | "RECEIVING" | "VERIFIED"
            ) && stored as u64 != token
            {
                return Err(StoreError::State(
                    "stale provisional session token; the active reservation is untouched",
                ));
            }
        }
        self.abort_upload(idempotency)
    }

    pub(crate) fn seal_provisional_upload(
        &self,
        idempotency: &Id32,
        token: u64,
        expected_chunks: u64,
        seal_digest: &Id32,
        boundary_token_id: u32,
    ) -> Result<(), StoreError> {
        if seal_digest.iter().all(|byte| *byte == 0) {
            return Err(StoreError::Authentication(
                "provisional seal digest is zero",
            ));
        }
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (state, cursor, stored, session_token, lease_expires_ns): (
            String,
            u64,
            Vec<u8>,
            i64,
            i64,
        ) = transaction.query_row(
            "SELECT state,next_chunk_ordinal,seal_digest,session_token,lease_expires_ns FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )?;
        if session_token as u64 != token {
            return Err(StoreError::State(
                "provisional upload session token is stale",
            ));
        }
        if state != "PUBLISHED" && cursor != expected_chunks {
            return Err(StoreError::Authentication(
                "provisional seal observed an incomplete chunk ledger",
            ));
        }
        match state.as_str() {
            "PUBLISHED" | "VERIFIED" => {
                if vec_id(stored)? != *seal_digest {
                    return Err(StoreError::Authentication(
                        "provisional replay seal identity changed",
                    ));
                }
            }
            "RECEIVING" => {
                reject_expired_lease(lease_expires_ns)?;
                let changed = transaction.execute(
                    "UPDATE uploads SET state='VERIFIED',seal_digest=?3,boundary_token_id=?4,updated_ns=?5 WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING' AND next_chunk_ordinal=?6 AND session_token=?7",
                    params![
                        self.tenant_namespace.as_slice(),
                        idempotency.as_slice(),
                        seal_digest.as_slice(),
                        i64::from(boundary_token_id),
                        now_ns() as i64,
                        expected_chunks,
                        token as i64,
                    ],
                )?;
                if changed != 1 {
                    return Err(StoreError::State("provisional seal transition raced"));
                }
            }
            _ => return Err(StoreError::State("provisional upload cannot accept a seal")),
        }
        transaction.commit()?;
        Ok(())
    }

    /// Record the producer/consumer clock offset before staging begins.
    pub(crate) fn record_provisional_clock_offset(
        &self,
        idempotency: &Id32,
        token: u64,
        clock_offset_ns: u64,
    ) -> Result<(), StoreError> {
        let connection = self.lock_catalog()?;
        let changed = connection.execute(
            "UPDATE uploads SET provenance_clock_offset_ns=?3,updated_ns=?4 WHERE tenant=?1 AND idempotency_key=?2 AND state='RECEIVING' AND session_token=?5",
            params![
                self.tenant_namespace.as_slice(),
                idempotency.as_slice(),
                clock_offset_ns as i64,
                now_ns() as i64,
                token as i64,
            ],
        )?;
        if changed != 1 {
            return Err(StoreError::State(
                "provisional clock offset requires the active session",
            ));
        }
        Ok(())
    }

    /// Seal-time metadata persisted with the upload row, for the decode side.
    pub fn provisional_upload_metadata(
        &self,
        idempotency: &Id32,
    ) -> Result<Option<ProvisionalUploadMetadata>, StoreError> {
        let connection = self.lock_catalog()?;
        let row: Option<(Option<i64>, i64, Option<i64>, i64)> = connection
            .query_row(
                "SELECT boundary_token_id,provenance_source_ns,provenance_clock_offset_ns,provenance_quiesced FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;
        row.map(|(boundary, source, offset, quiesced)| {
            Ok(ProvisionalUploadMetadata {
                boundary_token_id: boundary
                    .map(u32::try_from)
                    .transpose()
                    .map_err(|_| StoreError::State("catalog boundary token exceeds u32"))?,
                provenance: ProvisionalProvenance {
                    source_wall_clock_ns: source as u64,
                    clock_offset_ns: offset.map(|value| value as u64),
                    quiesced: quiesced != 0,
                },
            })
        })
        .transpose()
    }

    /// Lazily reap provisional uploads whose lease lapsed (producer died
    /// mid-upload). Reaping is identical to an abort: the reservation is
    /// released and staged files are removed.
    pub(crate) fn reap_expired_provisional_uploads(&self) -> Result<(), StoreError> {
        let expired: Vec<Id32> = {
            let connection = self.lock_catalog()?;
            let mut statement = connection.prepare(
                "SELECT idempotency_key FROM uploads WHERE tenant=?1 AND state IN ('INIT','RESERVED','RECEIVING','VERIFIED') AND lease_expires_ns<>0 AND lease_expires_ns<?2",
            )?;
            let rows = statement
                .query_map(
                    params![self.tenant_namespace.as_slice(), now_ns() as i64],
                    |row| row.get::<_, Vec<u8>>(0),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            rows.into_iter()
                .map(vec_id)
                .collect::<Result<Vec<_>, _>>()?
        };
        for idempotency in expired {
            self.finish_provisional_upload_dir(&idempotency)?;
            self.abort_upload(&idempotency)?;
            self.clear_provisional_ledger(&idempotency, false)?;
        }
        Ok(())
    }

    pub(crate) fn cancel_provisional_upload(
        &self,
        idempotency: &Id32,
        token: u64,
    ) -> Result<(), StoreError> {
        let row: Option<(String, i64)> = {
            let connection = self.lock_catalog()?;
            connection
                .query_row(
                    "SELECT state,session_token FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
                    params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?
        };
        if let Some((state, stored)) = row {
            if matches!(
                state.as_str(),
                "INIT" | "RESERVED" | "RECEIVING" | "VERIFIED"
            ) && stored as u64 != token
            {
                return Err(StoreError::State(
                    "stale provisional session token; cancel refused, reservation untouched",
                ));
            }
        }
        self.finish_provisional_upload_dir(idempotency)?;
        self.abort_upload(idempotency)?;
        self.clear_provisional_ledger(idempotency, false)
    }
}
