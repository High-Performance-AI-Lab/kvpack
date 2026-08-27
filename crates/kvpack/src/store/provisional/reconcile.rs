use super::*;

impl LocalStore {
    pub(crate) fn finish_provisional_upload_dir(
        &self,
        idempotency: &Id32,
    ) -> Result<(), StoreError> {
        let directory = self.provisional_upload_path(idempotency);
        match fs::read_dir(&directory) {
            Ok(entries) => {
                for entry in entries {
                    let entry = entry.map_err(io_error("read provisional upload cleanup"))?;
                    if !entry
                        .file_type()
                        .map_err(io_error("inspect provisional upload cleanup entry"))?
                        .is_file()
                    {
                        return Err(StoreError::State(
                            "provisional upload cleanup encountered a non-file",
                        ));
                    }
                    fs::remove_file(entry.path())
                        .map_err(io_error("remove provisional upload object"))?;
                }
                fs::remove_dir(&directory)
                    .map_err(io_error("remove provisional upload directory"))?;
                fsync_dir(&self.config.object_root.join("uploads"))?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(StoreError::Io {
                    op: "open provisional upload cleanup directory",
                    source,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn finish_provisional_ledger(&self, idempotency: &Id32) -> Result<(), StoreError> {
        self.clear_provisional_ledger(idempotency, true)
    }

    pub(in crate::store) fn reconcile_provisional_uploads(
        &self,
        bound: usize,
    ) -> Result<(), StoreError> {
        let uploads = self.config.object_root.join("uploads");
        let mut seen = 0usize;
        for entry in fs::read_dir(&uploads).map_err(io_error("read upload work directory"))? {
            seen = seen
                .checked_add(1)
                .ok_or(StoreError::State("upload reconciliation count overflow"))?;
            if seen > bound {
                return Err(StoreError::State(
                    "upload reconciliation exceeded its configured bound",
                ));
            }
            let entry = entry.map_err(io_error("read upload work entry"))?;
            let file_type = entry
                .file_type()
                .map_err(io_error("inspect upload work entry"))?;
            if file_type.is_symlink() {
                return Err(StoreError::State(
                    "upload work directory contains a symlink",
                ));
            }
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| StoreError::State("upload work name is not UTF-8"))?;
            if file_type.is_file() && valid_import_partial_name(&name) {
                continue;
            }
            if !file_type.is_dir() || !valid_hex_id(&name) {
                return Err(StoreError::State(
                    "upload work directory contains an unknown entry",
                ));
            }
            let idempotency = decode_hex_id(&name)?;
            // Chunk files are bounded per upload directory, not globally: a
            // single maximum bundle legitimately stages ~3,072 chunk files.
            let mut entries = 0usize;
            for child in fs::read_dir(entry.path())
                .map_err(io_error("read provisional restart directory"))?
            {
                entries = entries
                    .checked_add(1)
                    .ok_or(StoreError::State("provisional cleanup count overflow"))?;
                if entries > PROVISIONAL_DIRECTORY_ENTRY_BOUND {
                    return Err(StoreError::State(
                        "provisional cleanup exceeded its per-upload bound",
                    ));
                }
                let child = child.map_err(io_error("read provisional restart entry"))?;
                let child_type = child
                    .file_type()
                    .map_err(io_error("inspect provisional restart entry"))?;
                let child_name = child
                    .file_name()
                    .into_string()
                    .map_err(|_| StoreError::State("provisional object name is not UTF-8"))?;
                if !child_type.is_file() || !valid_provisional_chunk_name(&child_name) {
                    return Err(StoreError::State(
                        "provisional restart directory contains an unknown entry",
                    ));
                }
                fs::remove_file(child.path())
                    .map_err(io_error("remove provisional restart object"))?;
            }
            fs::remove_dir(entry.path())
                .map_err(io_error("remove provisional restart directory"))?;
            self.abort_upload(&idempotency)?;
            self.clear_provisional_ledger(&idempotency, false)?;
        }
        fsync_dir(&uploads)?;
        Ok(())
    }

    pub(crate) fn clear_provisional_ledger(
        &self,
        idempotency: &Id32,
        published: bool,
    ) -> Result<(), StoreError> {
        let expected_state = if published { "PUBLISHED" } else { "ABORTED" };
        let mut connection = self.lock_catalog()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let state: String = transaction.query_row(
            "SELECT state FROM uploads WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
            |row| row.get(0),
        )?;
        if state != expected_state {
            return Err(StoreError::State(
                "provisional ledger cannot be cleared in its current state",
            ));
        }
        transaction.execute(
            "DELETE FROM upload_chunks WHERE tenant=?1 AND idempotency_key=?2",
            params![self.tenant_namespace.as_slice(), idempotency.as_slice()],
        )?;
        transaction.execute(
            "UPDATE uploads SET next_chunk_ordinal=0,updated_ns=?3 WHERE tenant=?1 AND idempotency_key=?2 AND state=?4",
            params![
                self.tenant_namespace.as_slice(),
                idempotency.as_slice(),
                now_ns(),
                expected_state,
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }
}

fn valid_hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_import_partial_name(value: &str) -> bool {
    value
        .strip_suffix(".kvpack.partial")
        .is_some_and(valid_hex_id)
}

fn valid_provisional_chunk_name(value: &str) -> bool {
    value.strip_suffix(".kvchunk").is_some_and(|ordinal| {
        ordinal.len() == 20 && ordinal.bytes().all(|byte| byte.is_ascii_digit())
    })
}

fn decode_hex_id(value: &str) -> Result<Id32, StoreError> {
    if !valid_hex_id(value) {
        return Err(StoreError::State("invalid provisional upload identity"));
    }
    let mut result = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let pair = std::str::from_utf8(pair)
            .map_err(|_| StoreError::State("provisional upload identity is not ASCII"))?;
        result[index] = u8::from_str_radix(pair, 16)
            .map_err(|_| StoreError::State("invalid provisional upload identity"))?;
    }
    Ok(result)
}
