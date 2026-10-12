use super::*;

pub struct ExportStateWriter<'a> {
    pub(super) session: &'a mut ExportSession,
    pub(super) declaration: ExportStateDeclaration,
    pub(super) buffer: Vec<u8>,
    pub(super) chunks: Vec<(ChunkSpan, Option<ChunkRef>)>,
    pub(super) pending: Vec<(usize, ChunkObject)>,
    pub(super) pending_bytes: u64,
    pub(super) flushed_bytes: u64,
    pub(super) expected_bytes: u64,
    pub(super) bytes_per_token: u64,
    pub(super) maximum_chunk_tokens: u64,
    pub(super) finished: bool,
}

impl ExportStateWriter<'_> {
    pub fn write_all(&mut self, mut bytes: &[u8]) -> Result<(), StoreError> {
        if self.finished || self.session.poisoned {
            return Err(StoreError::Poisoned("export state writer is not writable"));
        }
        let received = self
            .flushed_bytes
            .checked_add(self.buffer.len() as u64)
            .and_then(|value| value.checked_add(bytes.len() as u64))
            .ok_or(StoreError::State("export state byte count overflow"))?;
        if received > self.expected_bytes {
            self.session.poisoned = true;
            return Err(StoreError::State(
                "export state received more bytes than its declared source bound",
            ));
        }
        let source_bytes = bytes.len() as u64;
        while !bytes.is_empty() {
            let target = next_chunk_bytes(
                self.flushed_bytes / self.bytes_per_token,
                self.session.final_token_count,
                self.bytes_per_token,
                self.maximum_chunk_tokens,
            )?;
            if target == 0 || self.buffer.len() > target {
                self.session.poisoned = true;
                return Err(StoreError::State(
                    "export chunk target is inconsistent with source bounds",
                ));
            }
            let take = (target - self.buffer.len()).min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buffer.len() == target {
                self.flush()?;
            }
        }
        let _ = self
            .session
            .store
            .telemetry
            .add_bytes(ByteCounter::SourceRead, source_bytes);
        Ok(())
    }

    /// Consume exactly one forward-only source. A short source, one extra byte,
    /// or any read error poisons the complete export session.
    pub fn write_source(mut self, source: &mut impl Read) -> Result<(), StoreError> {
        let mut scratch = [0u8; 64 * 1024];
        loop {
            let received = self.flushed_bytes + self.buffer.len() as u64;
            let remaining = self.expected_bytes.saturating_sub(received);
            if remaining == 0 {
                break;
            }
            let limit = usize::try_from(remaining.min(scratch.len() as u64))
                .map_err(|_| StoreError::State("source read bound exceeds usize"))?;
            let read = loop {
                match source.read(&mut scratch[..limit]) {
                    Ok(read) => break read,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(source) => {
                        self.session.poisoned = true;
                        return Err(StoreError::Io {
                            op: "read export state source",
                            source,
                        });
                    }
                }
            };
            if read == 0 {
                self.session.poisoned = true;
                return Err(StoreError::State(
                    "export state source ended before its declared bound",
                ));
            }
            self.write_all(&scratch[..read])?;
        }
        let mut extra = [0u8; 1];
        loop {
            match source.read(&mut extra) {
                Ok(0) => break,
                Ok(_) => {
                    self.session.poisoned = true;
                    return Err(StoreError::State(
                        "export state source exceeded its declared bound",
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(source) => {
                    self.session.poisoned = true;
                    return Err(StoreError::Io {
                        op: "check export state source bound",
                        source,
                    });
                }
            }
        }
        self.finish()
    }

    fn flush(&mut self) -> Result<(), StoreError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if self.buffer.len() as u64 % self.bytes_per_token != 0 {
            self.session.poisoned = true;
            return Err(StoreError::State(
                "export chunk boundary splits one logical state token",
            ));
        }
        let plaintext = std::mem::take(&mut self.buffer);
        let token_start = self.flushed_bytes / self.bytes_per_token;
        let token_count = plaintext.len() as u64 / self.bytes_per_token;
        let token_end = token_start
            .checked_add(token_count)
            .ok_or(StoreError::State("export chunk token range overflow"))?;
        if token_count == 0
            || (token_start / PREFIX_BLOCK_TOKENS as u64
                != (token_end - 1) / PREFIX_BLOCK_TOKENS as u64)
        {
            self.session.poisoned = true;
            return Err(StoreError::State(
                "export chunk crossed an admitted checkpoint boundary",
            ));
        }
        let span = ChunkSpan {
            token_start,
            token_count,
            plaintext_offset: self.flushed_bytes,
            plaintext_bytes: u32::try_from(plaintext.len())
                .map_err(|_| StoreError::State("export chunk plaintext exceeds u32"))?,
        };
        match self
            .session
            .store_chunk(&self.declaration.key, &plaintext, span)
        {
            Ok(staged) => {
                self.flushed_bytes = self
                    .flushed_bytes
                    .checked_add(plaintext.len() as u64)
                    .ok_or(StoreError::State("export state size overflow"))?;
                match staged {
                    ExportStagedChunk::Deduplicated(reference) => {
                        self.chunks.push((span, Some(reference)));
                    }
                    ExportStagedChunk::Encoded(object) => {
                        // Queue for the batched put: one directory sync set
                        // and one catalog transaction per
                        // CHUNK_PUT_BATCH_BYTES (and always at state end)
                        // instead of a 4-6 fsync storm per chunk.
                        self.pending_bytes = self
                            .pending_bytes
                            .checked_add(object.bytes.len() as u64)
                            .ok_or(StoreError::State("pending chunk byte total overflow"))?;
                        self.pending.push((self.chunks.len(), object));
                        self.chunks.push((span, None));
                    }
                }
                let capacity = next_chunk_bytes(
                    self.flushed_bytes / self.bytes_per_token,
                    self.session.final_token_count,
                    self.bytes_per_token,
                    self.maximum_chunk_tokens,
                )?;
                self.buffer = Vec::with_capacity(capacity);
                if self.pending_bytes >= CHUNK_PUT_BATCH_BYTES {
                    self.flush_pending()?;
                }
                Ok(())
            }
            Err(error) => {
                self.session.poisoned = true;
                Err(error)
            }
        }
    }

    fn flush_pending(&mut self) -> Result<(), StoreError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut self.pending);
        self.pending_bytes = 0;
        let objects: Vec<&ChunkObject> = pending.iter().map(|(_, object)| object).collect();
        let references = match self.session.store.put_chunks_batch_with_retention(
            &objects,
            self.session.store.key_epoch(),
            self.session.policy.retention,
            true,
        ) {
            Ok(references) => references,
            Err(error) => {
                self.session.poisoned = true;
                return Err(error);
            }
        };
        for ((slot, object), reference) in pending.iter().zip(references) {
            if reference.chunk_id != object.chunk_id
                || reference.plaintext_bytes != object.plaintext_bytes
            {
                self.session.poisoned = true;
                return Err(StoreError::Authentication(
                    "deduplicated export chunk metadata mismatch",
                ));
            }
            self.chunks[*slot].1 = Some(reference);
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), StoreError> {
        let received = self.flushed_bytes + self.buffer.len() as u64;
        if received != self.expected_bytes {
            self.session.poisoned = true;
            return Err(StoreError::State(
                "export state ended before its exact source bound",
            ));
        }
        self.flush()?;
        self.flush_pending()?;
        if self.flushed_bytes != self.expected_bytes || self.chunks.len() > MAX_CHUNKS_PER_STATE {
            self.session.poisoned = true;
            return Err(StoreError::State(
                "export state chunk inventory exceeds or disagrees with its bound",
            ));
        }
        let chunks = std::mem::take(&mut self.chunks)
            .into_iter()
            .map(|(span, reference)| {
                reference
                    .map(|reference| StoredStateChunk { reference, span })
                    .ok_or(StoreError::State("export chunk reference was not resolved"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.session.completed.push(StoredState {
            key: self.declaration.key.clone(),
            chunks,
        });
        self.session.next_state += 1;
        self.finished = true;
        Ok(())
    }
}

impl Drop for ExportStateWriter<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.session.poisoned = true;
        }
    }
}
