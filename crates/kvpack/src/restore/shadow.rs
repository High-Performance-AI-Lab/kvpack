use super::*;

/// One verified chunk write recorded into the shadow stage. The plaintext is
/// already authenticated by the restore pipeline; promotion replays it
/// verbatim from memory without any further store read.
#[derive(Debug)]
struct StagedChunk {
    state: StateKey,
    offset: u64,
    plaintext: Vec<u8>,
}

/// Engine-invisible staging sink. Writes land in private heap buffers and
/// `commit_restore` only finalizes the shadow; nothing becomes visible to a
/// running engine through this sink, so the shadow-only invariant holds by
/// construction. `abort_restore` discards every staged byte.
#[derive(Debug, Default)]
struct ShadowStageSink {
    state_bytes: BTreeMap<StateKey, u64>,
    staged: Vec<StagedChunk>,
    committed: bool,
}

impl VerifiedRestoreSink for ShadowStageSink {
    fn begin_restore(
        &mut self,
        artifact: Id32,
        states: &[RestoreStatePlan],
    ) -> Result<(), StoreError> {
        if artifact == [0; 32] || states.is_empty() {
            return Err(StoreError::Expectation(
                "shadow restore requires an authenticated artifact and states",
            ));
        }
        self.state_bytes.clear();
        self.staged.clear();
        self.committed = false;
        for state in states {
            if self
                .state_bytes
                .insert(state.declaration.key.clone(), state.plaintext_bytes)
                .is_some()
            {
                return Err(StoreError::Expectation(
                    "shadow restore plan repeats a state key",
                ));
            }
        }
        Ok(())
    }

    fn write_verified_chunk(
        &mut self,
        state: &StateKey,
        logical_offset: u64,
        plaintext: &[u8],
    ) -> Result<(), StoreError> {
        let total = self
            .state_bytes
            .get(state)
            .copied()
            .ok_or(StoreError::State(
                "shadow write targets an undeclared state",
            ))?;
        let end = logical_offset
            .checked_add(plaintext.len() as u64)
            .ok_or(StoreError::State("shadow write offset overflow"))?;
        if end > total {
            return Err(StoreError::State(
                "shadow write exceeds the declared state bytes",
            ));
        }
        self.staged.push(StagedChunk {
            state: state.clone(),
            offset: logical_offset,
            plaintext: plaintext.to_vec(),
        });
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        // Finalize the shadow only; visibility is the promotion path's job.
        self.committed = true;
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.state_bytes.clear();
        self.staged.clear();
        self.committed = false;
    }
}

/// An abort-only speculative pre-stage of a prefix restore. The handle owns
/// the fully verified plaintext of one authenticated plan plus the restore
/// reservation and source pins a normal restore would hold, so a promotion
/// transfers the staged buffers into the engine sink without re-reading the
/// store. Promotion is gated by `promote_if`: any key other than the
/// authenticated manifest the shadow was built for aborts the shadow and
/// fails closed. `abort` discards every staged byte and releases the
/// reservation; dropping an unsettled handle deliberately retains the
/// reservation (recoverable by `restore_id`), matching `InstalledRestore`.
#[must_use = "a shadow restore is abort-only unless promoted; dropping retains its reservation"]
#[derive(Debug)]
pub struct ShadowRestoreHandle {
    manifest_id: Id32,
    matched_cut: InputCutId,
    states: Vec<RestoreStatePlan>,
    staged: Vec<StagedChunk>,
    staged_bytes: u64,
    install: Option<InstalledRestore>,
}

impl AuthenticatedRestorePlan {
    /// Pre-stage this plan into an engine-invisible shadow restore. The full
    /// verification pipeline (authentication, pin, decode) runs exactly once,
    /// here; a later promotion replays the recorded writes from memory.
    pub fn prestage_shadow(
        &self,
        cancellation: &RestoreCancellation,
    ) -> Result<ShadowRestoreHandle, StoreError> {
        let mut stage = ShadowStageSink::default();
        let install = self.restore_sequential(&mut stage, cancellation)?;
        if !stage.committed {
            return Err(StoreError::State("shadow stage did not finalize"));
        }
        let staged_bytes = stage
            .staged
            .iter()
            .try_fold(0u64, |sum, chunk| {
                sum.checked_add(chunk.plaintext.len() as u64)
            })
            .ok_or(StoreError::State("shadow staged byte total overflow"))?;
        if staged_bytes != self.realized_schema.complete_restored_bytes {
            return Err(StoreError::Authentication(
                "shadow staged bytes disagree with the authenticated schema",
            ));
        }
        Ok(ShadowRestoreHandle {
            manifest_id: self.manifest_id,
            matched_cut: self.matched_cut,
            states: self.states.clone(),
            staged: stage.staged,
            staged_bytes,
            install: Some(install),
        })
    }
}

impl ShadowRestoreHandle {
    /// The authenticated manifest identity this shadow was built for; this is
    /// the exact key `promote_if` gates on.
    pub fn manifest_id(&self) -> Id32 {
        self.manifest_id
    }

    pub fn matched_cut(&self) -> InputCutId {
        self.matched_cut
    }

    /// Verified plaintext bytes currently staged off-heap.
    pub fn staged_bytes(&self) -> u64 {
        self.staged_bytes
    }

    /// The restore reservation identity, for uncertainty recovery after a
    /// dropped handle (see `LocalStore::acknowledge_engine_free`).
    pub fn restore_id(&self) -> Id32 {
        self.install
            .as_ref()
            .map(InstalledRestore::restore_id)
            .unwrap_or([0; 32])
    }

    /// Promote the shadow into `sink` if and only if `exact_key` is the
    /// authenticated manifest the shadow was built for. Promotion replays the
    /// staged writes from memory and commits the sink; it performs no store
    /// reads. A mismatched key, a cancellation, or any sink failure aborts the
    /// shadow completely and fails closed.
    pub fn promote_if(
        mut self,
        exact_key: Id32,
        sink: &mut dyn VerifiedRestoreSink,
        cancellation: &RestoreCancellation,
    ) -> Result<InstalledRestore, StoreError> {
        if exact_key != self.manifest_id {
            let _ = self.abort_inner();
            return Err(StoreError::Authentication(
                "shadow restore promoted for an artifact it was not staged for",
            ));
        }
        if cancellation.is_cancelled() {
            let _ = self.abort_inner();
            return Err(StoreError::Cancelled);
        }
        if let Err(error) = sink.begin_restore(self.manifest_id, &self.states) {
            sink.abort_restore();
            let _ = self.abort_inner();
            return Err(error);
        }
        for chunk in &self.staged {
            if let Err(error) =
                sink.write_verified_chunk(&chunk.state, chunk.offset, &chunk.plaintext)
            {
                sink.abort_restore();
                let _ = self.abort_inner();
                return Err(error);
            }
        }
        if cancellation.is_cancelled() {
            sink.abort_restore();
            let _ = self.abort_inner();
            return Err(StoreError::Cancelled);
        }
        if let Err(error) = sink.commit_restore() {
            sink.reset_restore();
            let _ = self.abort_inner();
            return Err(error);
        }
        self.install.take().ok_or(StoreError::State(
            "shadow restore handle is already settled",
        ))
    }

    /// Discard every staged byte and release the restore reservation and
    /// source pins. Nothing was ever engine-visible, so abort is complete.
    pub fn abort(mut self) -> Result<(), StoreError> {
        self.abort_inner()
    }

    fn abort_inner(&mut self) -> Result<(), StoreError> {
        self.staged.clear();
        self.staged_bytes = 0;
        match self.install.take() {
            Some(install) => install.engine_free(),
            None => Ok(()),
        }
    }
}
