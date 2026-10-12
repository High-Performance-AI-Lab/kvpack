use kvpack_core::{Id32, StateDeclaration, StateKey};

use crate::StoreError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreStatePlan {
    pub declaration: StateDeclaration,
    pub plaintext_bytes: u64,
    pub physical_span_bytes: u64,
    pub atomic_group: u32,
    pub chunk_count: usize,
}

/// Engine-owned transactional shadow-buffer interface.  Implementations must
/// keep writes invisible to the running engine until `commit_restore` and must
/// discard every shadow allocation on `abort_restore`.
pub trait VerifiedRestoreSink {
    fn begin_restore(
        &mut self,
        artifact: Id32,
        states: &[RestoreStatePlan],
    ) -> Result<(), StoreError>;
    fn write_verified_chunk(
        &mut self,
        state: &StateKey,
        logical_offset: u64,
        plaintext: &[u8],
    ) -> Result<(), StoreError>;
    fn commit_restore(&mut self) -> Result<(), StoreError>;
    fn abort_restore(&mut self);

    /// Reset live engine state after a commit attempt fails. Implementations
    /// with transactional commit may use the same operation as abort.
    fn reset_restore(&mut self) {
        self.abort_restore();
    }
}
