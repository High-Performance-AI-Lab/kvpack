use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

use kvpack_core::Id32;

use crate::StoreError;

enum FillState {
    Running,
    Ready(Arc<[u8]>),
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct FillCancellation {
    cancelled: Arc<AtomicBool>,
}

impl FillCancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

struct FillEntry {
    state: Mutex<FillState>,
    changed: Condvar,
}

struct LeaderGuard {
    entry: Arc<FillEntry>,
    completed: bool,
}

impl LeaderGuard {
    fn complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for LeaderGuard {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if let Ok(mut state) = self.entry.state.lock() {
            *state = FillState::Failed;
            self.entry.changed.notify_all();
        }
    }
}

/// Process-local singleflight for demand fills. The map stores weak entries so
/// completed object bytes are not retained as a second cache.
#[derive(Default)]
pub struct SingleflightFill {
    entries: Mutex<HashMap<Id32, Weak<FillEntry>>>,
}

impl SingleflightFill {
    pub fn get_or_fill(
        &self,
        object_key: Id32,
        fetch: impl FnOnce() -> Result<Vec<u8>, StoreError>,
    ) -> Result<Arc<[u8]>, StoreError> {
        self.get_or_fill_cancellable(object_key, &FillCancellation::default(), fetch)
    }

    pub fn get_or_fill_cancellable(
        &self,
        object_key: Id32,
        cancellation: &FillCancellation,
        fetch: impl FnOnce() -> Result<Vec<u8>, StoreError>,
    ) -> Result<Arc<[u8]>, StoreError> {
        let (entry, leader) = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| StoreError::State("singleflight map mutex poisoned"))?;
            if let Some(entry) = entries.get(&object_key).and_then(Weak::upgrade) {
                (entry, false)
            } else {
                let entry = Arc::new(FillEntry {
                    state: Mutex::new(FillState::Running),
                    changed: Condvar::new(),
                });
                entries.insert(object_key, Arc::downgrade(&entry));
                (entry, true)
            }
        };

        if leader {
            let mut guard = LeaderGuard {
                entry: Arc::clone(&entry),
                completed: false,
            };
            if cancellation.is_cancelled() {
                return Err(StoreError::Cancelled);
            }
            return match fetch() {
                Ok(bytes) => {
                    let bytes: Arc<[u8]> = bytes.into();
                    let mut state = entry
                        .state
                        .lock()
                        .map_err(|_| StoreError::State("singleflight state mutex poisoned"))?;
                    *state = FillState::Ready(Arc::clone(&bytes));
                    entry.changed.notify_all();
                    guard.complete();
                    Ok(bytes)
                }
                Err(error) => {
                    if let Ok(mut state) = entry.state.lock() {
                        *state = FillState::Failed;
                        entry.changed.notify_all();
                    }
                    guard.complete();
                    Err(error)
                }
            };
        }

        let mut state = entry
            .state
            .lock()
            .map_err(|_| StoreError::State("singleflight state mutex poisoned"))?;
        loop {
            match &*state {
                FillState::Running => {
                    if cancellation.is_cancelled() {
                        return Err(StoreError::Cancelled);
                    }
                    let waited = entry
                        .changed
                        .wait_timeout(state, std::time::Duration::from_millis(50))
                        .map_err(|_| StoreError::State("singleflight state mutex poisoned"))?;
                    state = waited.0;
                }
                FillState::Ready(bytes) => return Ok(Arc::clone(bytes)),
                FillState::Failed => {
                    return Err(StoreError::State("singleflight upstream fill failed"));
                }
            }
        }
    }
}
