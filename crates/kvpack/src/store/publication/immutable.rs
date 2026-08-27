use super::*;

pub(super) struct DeferredImmutableDurability {
    published: bool,
    target_directory: Option<PathBuf>,
    partial_directory: Option<PathBuf>,
}

struct StagedImmutable {
    target: PathBuf,
    partial: Option<PathBuf>,
}

impl Drop for StagedImmutable {
    fn drop(&mut self) {
        if let Some(partial) = self.partial.take() {
            let _ = fs::remove_file(partial);
        }
    }
}

pub(super) fn transition(
    transaction: &Transaction<'_>,
    tenant: &Id32,
    idempotency: &Id32,
    from: &str,
    to: &str,
) -> Result<(), StoreError> {
    let changed = transaction.execute("UPDATE uploads SET state=?4,updated_ns=?5 WHERE tenant=?1 AND idempotency_key=?2 AND state=?3", params![tenant.as_slice(), idempotency.as_slice(), from, to, now_ns()])?;
    if changed != 1 {
        return Err(StoreError::State("invalid upload state transition"));
    }
    Ok(())
}

/// Proof that the current operation already verified the exact bytes behind
/// an immutable target it created itself (for example a provisional promotion
/// that re-hashed the staged object immediately before linking it).  Lets the
/// publish step skip the same-inode dedup re-read; the pre-existing-target
/// dedup branches keep verifying.
#[derive(Debug, Clone, Copy)]
pub(in crate::store) struct AlreadyVerifiedTarget;

pub(in crate::store) fn write_immutable(
    store: &LocalStore,
    kind: DurableObjectKind,
    partial_dir: &str,
    target: &Path,
    bytes: &[u8],
) -> Result<bool, StoreError> {
    let durability = write_immutable_deferred(store, kind, partial_dir, target, bytes, None)?;
    let published = durability.published;
    sync_immutable_batch(store, kind, std::slice::from_ref(&durability))?;
    Ok(published)
}

pub(in crate::store) fn write_immutable_deferred_cleanup(
    store: &LocalStore,
    kind: DurableObjectKind,
    partial_dir: &str,
    target: &Path,
    bytes: &[u8],
    verified: Option<AlreadyVerifiedTarget>,
) -> Result<bool, StoreError> {
    let durability = write_immutable_deferred(store, kind, partial_dir, target, bytes, verified)?;
    if !durability.published {
        return Ok(false);
    }
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::TargetDirectorySync),
    )?;
    fsync_dir(durability.target_directory.as_ref().unwrap())?;
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::PartialDirectorySync),
    )?;
    Ok(true)
}

pub(super) fn write_immutable_deferred(
    store: &LocalStore,
    kind: DurableObjectKind,
    partial_dir: &str,
    target: &Path,
    bytes: &[u8],
    verified: Option<AlreadyVerifiedTarget>,
) -> Result<DeferredImmutableDurability, StoreError> {
    let root = &store.config.object_root;
    if let Some(parent) = target.parent() {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(parent)
            .map_err(io_error("create object shard directory"))?;
    }
    if target.exists() {
        if verified.is_none() {
            verify_immutable_target(target, bytes)?;
        }
        return Ok(DeferredImmutableDurability {
            published: false,
            target_directory: None,
            partial_directory: None,
        });
    }
    let mut random = [0u8; 16];
    getrandom::fill(&mut random)
        .map_err(|_| StoreError::State("partial filename entropy failed"))?;
    let partial = root
        .join(partial_dir)
        .join(format!("{}.partial", hex(&random)));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::Create),
    )?;
    let mut file = options
        .open(&partial)
        .map_err(io_error("create private object partial"))?;
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::Write),
    )?;
    file.write_all(bytes)
        .map_err(io_error("write private object partial"))?;
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::FileSync),
    )?;
    file.sync_all()
        .map_err(io_error("fsync private object partial"))?;
    drop(file);
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::NoReplace),
    )?;
    let published = match fs::hard_link(&partial, target) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            verify_immutable_target(target, bytes)?;
            false
        }
        Err(source) => {
            return Err(StoreError::Io {
                op: "no-replace object publication",
                source,
            });
        }
    };
    fs::remove_file(&partial).map_err(io_error("unlink private object partial"))?;
    Ok(DeferredImmutableDurability {
        published,
        target_directory: published.then(|| target.parent().unwrap().to_path_buf()),
        partial_directory: published.then(|| root.join(partial_dir)),
    })
}

pub(super) fn sync_immutable_batch(
    store: &LocalStore,
    kind: DurableObjectKind,
    writes: &[DeferredImmutableDurability],
) -> Result<(), StoreError> {
    let target_directories: BTreeSet<_> = writes
        .iter()
        .filter_map(|write| write.target_directory.as_ref())
        .collect();
    let partial_directories: BTreeSet<_> = writes
        .iter()
        .filter_map(|write| write.partial_directory.as_ref())
        .collect();
    if target_directories.is_empty() {
        return Ok(());
    }
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::TargetDirectorySync),
    )?;
    for directory in target_directories {
        fsync_dir(directory)?;
    }
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::PartialDirectorySync),
    )?;
    for directory in partial_directories {
        fsync_dir(directory)?;
    }
    Ok(())
}

pub(super) fn write_immutable_batch(
    store: &LocalStore,
    kind: DurableObjectKind,
    partial_dir: &str,
    objects: &[(PathBuf, &[u8])],
) -> Result<(), StoreError> {
    write_immutable_batch_mode(store, kind, partial_dir, objects, false).map(drop)
}

/// Batch immutable publication returning the per-object `published` flags.
/// When `defer_partial_directory_sync` is set, the partial-directory fsync is
/// skipped exactly like `write_immutable_deferred_cleanup`: the caller owes an
/// explicit partial-directory checkpoint afterwards (the export commit path
/// runs `sync_export_partial_cleanup`).
pub(super) fn write_immutable_batch_mode(
    store: &LocalStore,
    kind: DurableObjectKind,
    partial_dir: &str,
    objects: &[(PathBuf, &[u8])],
    defer_partial_directory_sync: bool,
) -> Result<Vec<bool>, StoreError> {
    let mut staged = Vec::with_capacity(objects.len());
    for (target, bytes) in objects {
        staged.push(stage_immutable(store, kind, partial_dir, target, bytes)?);
    }
    for object in &staged {
        let Some(partial) = object.partial.as_ref() else {
            continue;
        };
        durability_fault(
            store,
            DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::FileSync),
        )?;
        fs::File::open(partial)
            .and_then(|file| file.sync_all())
            .map_err(io_error("fsync private object partial"))?;
    }

    let mut durability = Vec::with_capacity(staged.len());
    for (object, (_, bytes)) in staged.iter_mut().zip(objects) {
        let Some(partial) = object.partial.as_ref() else {
            durability.push(DeferredImmutableDurability {
                published: false,
                target_directory: None,
                partial_directory: None,
            });
            continue;
        };
        durability_fault(
            store,
            DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::NoReplace),
        )?;
        let linked = match fs::hard_link(partial, &object.target) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                verify_immutable_target(&object.target, bytes)?;
                false
            }
            Err(source) => {
                return Err(StoreError::Io {
                    op: "no-replace object publication",
                    source,
                });
            }
        };
        fs::remove_file(partial).map_err(io_error("unlink private object partial"))?;
        object.partial = None;
        durability.push(DeferredImmutableDurability {
            published: linked,
            target_directory: linked.then(|| object.target.parent().unwrap().to_path_buf()),
            partial_directory: linked.then(|| store.config.object_root.join(partial_dir)),
        });
    }
    let published = durability.iter().map(|write| write.published).collect();
    if defer_partial_directory_sync {
        let target_directories: BTreeSet<_> = durability
            .iter()
            .filter_map(|write| write.target_directory.as_ref())
            .collect();
        if !target_directories.is_empty() {
            durability_fault(
                store,
                DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::TargetDirectorySync),
            )?;
            for directory in target_directories {
                fsync_dir(directory)?;
            }
            durability_fault(
                store,
                DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::PartialDirectorySync),
            )?;
        }
    } else {
        sync_immutable_batch(store, kind, &durability)?;
    }
    Ok(published)
}

fn stage_immutable(
    store: &LocalStore,
    kind: DurableObjectKind,
    partial_dir: &str,
    target: &Path,
    bytes: &[u8],
) -> Result<StagedImmutable, StoreError> {
    if let Some(parent) = target.parent() {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(parent)
            .map_err(io_error("create object shard directory"))?;
    }
    if target.exists() {
        verify_immutable_target(target, bytes)?;
        return Ok(StagedImmutable {
            target: target.to_path_buf(),
            partial: None,
        });
    }
    let mut random = [0u8; 16];
    getrandom::fill(&mut random)
        .map_err(|_| StoreError::State("partial filename entropy failed"))?;
    let partial = store
        .config
        .object_root
        .join(partial_dir)
        .join(format!("{}.partial", hex(&random)));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true).mode(0o600);
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::Create),
    )?;
    let mut file = options
        .open(&partial)
        .map_err(io_error("create private object partial"))?;
    durability_fault(
        store,
        DurabilityFaultPoint::Immutable(kind, ImmutableFaultPhase::Write),
    )?;
    file.write_all(bytes)
        .map_err(io_error("write private object partial"))?;
    drop(file);
    Ok(StagedImmutable {
        target: target.to_path_buf(),
        partial: Some(partial),
    })
}

fn verify_immutable_target(target: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let mut existing = fs::File::open(target).map_err(io_error("open immutable target"))?;
    let metadata = existing
        .metadata()
        .map_err(io_error("inspect immutable target"))?;
    if !metadata.is_file() || metadata.len() != bytes.len() as u64 {
        return Err(StoreError::Authentication(
            "existing immutable target has different bounds",
        ));
    }
    let mut offset = 0usize;
    let mut buffer = [0u8; 64 * 1024];
    while offset < bytes.len() {
        let count = (bytes.len() - offset).min(buffer.len());
        existing
            .read_exact(&mut buffer[..count])
            .map_err(io_error("verify immutable target"))?;
        if buffer[..count] != bytes[offset..offset + count] {
            return Err(StoreError::Authentication(
                "existing immutable target has different bytes",
            ));
        }
        offset += count;
    }
    Ok(())
}

pub(super) fn durability_fault(
    store: &LocalStore,
    point: DurabilityFaultPoint,
) -> Result<(), StoreError> {
    #[cfg(test)]
    {
        let mut selected = store
            .durability_fault
            .lock()
            .map_err(|_| StoreError::State("durability fault mutex poisoned"))?;
        if selected.as_ref() == Some(&point) {
            selected.take();
            return Err(StoreError::Io {
                op: "injected durability fault",
                source: std::io::Error::other(format!("{point:?}")),
            });
        }
    }
    #[cfg(not(test))]
    let _ = (store, point);
    Ok(())
}

pub(super) fn quarantine_object(store: &LocalStore, path: &Path) -> Result<(), StoreError> {
    if !path.exists() {
        return Ok(());
    }
    let file_bytes = path
        .metadata()
        .map_err(io_error("inspect duplicate object"))?
        .len();
    let mut entry_id = [0u8; 32];
    getrandom::fill(&mut entry_id)
        .map_err(|_| StoreError::State("quarantine identity entropy failed"))?;
    let path_token = format!("{}.duplicate.quarantine", hex(&entry_id));
    let directory = store.config.object_root.join("quarantine");
    let destination: PathBuf = directory.join(&path_token);
    let mut connection = store.lock_catalog()?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let audit_events = [AuditEventKey::new(
        AuditEventKind::Quarantined,
        AuditObjectKind::Quarantine,
        entry_id,
        0,
    )];
    if audit::preflight_events(&transaction, &store.tenant_namespace, &audit_events)?
        == AuditCapacity::Backpressured
    {
        transaction.commit()?;
        let _ = store.telemetry.record_audit(AuditOutcome::Backpressure);
        return Err(StoreError::Busy);
    }
    fs::rename(path, destination).map_err(io_error("quarantine duplicate object"))?;
    fsync_dir(path.parent().unwrap())?;
    fsync_dir(&directory)?;
    let created = now_ns();
    transaction.execute(
        "INSERT INTO quarantine_entries(tenant,entry_id,object_kind,path_token,file_bytes,created_ns,expires_ns,reason) VALUES(?1,?2,'duplicate_chunk',?3,?4,?5,?6,'conflicting chunk publication')",
        params![
            store.tenant_namespace.as_slice(),
            entry_id.as_slice(),
            path_token,
            file_bytes,
            created,
            created.saturating_add(24 * 60 * 60 * 1_000_000_000),
        ],
    )?;
    let enqueued = audit::append_events(
        &transaction,
        &store.tenant_namespace,
        &audit_events,
        created,
    )?;
    transaction.commit()?;
    drop(connection);
    let _ = store
        .telemetry
        .record_lifecycle(CacheLifecycle::Quarantined);
    let _ = store
        .telemetry
        .record_audit_count(AuditOutcome::Enqueued, enqueued);
    store.maintain_quarantine_cap()
}
