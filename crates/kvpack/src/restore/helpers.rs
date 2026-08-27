use super::*;

pub(super) fn tier_order(tier: RestoreTier) -> u8 {
    match tier {
        RestoreTier::Resident => 0,
        RestoreTier::Local => 1,
        RestoreTier::Gateway => 2,
        RestoreTier::Recompute => 3,
    }
}

pub(super) fn load_authenticated_manifest(
    store: &Arc<LocalStore>,
    manifest_id: Id32,
    minimum_key_epoch: u64,
    context: &ValidationContext,
) -> Result<AuthenticatedManifest, StoreError> {
    let bytes = store.read_manifest_bytes(&manifest_id)?;
    let header = inspect_pack_header(&bytes)?;
    if header.tenant_namespace != store.tenant_namespace() {
        return Err(StoreError::Authentication(
            "manifest candidate belongs to another tenant",
        ));
    }
    if header.key_epoch < minimum_key_epoch || header.key_epoch > store.key_epoch() {
        return Err(StoreError::NotFound);
    }
    let keys = store.schedule(header.key_epoch)?;
    let manifest = decode_authenticated_pack(&bytes, &keys, context)?;
    if kvpack_core::manifest_id(&manifest.encode_canonical()?) != manifest_id {
        return Err(StoreError::Authentication(
            "manifest candidate identity does not match its catalog row",
        ));
    }
    Ok(AuthenticatedManifest {
        manifest_id,
        manifest: Arc::new(manifest),
    })
}

pub(super) fn authenticate_chain(
    store: &Arc<LocalStore>,
    leaf_id: Id32,
    minimum_key_epoch: u64,
    expected: Option<(Id32, Id32, InputCutId)>,
    context: &ValidationContext,
) -> Result<Vec<AuthenticatedManifest>, StoreError> {
    let mut cache = BTreeMap::new();
    authenticate_chain_cached(
        store,
        leaf_id,
        minimum_key_epoch,
        expected,
        context,
        &mut cache,
    )
}

pub(super) fn authenticate_chain_cached(
    store: &Arc<LocalStore>,
    leaf_id: Id32,
    minimum_key_epoch: u64,
    expected: Option<(Id32, Id32, InputCutId)>,
    context: &ValidationContext,
    cache: &mut BTreeMap<Id32, AuthenticatedManifest>,
) -> Result<Vec<AuthenticatedManifest>, StoreError> {
    let leaf =
        load_authenticated_manifest_cached(store, leaf_id, minimum_key_epoch, context, cache)?;
    if let Some((semantic, family, cut)) = expected {
        if semantic_model_id(&leaf.manifest.semantic_model) != semantic
            || representation_family_id(&leaf.manifest.family)? != family
            || leaf.manifest.input_cut != cut
        {
            return Err(StoreError::Authentication(
                "catalog candidate does not authenticate the matched prefix",
            ));
        }
    }
    let leaf_semantic = leaf.manifest.semantic_model;
    let leaf_family = leaf.manifest.family.clone();
    let leaf_tenant = leaf.manifest.tenant_namespace;
    let leaf_epoch = leaf.manifest.key_epoch;
    let leaf_depth = leaf.manifest.realized_schema.kind.depth();
    let mut reverse = vec![leaf];
    let mut seen = BTreeSet::from([leaf_id]);
    loop {
        let current = reverse.last().unwrap();
        let ManifestKind::Delta {
            parent,
            parent_cut,
            depth,
        } = current.manifest.realized_schema.kind.clone()
        else {
            break;
        };
        if depth == 0 || depth > MAX_DELTA_DEPTH || !seen.insert(parent) {
            return Err(StoreError::Authentication(
                "manifest parent chain depth or cycle is invalid",
            ));
        }
        let parent_manifest =
            load_authenticated_manifest_cached(store, parent, minimum_key_epoch, context, cache)?;
        if parent_manifest.manifest.input_cut != parent_cut
            || parent_manifest.manifest.semantic_model != leaf_semantic
            || parent_manifest.manifest.family != leaf_family
            || parent_manifest.manifest.tenant_namespace != leaf_tenant
            || parent_manifest.manifest.key_epoch != leaf_epoch
            || parent_manifest.manifest.realized_schema.kind.depth() + 1 != depth
        {
            return Err(StoreError::Authentication(
                "manifest parent is incompatible with its authenticated child",
            ));
        }
        reverse.push(parent_manifest);
    }
    reverse.reverse();
    if !matches!(reverse[0].manifest.realized_schema.kind, ManifestKind::Full)
        || reverse.len() != leaf_depth as usize + 1
    {
        return Err(StoreError::Authentication(
            "manifest chain does not terminate at the declared full root",
        ));
    }
    Ok(reverse)
}

fn load_authenticated_manifest_cached(
    store: &Arc<LocalStore>,
    manifest_id: Id32,
    minimum_key_epoch: u64,
    context: &ValidationContext,
    cache: &mut BTreeMap<Id32, AuthenticatedManifest>,
) -> Result<AuthenticatedManifest, StoreError> {
    if let Some(manifest) = cache.get(&manifest_id) {
        return Ok(manifest.clone());
    }
    let manifest = load_authenticated_manifest(store, manifest_id, minimum_key_epoch, context)?;
    cache.insert(manifest_id, manifest.clone());
    Ok(manifest)
}

#[derive(Debug, Clone)]
pub(super) enum CachedChunkAvailability {
    Missing,
    Available(StoredChunkAvailabilityRow),
    /// Tombstone-rung row: the chained key and token-cut catalog row are
    /// retained, but the bytes were dropped by fidelity demotion.  Restore
    /// planning marks these chunks as guided-recompute candidates and never
    /// serves bytes for them.
    Tombstoned(StoredChunkAvailabilityRow),
}

/// Disposition of every chunk referenced by an authenticated chain.
#[derive(Debug, Clone, Default)]
pub(super) struct ChainChunkDisposition {
    /// Every referenced chunk row is present (available or tombstoned).
    pub(super) complete: bool,
    /// Object keys whose local bytes were dropped by fidelity demotion.
    pub(super) tombstoned: BTreeSet<Id32>,
}

pub(super) fn chain_chunk_disposition(
    store: &Arc<LocalStore>,
    chain: &[AuthenticatedManifest],
) -> Result<ChainChunkDisposition, StoreError> {
    let mut cache = BTreeMap::new();
    chain_chunk_disposition_cached(store, chain, &mut cache)
}

pub(super) fn chain_chunk_disposition_cached(
    store: &Arc<LocalStore>,
    chain: &[AuthenticatedManifest],
    cache: &mut BTreeMap<Id32, CachedChunkAvailability>,
) -> Result<ChainChunkDisposition, StoreError> {
    let fully_cached = chain.iter().all(|entry| {
        entry.manifest.states.iter().all(|state| {
            state
                .chunks
                .iter()
                .all(|reference| cache.contains_key(&reference.object_key))
        })
    });
    if fully_cached {
        return validate_cached_chain(store, chain, cache);
    }

    let mut references = BTreeMap::new();
    for entry in chain {
        for state in &entry.manifest.states {
            for reference in &state.chunks {
                if let Some(existing) = references.insert(reference.object_key, reference) {
                    if existing != reference {
                        return Err(StoreError::Authentication(
                            "one local object key has conflicting authenticated references",
                        ));
                    }
                }
            }
        }
    }
    prefetch_local_availability(store, &references, cache)?;
    validate_cached_chain(store, chain, cache)
}

fn prefetch_local_availability(
    store: &Arc<LocalStore>,
    references: &BTreeMap<Id32, &ChunkRef>,
    cache: &mut BTreeMap<Id32, CachedChunkAvailability>,
) -> Result<(), StoreError> {
    const BATCH: usize = 800;
    let unknown: Vec<Id32> = references
        .keys()
        .filter(|object_key| !cache.contains_key(*object_key))
        .copied()
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }

    let mut available = BTreeMap::new();
    {
        let connection = store.lock_catalog()?;
        let tenant = store.tenant_namespace();
        for batch in unknown.chunks(BATCH) {
            let mut sql = String::from(
                "SELECT c.object_key,c.chunk_id,c.object_digest,c.key_epoch,c.plaintext_bytes,c.object_bytes,c.fidelity_rung,c.location_state FROM chunks c WHERE c.tenant=?1 AND c.object_key IN (",
            );
            for index in 0..batch.len() {
                if index != 0 {
                    sql.push(',');
                }
                sql.push('?');
                sql.push_str(&(index + 2).to_string());
            }
            sql.push_str(
                ") AND ((c.location_state='AVAILABLE' AND EXISTS(SELECT 1 FROM locations l WHERE l.tenant=c.tenant AND l.object_kind='chunk' AND l.object_id=c.object_key AND l.tier='local' AND l.state='AVAILABLE')) OR (c.location_state='TOMBSTONED' AND c.fidelity_rung=2)) AND NOT EXISTS(SELECT 1 FROM tombstones t WHERE t.tenant=c.tenant AND t.object_kind='chunk' AND t.object_id=c.object_key)",
            );
            let mut values: Vec<&dyn rusqlite::ToSql> = vec![&tenant];
            for object_key in batch {
                values.push(object_key);
            }
            let mut statement = connection.prepare(&sql)?;
            let mut rows = statement.query(values.as_slice())?;
            while let Some(row) = rows.next()? {
                let raw_object_key: Vec<u8> = row.get(0)?;
                let object_key: Id32 = raw_object_key.try_into().map_err(|_| {
                    StoreError::Authentication("local chunk catalog object key is invalid")
                })?;
                let availability_tuple: StoredChunkAvailabilityRowTuple = (
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                );
                let availability = StoredChunkAvailabilityRow {
                    chunk_id: availability_tuple.0,
                    object_digest: availability_tuple.1,
                    key_epoch: availability_tuple.2,
                    plaintext_bytes: availability_tuple.3,
                    object_bytes: availability_tuple.4,
                    fidelity_rung: availability_tuple.5,
                    location_state: availability_tuple.6,
                };
                if available.insert(object_key, availability).is_some() {
                    return Err(StoreError::Authentication(
                        "local chunk catalog returned a duplicate object key",
                    ));
                }
            }
        }
    }

    for object_key in unknown {
        let Some(row) = available.remove(&object_key) else {
            cache.insert(object_key, CachedChunkAvailability::Missing);
            continue;
        };
        let reference = references
            .get(&object_key)
            .ok_or(StoreError::State("local availability reference is absent"))?;
        validate_available_reference(
            &row,
            reference,
            store.minimum_readable_key_epoch(),
            store.key_epoch(),
        )?;
        if availability_is_tombstoned(&row) {
            // The chained key and token-cut row authenticate the reference;
            // there are no local bytes left to stat.
            cache.insert(object_key, CachedChunkAvailability::Tombstoned(row));
            continue;
        }
        let metadata = match fs::symlink_metadata(store.chunk_path(&object_key)) {
            Ok(metadata) => metadata,
            Err(_) => {
                cache.insert(object_key, CachedChunkAvailability::Missing);
                continue;
            }
        };
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() != reference.object_bytes as u64
        {
            return Err(StoreError::Authentication(
                "local chunk object metadata disagrees with authenticated manifest",
            ));
        }
        cache.insert(object_key, CachedChunkAvailability::Available(row));
    }
    Ok(())
}

fn availability_is_tombstoned(row: &StoredChunkAvailabilityRow) -> bool {
    row.fidelity_rung == 2 && row.location_state == "TOMBSTONED"
}

fn validate_cached_chain(
    store: &Arc<LocalStore>,
    chain: &[AuthenticatedManifest],
    cache: &BTreeMap<Id32, CachedChunkAvailability>,
) -> Result<ChainChunkDisposition, StoreError> {
    let mut missing = BTreeMap::new();
    let mut tombstoned = BTreeSet::new();
    for entry in chain {
        for state in &entry.manifest.states {
            for reference in &state.chunks {
                match cache
                    .get(&reference.object_key)
                    .ok_or(StoreError::State("local availability cache is incomplete"))?
                {
                    CachedChunkAvailability::Missing => {
                        if let Some(existing) = missing.insert(reference.object_key, reference) {
                            if existing != reference {
                                return Err(StoreError::Authentication(
                                    "one local object key has conflicting authenticated references",
                                ));
                            }
                        }
                    }
                    CachedChunkAvailability::Available(row) => validate_available_reference(
                        row,
                        reference,
                        store.minimum_readable_key_epoch(),
                        store.key_epoch(),
                    )?,
                    CachedChunkAvailability::Tombstoned(row) => {
                        validate_available_reference(
                            row,
                            reference,
                            store.minimum_readable_key_epoch(),
                            store.key_epoch(),
                        )?;
                        tombstoned.insert(reference.object_key);
                    }
                }
            }
        }
    }
    Ok(ChainChunkDisposition {
        complete: missing.is_empty(),
        tombstoned,
    })
}

fn validate_available_reference(
    row: &StoredChunkAvailabilityRow,
    reference: &ChunkRef,
    minimum_key_epoch: u64,
    maximum_key_epoch: u64,
) -> Result<(), StoreError> {
    if row.key_epoch < minimum_key_epoch
        || row.key_epoch > maximum_key_epoch
        || row.key_epoch != reference.key_epoch
        || row.chunk_id.as_slice() != reference.chunk_id
        || row.object_digest.as_slice() != reference.object_digest
        || row.plaintext_bytes != reference.plaintext_bytes
        || row.object_bytes != reference.object_bytes
    {
        return Err(StoreError::Authentication(
            "local chunk catalog disagrees with authenticated manifest",
        ));
    }
    Ok(())
}
pub(crate) fn authenticated_chain_for_compaction(
    store: &Arc<LocalStore>,
    leaf_id: Id32,
    minimum_key_epoch: u64,
    semantic_model: &SemanticModelId,
    family: &RepresentationFamilyId,
    cut: InputCutId,
) -> Result<Vec<CutManifest>, StoreError> {
    let semantic_id = semantic_model_id(semantic_model);
    let family_id = representation_family_id(family)?;
    let context = ValidationContext::default();
    Ok(authenticate_chain(
        store,
        leaf_id,
        minimum_key_epoch,
        Some((semantic_id, family_id, cut)),
        &context,
    )?
    .into_iter()
    .map(|entry| Arc::unwrap_or_clone(entry.manifest))
    .collect())
}

pub(super) fn chain_identity(chain: &[AuthenticatedManifest]) -> Id32 {
    let mut digest = Sha256::new();
    digest.update(b"kvpack/v1/restore-chain\0");
    digest.update((chain.len() as u64).to_le_bytes());
    for entry in chain {
        digest.update(entry.manifest_id);
    }
    digest.finalize().into()
}

pub(super) fn scatter_descriptor_digest(descriptor: &AuthenticatedScatterDescriptor) -> Id32 {
    let mut digest = Sha256::new();
    digest.update(b"kvpack/v1/scatter-descriptor\0");
    digest.update(descriptor.manifest_id);
    digest.update(descriptor.state_key.layer.to_le_bytes());
    digest.update((descriptor.state_key.state_name.len() as u64).to_le_bytes());
    digest.update(descriptor.state_key.state_name.as_bytes());
    digest.update(descriptor.chunk_ordinal.to_le_bytes());
    digest.update(descriptor.batch_number.to_le_bytes());
    digest.update(descriptor.fd_index.to_le_bytes());
    digest.update(descriptor.fd_offset.to_le_bytes());
    digest.update(descriptor.fd_bytes.to_le_bytes());
    digest.update(descriptor.object_key);
    digest.update(descriptor.object_digest);
    digest.update(descriptor.object_bytes.to_le_bytes());
    digest.update(descriptor.plaintext_bytes.to_le_bytes());
    digest.update(descriptor.key_epoch.to_le_bytes());
    digest.update(descriptor.target_offset.to_le_bytes());
    digest.update(descriptor.target_bytes.to_le_bytes());
    digest.update(descriptor.atomic_group.to_le_bytes());
    digest.update(descriptor.attempt);
    digest.finalize().into()
}

pub(super) fn scatter_pin_identity(
    tenant: Id32,
    manifest_id: Id32,
    attempt: Id32,
    chunk_ordinal: u64,
    object_key: Id32,
) -> Result<Id32, StoreError> {
    let mut digest = Sha256::new();
    digest.update(b"kvpack/v1/scatter-source-pin\0");
    digest.update(tenant);
    digest.update(manifest_id);
    digest.update(attempt);
    digest.update(chunk_ordinal.to_le_bytes());
    digest.update(object_key);
    let identity: Id32 = digest.finalize().into();
    if identity == [0; 32] {
        return Err(StoreError::State("derived scatter pin identity is zero"));
    }
    Ok(identity)
}

pub(super) fn chain_resources(
    chain: &[AuthenticatedManifest],
) -> Result<RestoreResourceRequirements, StoreError> {
    let leaf = &chain
        .last()
        .ok_or(StoreError::State("authenticated chain is empty"))?
        .manifest;
    let shadow_bytes = leaf
        .realized_schema
        .states
        .iter()
        .try_fold(0u64, |sum, state| {
            sum.checked_add(state.complete_physical_bytes)
                .ok_or(StoreError::State("restore shadow byte total overflow"))
        })?;
    let mut pinned_source_bytes = 0u64;
    let mut source_pins = 0u64;
    let mut scratch_bytes_per_task = 0u64;
    let mut safety_margin_bytes = 0u64;
    for entry in chain {
        for state in &entry.manifest.states {
            for chunk in &state.chunks {
                pinned_source_bytes = pinned_source_bytes
                    .checked_add(chunk.object_bytes as u64)
                    .ok_or(StoreError::State("restore pinned byte total overflow"))?;
                source_pins = source_pins
                    .checked_add(1)
                    .ok_or(StoreError::State("restore source pin count overflow"))?;
                scratch_bytes_per_task = scratch_bytes_per_task.max(
                    (chunk.object_bytes as u64)
                        .checked_add(chunk.plaintext_bytes as u64)
                        .ok_or(StoreError::State("restore scratch byte bound overflow"))?,
                );
                safety_margin_bytes = safety_margin_bytes.max(chunk.plaintext_bytes as u64);
            }
        }
    }
    Ok(RestoreResourceRequirements {
        shadow_bytes,
        pinned_source_bytes,
        scratch_bytes_per_task,
        staging_bytes: 0,
        receive_window_bytes: 0,
        safety_margin_bytes,
        source_pins,
        source_fds: source_pins,
    })
}

pub(super) fn validate_limits(
    resources: RestoreResourceRequirements,
    limits: RestoreLimits,
    parallelism: usize,
) -> Result<RestoreResourceCharge, StoreError> {
    let charge = resource_charge(resources, parallelism)?;
    if parallelism > limits.maximum_parallelism || !charge_within_limits(charge, limits) {
        return Err(StoreError::Quota(
            "authenticated restore plan exceeds declared resource limits",
        ));
    }
    Ok(charge)
}

pub(super) fn resource_charge(
    resources: RestoreResourceRequirements,
    parallelism: usize,
) -> Result<RestoreResourceCharge, StoreError> {
    let scratch_bytes = resources
        .scratch_bytes_per_task
        .checked_mul(parallelism as u64)
        .ok_or(StoreError::Quota("parallel restore scratch bound overflow"))?;
    if parallelism == 0 {
        return Err(StoreError::Quota("restore parallelism must be nonzero"));
    }
    Ok(RestoreResourceCharge {
        shadow_bytes: resources.shadow_bytes,
        pinned_source_bytes: resources.pinned_source_bytes,
        scratch_bytes,
        staging_bytes: resources.staging_bytes,
        receive_window_bytes: resources.receive_window_bytes,
        safety_margin_bytes: resources.safety_margin_bytes,
        source_pins: resources.source_pins,
        source_fds: resources.source_fds,
    })
}

pub(crate) fn charge_within_limits(charge: RestoreResourceCharge, limits: RestoreLimits) -> bool {
    charge.shadow_bytes <= limits.maximum_shadow_bytes
        && charge.pinned_source_bytes <= limits.maximum_pinned_source_bytes
        && charge.scratch_bytes <= limits.maximum_scratch_bytes
        && charge.staging_bytes <= limits.maximum_staging_bytes
        && charge.receive_window_bytes <= limits.maximum_receive_window_bytes
        && charge.safety_margin_bytes <= limits.maximum_safety_margin_bytes
        && charge.source_pins <= limits.maximum_source_pins
        && charge.source_fds <= limits.maximum_source_fds
}

/// Mark every operation whose chunk sits on the tombstone fidelity rung as
/// a guided-recompute candidate.  The marker carries the chained key (via
/// the authenticated `ChunkRef`) and the token-cut span; restore execution
/// refuses to serve bytes for a marked plan.
pub(super) fn mark_recompute_operations(
    operations: &mut [RestoreChunkOperation],
    disposition: &ChainChunkDisposition,
) {
    for operation in operations.iter_mut() {
        operation.recompute = disposition
            .tombstoned
            .contains(&operation.reference.object_key);
    }
}

pub(super) fn build_scatter_plan(
    chain: &[AuthenticatedManifest],
) -> Result<(Vec<RestoreStatePlan>, Vec<RestoreChunkOperation>), StoreError> {
    let leaf = &chain
        .last()
        .ok_or(StoreError::State("authenticated chain is empty"))?
        .manifest;
    let mut group_by_key = BTreeMap::new();
    for group in &leaf.realized_schema.atomic_groups {
        for state in &group.states {
            if group_by_key.insert(state.clone(), group.id).is_some() {
                return Err(StoreError::Authentication(
                    "authenticated atomic groups overlap",
                ));
            }
        }
    }
    let first = &chain
        .first()
        .ok_or(StoreError::State("authenticated chain is empty"))?
        .manifest;
    if first.states.len() != first.realized_schema.states.len() {
        return Err(StoreError::Authentication(
            "authenticated root state inventory changed",
        ));
    }
    let mut next_token: BTreeMap<StateKey, u64> = first
        .states
        .iter()
        .zip(&first.realized_schema.states)
        .map(|(state, schema)| (state.key.clone(), schema.logical_start))
        .collect();
    let mut chunk_counts: BTreeMap<StateKey, usize> = BTreeMap::new();
    let mut operations = Vec::new();
    for entry in chain {
        for ((state, schema), family) in entry
            .manifest
            .states
            .iter()
            .zip(&entry.manifest.realized_schema.states)
            .zip(&entry.manifest.family.states)
        {
            let expected = next_token
                .get_mut(&state.key)
                .ok_or(StoreError::Authentication(
                    "parent chain state inventory changed",
                ))?;
            let bytes_per_token = family
                .elements_per_token
                .checked_mul(
                    family
                        .dtype
                        .width_bytes()
                        .ok_or(StoreError::Authentication(
                            "authenticated family dtype has no width",
                        ))?,
                )
                .ok_or(StoreError::Authentication(
                    "authenticated state bytes-per-token overflow",
                ))?;
            for (reference, span) in state.chunks.iter().zip(&schema.chunk_spans) {
                if span.token_start != *expected
                    || span.plaintext_offset
                        != span.token_start.checked_mul(bytes_per_token).ok_or(
                            StoreError::Authentication("authenticated scatter offset overflow"),
                        )?
                {
                    return Err(StoreError::Authentication(
                        "parent chain chunk spans are not exactly composable",
                    ));
                }
                *expected =
                    expected
                        .checked_add(span.token_count)
                        .ok_or(StoreError::Authentication(
                            "authenticated scatter token range overflow",
                        ))?;
                *chunk_counts.entry(state.key.clone()).or_default() += 1;
                let target_offset =
                    if family.token_axis_rule == kvpack_core::TokenAxisRule::TailWindow {
                        let base = schema.logical_start.checked_mul(bytes_per_token).ok_or(
                            StoreError::Authentication(
                                "authenticated TailWindow target offset overflows",
                            ),
                        )?;
                        span.plaintext_offset.checked_sub(base).ok_or(
                            StoreError::Authentication(
                                "authenticated TailWindow chunk precedes its declared range",
                            ),
                        )?
                    } else {
                        span.plaintext_offset
                    };
                operations.push(RestoreChunkOperation {
                    state_key: state.key.clone(),
                    span: *span,
                    target_offset,
                    reference: reference.clone(),
                    tenant_namespace: entry.manifest.tenant_namespace,
                    family: entry.manifest.family.clone(),
                    recompute: false,
                });
            }
        }
    }
    let mut states = Vec::with_capacity(leaf.states.len());
    for ((state, schema), family) in leaf
        .states
        .iter()
        .zip(&leaf.realized_schema.states)
        .zip(&leaf.family.states)
    {
        let expected_end = schema
            .logical_start
            .checked_add(schema.logical_count)
            .ok_or(StoreError::Authentication(
                "authenticated state range overflows",
            ))?;
        if next_token.get(&state.key) != Some(&expected_end) {
            return Err(StoreError::Authentication(
                "parent chain does not cover the complete matched cut",
            ));
        }
        let group = *group_by_key
            .get(&state.key)
            .ok_or(StoreError::Authentication(
                "authenticated state has no atomic group",
            ))?;
        let (full_shape, segment_shape, logical_start, logical_count, window) =
            if family.token_axis_rule == kvpack_core::TokenAxisRule::TailWindow {
                (
                    schema.full_shape,
                    schema.segment_shape,
                    schema.logical_start,
                    schema.logical_count,
                    schema.window,
                )
            } else {
                (
                    schema.full_shape,
                    schema.full_shape,
                    0,
                    leaf.input_cut.token_count,
                    0,
                )
            };
        let plaintext_bytes = segment_shape
            .element_count()?
            .checked_mul(
                family
                    .dtype
                    .width_bytes()
                    .ok_or(StoreError::Authentication(
                        "authenticated family dtype has no width",
                    ))?,
            )
            .ok_or(StoreError::Authentication(
                "authenticated restore state byte count overflow",
            ))?;
        states.push(RestoreStatePlan {
            declaration: StateDeclaration {
                key: state.key.clone(),
                full_shape,
                segment_shape,
                strides: schema.strides.clone(),
                logical_start,
                logical_count,
                absolute_position: leaf.input_cut.token_count,
                window,
                atomic_group: group,
            },
            plaintext_bytes,
            physical_span_bytes: schema.complete_physical_bytes,
            atomic_group: group,
            chunk_count: chunk_counts[&state.key],
        });
    }
    if operations.len() > MAX_CHUNKS_PER_STATE.saturating_mul(leaf.states.len()) {
        return Err(StoreError::Authentication(
            "authenticated restore operation count exceeds bound",
        ));
    }
    Ok((states, operations))
}

pub(super) fn random_id(message: &'static str) -> Result<Id32, StoreError> {
    let mut id = [0u8; 32];
    getrandom::fill(&mut id).map_err(|_| StoreError::State(message))?;
    Ok(id)
}
