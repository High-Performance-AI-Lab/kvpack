use super::*;
use std::sync::atomic::AtomicUsize;

impl AuthenticatedRestorePlan {
    pub fn build(
        store: Arc<LocalStore>,
        candidate: &RestoreCandidate,
        limits: RestoreLimits,
    ) -> Result<Self, StoreError> {
        if candidate.tenant_namespace != store.tenant_namespace()
            || candidate.tier != RestoreTier::Local
        {
            return Err(StoreError::Expectation(
                "restore candidate is not a local artifact for this tenant",
            ));
        }
        let manifest_id = candidate.manifest_id.ok_or(StoreError::Expectation(
            "recomputation candidate cannot form an authenticated restore plan",
        ))?;
        let context = ValidationContext::default();
        let chain = authenticate_chain(
            &store,
            manifest_id,
            candidate.minimum_key_epoch,
            Some((
                candidate.semantic_id,
                candidate.family_id,
                candidate.matched_cut,
            )),
            &context,
        )?;
        let disposition = chain_chunk_disposition(&store, &chain)?;
        if !disposition.complete {
            return Err(StoreError::NotFound);
        }
        let resources = chain_resources(&chain)?;
        let chain_identity = chain_identity(&chain);
        if candidate.chain_identity != Some(chain_identity)
            || candidate.resources != resources
            || candidate.tombstoned_chunks != disposition.tombstoned.len() as u64
            || candidate.restored_bytes
                != chain
                    .last()
                    .unwrap()
                    .manifest
                    .realized_schema
                    .complete_restored_bytes
        {
            return Err(StoreError::Authentication(
                "restore candidate changed before plan construction",
            ));
        }
        validate_limits(resources, limits, 1)?;
        let leaf = chain
            .last()
            .ok_or(StoreError::State("authenticated chain is empty"))?;
        let semantic_model = leaf.manifest.semantic_model;
        let family = leaf.manifest.family.clone();
        let realized_schema = leaf.manifest.realized_schema.clone();
        let key_epoch = leaf.manifest.key_epoch;
        let (states, mut operations) = build_scatter_plan(&chain)?;
        mark_recompute_operations(&mut operations, &disposition);
        Ok(Self {
            store,
            manifest_id,
            semantic_model,
            family,
            realized_schema,
            key_epoch,
            matched_cut: candidate.matched_cut,
            requested_cut: candidate.requested_cut,
            chain_identity,
            states,
            operations,
            resources,
            limits,
        })
    }

    pub fn build_exact_manifest(
        store: Arc<LocalStore>,
        manifest_id: Id32,
        minimum_key_epoch: u64,
        limits: RestoreLimits,
        context: &ValidationContext,
    ) -> Result<Self, StoreError> {
        let effective_minimum = minimum_key_epoch.max(store.minimum_readable_key_epoch());
        let chain = authenticate_chain(&store, manifest_id, effective_minimum, None, context)?;
        let disposition = chain_chunk_disposition(&store, &chain)?;
        if !disposition.complete {
            return Err(StoreError::NotFound);
        }
        let leaf = chain
            .last()
            .ok_or(StoreError::State("authenticated chain is empty"))?;
        let resources = chain_resources(&chain)?;
        validate_limits(resources, limits, 1)?;
        let chain_identity = chain_identity(&chain);
        let matched_cut = leaf.manifest.input_cut;
        let semantic_model = leaf.manifest.semantic_model;
        let family = leaf.manifest.family.clone();
        let realized_schema = leaf.manifest.realized_schema.clone();
        let key_epoch = leaf.manifest.key_epoch;
        let (states, mut operations) = build_scatter_plan(&chain)?;
        mark_recompute_operations(&mut operations, &disposition);
        Ok(Self {
            store,
            manifest_id,
            semantic_model,
            family,
            realized_schema,
            key_epoch,
            matched_cut,
            requested_cut: matched_cut,
            chain_identity,
            states,
            operations,
            resources,
            limits,
        })
    }

    pub fn manifest_id(&self) -> Id32 {
        self.manifest_id
    }

    pub fn semantic_model(&self) -> &SemanticModelId {
        &self.semantic_model
    }

    pub fn family(&self) -> &RepresentationFamilyId {
        &self.family
    }

    pub fn realized_schema(&self) -> &RealizedCutSchemaId {
        &self.realized_schema
    }

    pub fn key_epoch(&self) -> u64 {
        self.key_epoch
    }

    pub fn matched_cut(&self) -> InputCutId {
        self.matched_cut
    }

    pub fn requested_cut(&self) -> InputCutId {
        self.requested_cut
    }

    pub fn chain_identity(&self) -> Id32 {
        self.chain_identity
    }

    pub fn resources(&self) -> RestoreResourceRequirements {
        self.resources
    }

    pub fn states(&self) -> &[RestoreStatePlan] {
        &self.states
    }

    /// True when any referenced chunk sits on the tombstone fidelity rung.
    /// Such plans are guided-recompute candidates: they carry the chained
    /// keys and token-cut spans for the missing bytes, and byte-serving
    /// execution (`restore`, `prepare_scatter_transfer`) fails closed.
    pub fn requires_guided_recompute(&self) -> bool {
        self.operations.iter().any(|operation| operation.recompute)
    }

    /// The tombstoned chunk spans that must be regenerated by guided
    /// recompute, in canonical chain order.  Empty for fully resident plans.
    pub fn recompute_spans(&self) -> Vec<(StateKey, kvpack_core::ChunkSpan)> {
        self.operations
            .iter()
            .filter(|operation| operation.recompute)
            .map(|operation| (operation.state_key.clone(), operation.span))
            .collect()
    }

    /// Read back the authenticated M7 statistics sidecars for every chunk
    /// operation in chain order (`None` for chunks published without one).
    /// Pins, reads, authenticates, and releases each object exactly like the
    /// restore decode path; fails closed on any authentication error and on
    /// plans that require guided recompute (their bytes are gone).
    pub fn chunk_stats_sidecars(&self) -> Result<Vec<Option<StatsSidecar>>, StoreError> {
        if self.requires_guided_recompute() {
            return Err(StoreError::Expectation(
                "guided-recompute plan has no bytes to read statistics from",
            ));
        }
        let pin_ids = self.store.pin_chunks_batch(&self.operation_references())?;
        let mut sidecars = Vec::with_capacity(self.operations.len());
        for (ordinal, operation) in self.operations.iter().enumerate() {
            match self.decode_operation_sidecar(operation, pin_ids[ordinal]) {
                Ok(sidecar) => sidecars.push(sidecar),
                Err(error) => {
                    crate::store::release_restore_pin_batch(
                        &self.store,
                        &mut Vec::new(),
                        &pin_ids[ordinal + 1..],
                    );
                    return Err(error);
                }
            }
        }
        Ok(sidecars)
    }

    fn decode_operation_sidecar(
        &self,
        operation: &RestoreChunkOperation,
        pin_id: Id32,
    ) -> Result<Option<StatsSidecar>, StoreError> {
        let (object, pin) = self
            .store
            .read_pinned_chunk_object(&operation.reference, pin_id)?;
        let keys = self.store.schedule(operation.reference.key_epoch)?;
        let decoded = decode_chunk_with_stats(
            &object,
            &operation.reference,
            &operation.span,
            &operation.tenant_namespace,
            &operation.family,
            &operation.state_key,
            &keys,
        );
        drop(pin);
        Ok(decoded?.1)
    }

    pub fn scatter_pin_ids(&self, attempt: Id32) -> Result<Vec<Id32>, StoreError> {
        if attempt == [0; 32] {
            return Err(StoreError::Expectation(
                "scatter transfer attempt must be nonzero",
            ));
        }
        self.operations
            .iter()
            .enumerate()
            .map(|(ordinal, operation)| {
                let ordinal = u64::try_from(ordinal)
                    .map_err(|_| StoreError::State("scatter chunk ordinal exceeds u64"))?;
                scatter_pin_identity(
                    self.store.tenant_namespace(),
                    self.manifest_id,
                    attempt,
                    ordinal,
                    operation.reference.object_key,
                )
            })
            .collect()
    }

    pub fn prepare_scatter_transfer(
        &self,
        attempt: Id32,
    ) -> Result<PreparedScatterTransfer, StoreError> {
        if attempt == [0; 32] {
            return Err(StoreError::Expectation(
                "scatter transfer attempt must be nonzero",
            ));
        }
        if self.requires_guided_recompute() {
            return Err(StoreError::Expectation(
                "guided-recompute plan must not serve tombstoned chunk bytes",
            ));
        }
        let groups: BTreeMap<_, _> = self
            .states
            .iter()
            .map(|state| (state.declaration.key.clone(), state.atomic_group))
            .collect();
        let total_batches_usize = self.operations.len().div_ceil(MAX_SCATTER_FDS_PER_BATCH);
        let total_batches = u32::try_from(total_batches_usize)
            .map_err(|_| StoreError::State("scatter batch count exceeds u32"))?;
        let mut batches = Vec::with_capacity(total_batches_usize);
        for (batch_index, operations) in self
            .operations
            .chunks(MAX_SCATTER_FDS_PER_BATCH)
            .enumerate()
        {
            let batch_number = u32::try_from(batch_index)
                .map_err(|_| StoreError::State("scatter batch number exceeds u32"))?;
            let mut ordinals = Vec::with_capacity(operations.len());
            let mut pin_entries = Vec::with_capacity(operations.len());
            for (fd_index, operation) in operations.iter().enumerate() {
                let chunk_ordinal = batch_index
                    .checked_mul(MAX_SCATTER_FDS_PER_BATCH)
                    .and_then(|base| base.checked_add(fd_index))
                    .and_then(|ordinal| u64::try_from(ordinal).ok())
                    .ok_or(StoreError::State("scatter chunk ordinal overflow"))?;
                let pin_id = scatter_pin_identity(
                    self.store.tenant_namespace(),
                    self.manifest_id,
                    attempt,
                    chunk_ordinal,
                    operation.reference.object_key,
                )?;
                ordinals.push(chunk_ordinal);
                pin_entries.push((operation.reference.clone(), pin_id));
            }
            // One catalog transaction for the complete scatter batch instead
            // of one autocommit INSERT per chunk FD.
            let pinned = self.store.pin_chunks_with_ids(&pin_entries)?;
            let mut pinned = pinned.into_iter();
            let mut files = Vec::with_capacity(operations.len());
            let mut pins = Vec::with_capacity(operations.len());
            let mut descriptors = Vec::with_capacity(operations.len());
            for (fd_index, operation) in operations.iter().enumerate() {
                let chunk_ordinal = ordinals[fd_index];
                let (file, pin) = pinned
                    .next()
                    .ok_or(StoreError::State("scatter pin batch is incomplete"))?
                    .into_retained_file()?;
                let fd_index = u32::try_from(fd_index)
                    .map_err(|_| StoreError::State("scatter FD index exceeds u32"))?;
                let atomic_group =
                    *groups
                        .get(&operation.state_key)
                        .ok_or(StoreError::Authentication(
                            "scatter state has no atomic group",
                        ))?;
                let mut descriptor = AuthenticatedScatterDescriptor {
                    manifest_id: self.manifest_id,
                    state_key: operation.state_key.clone(),
                    chunk_ordinal,
                    batch_number,
                    fd_index,
                    fd_offset: 0,
                    fd_bytes: operation.reference.object_bytes as u64,
                    object_key: operation.reference.object_key,
                    object_digest: operation.reference.object_digest,
                    object_bytes: operation.reference.object_bytes as u64,
                    plaintext_bytes: operation.reference.plaintext_bytes as u64,
                    key_epoch: operation.reference.key_epoch,
                    target_offset: operation.target_offset,
                    target_bytes: operation.span.plaintext_bytes as u64,
                    atomic_group,
                    attempt,
                    descriptor_digest: [0; 32],
                };
                descriptor.descriptor_digest = scatter_descriptor_digest(&descriptor);
                descriptors.push(descriptor);
                files.push(file);
                pins.push(pin);
            }
            batches.push(PinnedScatterBatch {
                batch_number,
                total_batches,
                descriptors,
                files,
                pins,
            });
        }
        if batches.is_empty() {
            return Err(StoreError::Authentication(
                "authenticated scatter plan has no chunk operations",
            ));
        }
        Ok(PreparedScatterTransfer {
            manifest_id: self.manifest_id,
            attempt,
            resources: self.resources,
            batches,
        })
    }

    pub fn restore_sequential(
        &self,
        sink: &mut dyn VerifiedRestoreSink,
        cancellation: &RestoreCancellation,
    ) -> Result<InstalledRestore, StoreError> {
        self.restore(sink, cancellation, 1)
    }

    pub fn restore_parallel(
        &self,
        sink: &mut dyn VerifiedRestoreSink,
        cancellation: &RestoreCancellation,
        parallelism: usize,
    ) -> Result<InstalledRestore, StoreError> {
        if parallelism < 2 {
            return Err(StoreError::Expectation(
                "parallel restore requires at least two tasks",
            ));
        }
        self.restore(sink, cancellation, parallelism)
    }

    fn restore(
        &self,
        sink: &mut dyn VerifiedRestoreSink,
        cancellation: &RestoreCancellation,
        parallelism: usize,
    ) -> Result<InstalledRestore, StoreError> {
        if self.requires_guided_recompute() {
            return Err(StoreError::Expectation(
                "guided-recompute plan must not serve tombstoned chunk bytes",
            ));
        }
        let charge = validate_limits(self.resources, self.limits, parallelism)?;
        if cancellation.is_cancelled() {
            return Err(StoreError::Cancelled);
        }
        let restore_id = random_id("restore identity entropy failed")?;
        self.store
            .reserve_restore_resources(restore_id, charge, self.limits)?;
        let mut reservation = RestoreReservation::new(&self.store, restore_id);
        if let Err(error) = sink.begin_restore(self.manifest_id, &self.states) {
            sink.abort_restore();
            return Err(error);
        }
        let context = TraceContext::new(ServiceComponent::Store).ok();
        let verification_started = Instant::now();
        let verification_started_unix_ns = restore_now_ns();
        let pins = if parallelism == 1 {
            self.restore_operations_sequential(sink, cancellation)
        } else {
            self.restore_operations_parallel(sink, cancellation, parallelism)
        };
        let verification_outcome = match &pins {
            Ok(_) => SpanOutcome::Ok,
            Err(StoreError::Cancelled) => SpanOutcome::Cancelled,
            Err(StoreError::Integrity(_) | StoreError::Authentication(_)) => {
                SpanOutcome::IntegrityError
            }
            Err(_) => SpanOutcome::Unavailable,
        };
        let _ = self
            .store
            .telemetry
            .observe_latency(TracePhase::Verification, verification_started.elapsed());
        let verification_span = context.as_ref().and_then(|context| {
            self.store
                .telemetry
                .record_span(
                    context,
                    None,
                    TracePhase::Verification,
                    verification_outcome,
                    verification_started_unix_ns,
                    restore_now_ns().max(verification_started_unix_ns),
                )
                .ok()
                .flatten()
        });
        let pins = match pins {
            Ok(pins) => pins,
            Err(error) => {
                sink.abort_restore();
                return Err(error);
            }
        };
        let pinned_bytes = match pins.iter().try_fold(0u64, |sum, pin| {
            sum.checked_add(pin.bytes())
                .ok_or(StoreError::State("retained source pin total overflow"))
        }) {
            Ok(bytes) => bytes,
            Err(error) => {
                sink.abort_restore();
                return Err(error);
            }
        };
        if pinned_bytes != self.resources.pinned_source_bytes
            || pins.len() as u64 != self.resources.source_pins
        {
            sink.abort_restore();
            return Err(StoreError::Authentication(
                "retained source pins disagree with authenticated plan resources",
            ));
        }
        if cancellation.is_cancelled() {
            sink.abort_restore();
            return Err(StoreError::Cancelled);
        }
        if let Err(error) = self.store.attach_restore_pins(&restore_id, pins) {
            sink.abort_restore();
            return Err(error);
        }
        let install_started = Instant::now();
        let install_started_unix_ns = restore_now_ns();
        let install = sink.commit_restore();
        let install_outcome = if install.is_ok() {
            SpanOutcome::Ok
        } else {
            SpanOutcome::Unavailable
        };
        let _ = self
            .store
            .telemetry
            .observe_latency(TracePhase::Install, install_started.elapsed());
        let install_span = context.as_ref().and_then(|context| {
            self.store
                .telemetry
                .record_span(
                    context,
                    verification_span,
                    TracePhase::Install,
                    install_outcome,
                    install_started_unix_ns,
                    restore_now_ns().max(install_started_unix_ns),
                )
                .ok()
                .flatten()
        });
        if let Err(error) = install {
            sink.reset_restore();
            return Err(error);
        }
        let _ = self.store.telemetry.add_bytes(
            ByteCounter::Restored,
            self.realized_schema.complete_restored_bytes,
        );
        reservation.retain_until_engine_free();
        Ok(InstalledRestore {
            store: Arc::clone(&self.store),
            restore_id,
            manifest_id: self.manifest_id,
            trace_context: context,
            release_parent_span: install_span,
            engine_freed: false,
        })
    }

    fn restore_operations_sequential(
        &self,
        sink: &mut dyn VerifiedRestoreSink,
        cancellation: &RestoreCancellation,
    ) -> Result<Vec<RetainedPin>, StoreError> {
        // One catalog transaction acquires every source pin up front instead
        // of one autocommit INSERT per chunk; a missing chunk now fails
        // before any sink write rather than mid-plan.
        let pin_ids = self.store.pin_chunks_batch(&self.operation_references())?;
        let mut pins = Vec::with_capacity(self.operations.len());
        for (ordinal, operation) in self.operations.iter().enumerate() {
            if cancellation.is_cancelled() {
                crate::store::release_restore_pin_batch(
                    &self.store,
                    &mut pins,
                    &pin_ids[ordinal..],
                );
                return Err(StoreError::Cancelled);
            }
            let (plaintext, pin) = match self.decode_operation(operation, pin_ids[ordinal]) {
                Ok(decoded) => decoded,
                Err(error) => {
                    crate::store::release_restore_pin_batch(
                        &self.store,
                        &mut pins,
                        &pin_ids[ordinal + 1..],
                    );
                    return Err(error);
                }
            };
            if let Err(error) =
                sink.write_verified_chunk(&operation.state_key, operation.target_offset, &plaintext)
            {
                pins.push(pin);
                crate::store::release_restore_pin_batch(
                    &self.store,
                    &mut pins,
                    &pin_ids[ordinal + 1..],
                );
                return Err(error);
            }
            pins.push(pin);
        }
        Ok(pins)
    }

    fn restore_operations_parallel(
        &self,
        sink: &mut dyn VerifiedRestoreSink,
        cancellation: &RestoreCancellation,
        parallelism: usize,
    ) -> Result<Vec<RetainedPin>, StoreError> {
        // Long-lived workers pull ordinals off an atomic cursor and decode;
        // the sink writes results as they arrive over a bounded channel, so
        // decode and sink I/O overlap instead of join-stepping per batch.
        let pin_ids = self.store.pin_chunks_batch(&self.operation_references())?;
        let total = self.operations.len();
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let (sender, receiver) = std::sync::mpsc::sync_channel::<(
            usize,
            Result<(Vec<u8>, RetainedPin), StoreError>,
        )>(parallelism);
        let mut pins: Vec<(usize, RetainedPin)> = Vec::with_capacity(total);
        let mut pending_error: Option<StoreError> = None;
        std::thread::scope(|scope| {
            for _ in 0..parallelism.min(total).max(1) {
                let sender = sender.clone();
                let next = &next;
                let failed = &failed;
                let pin_ids = &pin_ids;
                scope.spawn(move || {
                    loop {
                        let ordinal = next.fetch_add(1, Ordering::Relaxed);
                        if ordinal >= total {
                            return;
                        }
                        if cancellation.is_cancelled() || failed.load(Ordering::Relaxed) {
                            // Claimed but never decoded: release the pin row.
                            let _ = self
                                .store
                                .release_retained_source_pins(std::slice::from_ref(
                                    &pin_ids[ordinal],
                                ));
                            return;
                        }
                        let result =
                            self.decode_operation(&self.operations[ordinal], pin_ids[ordinal]);
                        let stop = result.is_err();
                        if sender.send((ordinal, result)).is_err() || stop {
                            return;
                        }
                    }
                });
            }
            drop(sender);
            for (ordinal, result) in receiver {
                match result {
                    Ok((plaintext, pin)) => {
                        if pending_error.is_none() && !cancellation.is_cancelled() {
                            if let Err(error) = sink.write_verified_chunk(
                                &self.operations[ordinal].state_key,
                                self.operations[ordinal].target_offset,
                                &plaintext,
                            ) {
                                pending_error = Some(error);
                            }
                        } else if pending_error.is_none() {
                            pending_error = Some(StoreError::Cancelled);
                        }
                        if pending_error.is_some() {
                            failed.store(true, Ordering::Relaxed);
                        }
                        pins.push((ordinal, pin));
                    }
                    Err(error) => {
                        if pending_error.is_none() {
                            pending_error = Some(error);
                        }
                        failed.store(true, Ordering::Relaxed);
                    }
                }
            }
        });
        let claimed = next.load(Ordering::Relaxed).min(total);
        match pending_error {
            Some(error) => {
                let mut pins: Vec<RetainedPin> = pins.into_iter().map(|(_, pin)| pin).collect();
                crate::store::release_restore_pin_batch(
                    &self.store,
                    &mut pins,
                    &pin_ids[claimed..],
                );
                Err(error)
            }
            None => {
                // Completion order is arbitrary; restore canonical ordinal
                // order so the returned pins match the sequential path.
                pins.sort_unstable_by_key(|(ordinal, _)| *ordinal);
                Ok(pins.into_iter().map(|(_, pin)| pin).collect())
            }
        }
    }

    fn operation_references(&self) -> Vec<ChunkRef> {
        self.operations
            .iter()
            .map(|operation| operation.reference.clone())
            .collect()
    }

    fn decode_operation(
        &self,
        operation: &RestoreChunkOperation,
        pin_id: Id32,
    ) -> Result<(Vec<u8>, RetainedPin), StoreError> {
        let (object, pin) = self
            .store
            .read_pinned_chunk_object(&operation.reference, pin_id)?;
        let keys = self.store.schedule(operation.reference.key_epoch)?;
        let plaintext = decode_chunk(
            &object,
            &operation.reference,
            &operation.span,
            &operation.tenant_namespace,
            &operation.family,
            &operation.state_key,
            &keys,
        )?;
        Ok((plaintext, pin))
    }
}
