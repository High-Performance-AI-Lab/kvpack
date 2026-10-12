use super::*;

pub(super) fn provisional_intent_digest(
    store: &LocalStore,
    declaration: &ProvisionalExportDeclaration,
    policy: &WritePolicy,
) -> Result<Id32, StoreError> {
    let mut digest = IntentHasher::new(b"kvpack/catalog/provisional-export-intent/v1");
    digest.id(&store.tenant_namespace());
    digest.u64(store.key_epoch());
    digest.id(&declaration.source_declaration_digest);
    digest.id(&declaration.sealed_prompt_token_ids_sha256);
    digest.u64(u64::from(declaration.cached_token_count) + 1);
    digest.id(&semantic_model_id(&declaration.semantic_model));
    digest.id(&representation_family_id(&declaration.family)?);
    digest.u64(declaration.states.len() as u64);
    for state in &declaration.states {
        digest.u32(state.key.layer);
        digest.bytes(state.key.state_name.as_bytes());
        digest.u64(state.strides.len() as u64);
        for stride in &state.strides {
            digest.u64(*stride);
        }
        digest.u32(state.atomic_group);
    }
    digest.byte(u8::from(policy.encrypt_chunks));
    digest.byte(u8::from(policy.encrypt_manifest));
    digest.u64(policy.maximum_restored_bytes);
    digest.u64(policy.publication_generation);
    Ok(digest.finish())
}

pub(super) fn provisional_seal_digest(
    source_declaration_digest: &Id32,
    artifact_digest: &Id32,
    input_cut: &kvpack_core::InputCutId,
    final_manifest: &Id32,
    boundary_token_id: u32,
    provenance: &ProvisionalProvenance,
    chunks: &[ProvisionalChunk],
) -> Id32 {
    let mut digest = IntentHasher::new(b"kvpack/catalog/provisional-export-seal/v1");
    digest.id(source_declaration_digest);
    digest.id(artifact_digest);
    digest.id(&input_cut.token_root);
    digest.id(&input_cut.auxiliary_input_root);
    digest.u64(input_cut.token_count);
    digest.id(final_manifest);
    digest.u32(boundary_token_id);
    digest.u64(provenance.source_wall_clock_ns);
    match provenance.clock_offset_ns {
        Some(offset) => {
            digest.byte(1);
            digest.u64(offset);
        }
        None => digest.byte(0),
    }
    digest.byte(u8::from(provenance.quiesced));
    digest.u64(chunks.len() as u64);
    for chunk in chunks {
        digest.id(&chunk.stored.reference.chunk_id);
        digest.id(&chunk.stored.reference.object_key);
        digest.id(&chunk.stored.reference.object_digest);
        digest.u64(chunk.stored.reference.key_epoch);
    }
    digest.finish()
}

pub(super) fn read_exact_state(
    source: &mut impl Read,
    output: &mut [u8],
) -> Result<(), StoreError> {
    let mut offset = 0usize;
    while offset < output.len() {
        match source.read(&mut output[offset..]) {
            Ok(0) => {
                return Err(StoreError::State(
                    "provisional state source ended before its declared bound",
                ));
            }
            Ok(count) => offset += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(StoreError::Io {
                    op: "read provisional state source",
                    source,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn ensure_source_ended(source: &mut impl Read) -> Result<(), StoreError> {
    let mut extra = [0u8; 1];
    loop {
        match source.read(&mut extra) {
            Ok(0) => return Ok(()),
            Ok(_) => {
                return Err(StoreError::State(
                    "provisional state source exceeded its declared bound",
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(StoreError::Io {
                    op: "check provisional state source bound",
                    source,
                });
            }
        }
    }
}

pub(super) fn read_exact_object(
    path: &std::path::Path,
    expected: usize,
) -> Result<Vec<u8>, StoreError> {
    let metadata = std::fs::symlink_metadata(path).map_err(crate::error::io_error(
        "inspect provisional object for promotion",
    ))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != expected as u64
    {
        return Err(StoreError::Authentication(
            "provisional object metadata changed before promotion",
        ));
    }
    let mut bytes = vec![0u8; expected];
    let mut file = std::fs::File::open(path).map_err(crate::error::io_error(
        "open provisional object for promotion",
    ))?;
    read_exact_state(&mut file, &mut bytes)?;
    ensure_source_ended(&mut file)?;
    Ok(bytes)
}

pub(super) fn elapsed_ns(origin: Instant) -> u64 {
    duration_ns(origin.elapsed())
}

pub(super) fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
