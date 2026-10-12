use std::collections::BTreeSet;

use kvpack_core::{CutManifest, Id32, ManifestKind, MAX_DELTA_DEPTH};

use super::immutable::write_immutable_batch;
use super::*;

pub(super) fn validate_and_write_manifest_batch(
    store: &LocalStore,
    publications: &[PendingManifest<'_>],
) -> Result<(Id32, Id32), StoreError> {
    if publications.is_empty()
        || !publications.last().is_some_and(|entry| entry.exact_final)
        || publications
            .iter()
            .filter(|entry| entry.exact_final)
            .count()
            != 1
    {
        return Err(StoreError::State(
            "manifest batch requires one final publication in last position",
        ));
    }
    let mut previous_count = 0u64;
    let mut previous_manifest: Option<(&CutManifest, Id32)> = None;
    let mut manifest_ids = BTreeSet::new();
    let semantic = semantic_digest(&publications[0].manifest.semantic_model);
    let family = family_digest(&publications[0].manifest.family)?;
    for publication in publications {
        let manifest = publication.manifest;
        if manifest.tenant_namespace != store.tenant_namespace
            || semantic_digest(&manifest.semantic_model) != semantic
            || family_digest(&manifest.family)? != family
            || publication.prefix_node.token_count != manifest.input_cut.token_count
            || publication.prefix_node.id != manifest.input_cut.token_root
            || manifest.input_cut.token_count <= previous_count
            || kvpack_core::manifest_id(&manifest.encode_canonical()?)
                != publication.encoded.manifest_id
            || !manifest_ids.insert(publication.encoded.manifest_id)
        {
            return Err(StoreError::Expectation(
                "manifest batch identities, order, or exact prefix nodes disagree",
            ));
        }
        if publications.len() > 1 {
            match (previous_manifest, &manifest.realized_schema.kind) {
                (None, ManifestKind::Full) => {}
                (
                    Some((previous, previous_id)),
                    ManifestKind::Delta {
                        parent,
                        parent_cut,
                        depth,
                    },
                ) if *parent == previous_id
                    && *parent_cut == previous.input_cut
                    && *depth == previous.realized_schema.kind.depth() + 1
                    && *depth <= MAX_DELTA_DEPTH => {}
                (Some((previous, _)), ManifestKind::Full)
                    if previous.realized_schema.kind.depth() == MAX_DELTA_DEPTH => {}
                _ => {
                    return Err(StoreError::Expectation(
                        "manifest batch is not one full plus bounded deltas and compactions",
                    ));
                }
            }
        }
        previous_count = manifest.input_cut.token_count;
        previous_manifest = Some((manifest, publication.encoded.manifest_id));
    }
    let objects: Vec<_> = publications
        .iter()
        .map(|publication| {
            (
                store.manifest_path(&publication.encoded.manifest_id),
                publication.encoded.bytes.as_slice(),
            )
        })
        .collect();
    write_immutable_batch(store, DurableObjectKind::Manifest, "partials", &objects)?;
    Ok((semantic, family))
}
