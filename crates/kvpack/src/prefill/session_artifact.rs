//! Durable Muse Glimmer session snapshots built on the production manifest,
//! authenticated chunk, and atomic catalog-publication path.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

use kvpack_core::{
    CacheKind, Codec, DType, Id32, InputCutId, Layout, ManifestDeclaration, ManifestKind, Shape,
    StateDeclaration, StateKey, StaticDimension, TokenAxisRule, ValidationContext,
};

use super::{
    class_state_names, effective_window_tokens, muse_session_resume_preconditions,
    portable_prefill_layout_v2, portable_prefill_token_ids_sha256,
    verify_nope_planes_require_no_rotation, ArtifactTailCoverage, PortablePrefillDescriptorV1,
    PortablePrefillLayoutV2,
};
use crate::writer::{ArtifactWriter, StateWriter};
use crate::{
    AuthenticatedRestorePlan, LocalStore, PublishedArtifact, RestoreLimits, StoreError, WritePolicy,
};

const MUSE_LAYOUT_NAME: &str = "muse-glimmer-30b";
pub const MUSE_EXACT_LOGITS_LAYER: u32 = 52;
pub const MUSE_EXACT_LOGITS_STATE: &str = "decoder.last_logits";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseSessionArtifactReceipt {
    pub manifest_id: Id32,
    pub input_cut: InputCutId,
    pub prompt_token_ids_sha256: Id32,
    pub tail_coverage: ArtifactTailCoverage,
    pub restored_bytes: u64,
    pub publication_generation: u64,
}

/// Streaming writer for one authenticated Muse session snapshot.
///
/// Planes are accepted in the descriptor's canonical layer/K-before-V order.
/// Dropping the writer or a plane before `commit` poisons and aborts the
/// upload; only the final catalog transaction makes the complete artifact
/// discoverable.
pub struct MuseSessionWriter {
    inner: ArtifactWriter,
    input_cut: InputCutId,
    prompt_token_ids_sha256: Id32,
    tail_coverage: ArtifactTailCoverage,
    restored_bytes: u64,
}

impl MuseSessionWriter {
    pub fn begin(
        store: Arc<LocalStore>,
        descriptor: PortablePrefillDescriptorV1,
        prompt_token_ids: Vec<u32>,
        policy: WritePolicy,
    ) -> Result<Self, StoreError> {
        let cached = checked_cached_token_count(&prompt_token_ids)?;
        let layout = muse_layout()?;
        let states = muse_state_declarations(&descriptor, layout, cached)?;
        verify_nope_planes_require_no_rotation(&descriptor, layout)?;
        let (restored_bytes, bytes_per_state) = descriptor_byte_bounds(&descriptor, &states)?;
        if descriptor.restored_bytes != restored_bytes
            || descriptor.bytes_per_state != bytes_per_state
        {
            return Err(StoreError::Expectation(
                "Muse descriptor byte bounds do not match the session ranges",
            ));
        }
        let input_cut = store
            .derive_input_cut(
                &descriptor.semantic_model,
                &descriptor.family,
                &prompt_token_ids,
                &[],
            )?
            .0;
        let prompt_token_ids_sha256 = portable_prefill_token_ids_sha256(&prompt_token_ids);
        let tail_coverage = coverage_from_declarations(&descriptor, &states)?;
        let declaration = ManifestDeclaration {
            semantic_model: descriptor.semantic_model,
            input_tokens: prompt_token_ids,
            auxiliary_inputs: Vec::new(),
            family: descriptor.family,
            kind: ManifestKind::Full,
            states,
        };
        let inner = ArtifactWriter::begin(store, declaration, policy)?;
        Ok(Self {
            inner,
            input_cut,
            prompt_token_ids_sha256,
            tail_coverage,
            restored_bytes,
        })
    }

    pub fn is_published_replay(&self) -> bool {
        self.inner.published_replay().is_some()
    }

    pub fn next_plane(
        &mut self,
        expected_state_key: StateKey,
    ) -> Result<MuseSessionPlaneWriter<'_>, StoreError> {
        Ok(MuseSessionPlaneWriter {
            inner: self.inner.next_state(expected_state_key)?,
        })
    }

    pub fn commit(self) -> Result<MuseSessionArtifactReceipt, StoreError> {
        let published = self.inner.commit()?;
        if published.restored_bytes != self.restored_bytes {
            return Err(StoreError::Authentication(
                "published Muse session byte total changed",
            ));
        }
        Ok(receipt(
            published,
            self.input_cut,
            self.prompt_token_ids_sha256,
            self.tail_coverage,
        ))
    }
}

pub struct MuseSessionPlaneWriter<'a> {
    inner: StateWriter<'a>,
}

impl MuseSessionPlaneWriter<'_> {
    pub fn write_all(&mut self, bytes: &[u8]) -> Result<(), StoreError> {
        self.inner.write_all(bytes)
    }

    pub fn write_source(mut self, source: &mut impl Read) -> Result<(), StoreError> {
        let mut buffer = [0u8; 64 * 1024];
        loop {
            match source.read(&mut buffer) {
                Ok(0) => return self.inner.finish(),
                Ok(read) => self.inner.write_all(&buffer[..read])?,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(source) => {
                    return Err(StoreError::Io {
                        op: "read Muse session plane",
                        source,
                    });
                }
            }
        }
    }

    pub fn finish(self) -> Result<(), StoreError> {
        self.inner.finish()
    }
}

/// An authenticated Muse session manifest and its verified chunk plan.
pub struct MuseSessionArtifact {
    plan: AuthenticatedRestorePlan,
    stored_descriptor: PortablePrefillDescriptorV1,
    prompt_token_ids_sha256: Id32,
    tail_coverage: ArtifactTailCoverage,
}

impl MuseSessionArtifact {
    pub fn open(
        store: Arc<LocalStore>,
        manifest_id: Id32,
        runtime_descriptor: &PortablePrefillDescriptorV1,
        prompt_token_ids: &[u32],
        minimum_key_epoch: u64,
        limits: RestoreLimits,
    ) -> Result<Self, StoreError> {
        let cached = checked_cached_token_count(prompt_token_ids)?;
        let plan = AuthenticatedRestorePlan::build_exact_manifest(
            Arc::clone(&store),
            manifest_id,
            minimum_key_epoch,
            limits,
            &ValidationContext::default(),
        )?;
        let expected_cut = store
            .derive_input_cut(
                &runtime_descriptor.semantic_model,
                &runtime_descriptor.family,
                prompt_token_ids,
                &[],
            )?
            .0;
        if plan.matched_cut() != expected_cut {
            return Err(StoreError::Authentication(
                "Muse session prompt-token identity mismatch",
            ));
        }
        let stored_descriptor = descriptor_from_plan(&plan)?;
        let layout = muse_layout()?;
        let expected_states = muse_state_declarations(&stored_descriptor, layout, cached)?;
        if plan.states().len() != expected_states.len()
            || !plan
                .states()
                .iter()
                .zip(&expected_states)
                .all(|(actual, expected)| actual.declaration == *expected)
        {
            return Err(StoreError::Authentication(
                "Muse session state ranges do not match the registered layout",
            ));
        }
        let tail_coverage = coverage_from_declarations(&stored_descriptor, &expected_states)?;
        muse_session_resume_preconditions(
            &stored_descriptor,
            runtime_descriptor,
            layout,
            cached,
            &tail_coverage,
        )?;
        verify_nope_planes_require_no_rotation(&stored_descriptor, layout)?;
        if stored_descriptor != *runtime_descriptor {
            return Err(StoreError::Authentication(
                "Muse session portable descriptor mismatch",
            ));
        }
        Ok(Self {
            plan,
            stored_descriptor,
            prompt_token_ids_sha256: portable_prefill_token_ids_sha256(prompt_token_ids),
            tail_coverage,
        })
    }

    pub fn manifest_id(&self) -> Id32 {
        self.plan.manifest_id()
    }

    pub fn input_cut(&self) -> InputCutId {
        self.plan.matched_cut()
    }

    pub fn prompt_token_ids_sha256(&self) -> Id32 {
        self.prompt_token_ids_sha256
    }

    pub fn stored_descriptor(&self) -> &PortablePrefillDescriptorV1 {
        &self.stored_descriptor
    }

    pub fn tail_coverage(&self) -> &ArtifactTailCoverage {
        &self.tail_coverage
    }

    pub fn restore_plan(&self) -> &AuthenticatedRestorePlan {
        &self.plan
    }

    pub fn into_restore_plan(self) -> AuthenticatedRestorePlan {
        self.plan
    }
}

fn muse_layout() -> Result<&'static PortablePrefillLayoutV2, StoreError> {
    portable_prefill_layout_v2(MUSE_LAYOUT_NAME)
}

fn checked_cached_token_count(prompt_token_ids: &[u32]) -> Result<u32, StoreError> {
    if prompt_token_ids.is_empty() {
        return Err(StoreError::State("zero is not a durable Muse session cut"));
    }
    u32::try_from(prompt_token_ids.len())
        .map_err(|_| StoreError::State("Muse session token count exceeds u32"))
}

fn class_for_layer(
    layout: &'static PortablePrefillLayoutV2,
    layer: u32,
) -> Result<&'static super::PortablePrefillLayoutClassV2, StoreError> {
    layout
        .classes
        .iter()
        .find(|class| class.layers().contains(&layer))
        .ok_or(StoreError::Expectation(
            "Muse descriptor state names a layer outside the registered layout",
        ))
}

fn muse_state_declarations(
    descriptor: &PortablePrefillDescriptorV1,
    layout: &'static PortablePrefillLayoutV2,
    cached: u32,
) -> Result<Vec<StateDeclaration>, StoreError> {
    let expected_state_count = layout.classes.iter().try_fold(0usize, |total, class| {
        let class_states = class
            .layers()
            .len()
            .checked_mul(class_state_names(class.class).len())
            .ok_or(StoreError::State("Muse state count overflow"))?;
        total
            .checked_add(class_states)
            .ok_or(StoreError::State("Muse state count overflow"))
    })?;
    let state_count = descriptor.states.len();
    if descriptor.family.states.len() != state_count
        || (state_count != expected_state_count && state_count != expected_state_count + 1)
    {
        return Err(StoreError::Expectation(
            "Muse descriptor state inventories disagree",
        ));
    }
    descriptor
        .family
        .states
        .iter()
        .zip(&descriptor.states)
        .map(|(family, declared)| {
            if family.key != declared.key {
                return Err(StoreError::Expectation(
                    "Muse descriptor state order is not canonical",
                ));
            }
            if family.key.layer == MUSE_EXACT_LOGITS_LAYER
                && family.key.state_name == MUSE_EXACT_LOGITS_STATE
            {
                let vocab = match family.dimensions.as_slice() {
                    [StaticDimension::Token, StaticDimension::Fixed(vocab)] if *vocab > 0 => *vocab,
                    _ => {
                        return Err(StoreError::Expectation(
                            "Muse exact logits state has invalid dimensions",
                        ))
                    }
                };
                if family.cache_kind != CacheKind::OrdinaryKv
                    || family.dtype != DType::F32
                    || family.codec != Codec::Raw
                    || family.codec_version != 1
                    || family.layout != Layout::Contiguous
                    || family.token_axis_rule != TokenAxisRule::TailWindow
                    || family.token_axis != 0
                    || family.elements_per_token != vocab
                    || !family.dependencies.is_empty()
                    || declared.strides != [vocab, 1]
                    || declared.atomic_group != MUSE_EXACT_LOGITS_LAYER + 1
                {
                    return Err(StoreError::Expectation(
                        "Muse exact logits state geometry is invalid",
                    ));
                }
                let shape = Shape::new(&[1, vocab])?;
                return Ok(StateDeclaration {
                    key: family.key.clone(),
                    full_shape: shape,
                    segment_shape: shape,
                    strides: declared.strides.clone(),
                    logical_start: u64::from(cached) - 1,
                    logical_count: 1,
                    absolute_position: u64::from(cached),
                    window: 0,
                    atomic_group: declared.atomic_group,
                });
            }
            let class = class_for_layer(layout, family.key.layer)?;
            let expected_names = class_state_names(class.class);
            let count = u64::from(effective_window_tokens(class.window_tokens, cached));
            let expected_rule = if class.window_tokens == 0 {
                TokenAxisRule::Direct
            } else {
                TokenAxisRule::TailWindow
            };
            let elements_per_token = u64::from(class.kv_heads)
                .checked_mul(u64::from(class.head_dim))
                .ok_or(StoreError::State("Muse elements-per-token overflow"))?;
            if count == 0
                || !expected_names.contains(&family.key.state_name.as_str())
                || family.cache_kind != CacheKind::OrdinaryKv
                || family.dtype != DType::F16
                || family.codec != Codec::Raw
                || family.codec_version != 1
                || family.layout != Layout::Contiguous
                || family.token_axis_rule != expected_rule
                || family.token_axis != 0
                || family.elements_per_token != elements_per_token
                || family.dimensions
                    != [
                        StaticDimension::Token,
                        StaticDimension::Fixed(u64::from(class.kv_heads)),
                        StaticDimension::Fixed(u64::from(class.head_dim)),
                    ]
                || !family.dependencies.is_empty()
                || declared.strides != [elements_per_token, u64::from(class.head_dim), 1]
                || declared.atomic_group != family.key.layer + 1
            {
                return Err(StoreError::Expectation(
                    "Muse descriptor state geometry disagrees with its registered layout class",
                ));
            }
            let dimensions = family
                .dimensions
                .iter()
                .map(|dimension| match dimension {
                    StaticDimension::Token => count,
                    StaticDimension::Fixed(value) => *value,
                })
                .collect::<Vec<_>>();
            let shape = Shape::new(&dimensions)?;
            Ok(StateDeclaration {
                key: family.key.clone(),
                full_shape: shape,
                segment_shape: shape,
                strides: declared.strides.clone(),
                logical_start: u64::from(cached) - count,
                logical_count: count,
                absolute_position: u64::from(cached),
                window: 0,
                atomic_group: declared.atomic_group,
            })
        })
        .collect()
}

fn descriptor_byte_bounds(
    descriptor: &PortablePrefillDescriptorV1,
    states: &[StateDeclaration],
) -> Result<(u64, u64), StoreError> {
    let mut total = 0u64;
    let mut maximum = 0u64;
    for (family, state) in descriptor.family.states.iter().zip(states) {
        let bytes = state
            .segment_shape
            .element_count()?
            .checked_mul(
                family
                    .dtype
                    .width_bytes()
                    .ok_or(StoreError::State("Muse state dtype has no fixed width"))?,
            )
            .ok_or(StoreError::State("Muse state byte bound overflow"))?;
        total = total
            .checked_add(bytes)
            .ok_or(StoreError::State("Muse artifact byte bound overflow"))?;
        maximum = maximum.max(bytes);
    }
    Ok((total, maximum))
}

fn coverage_from_declarations(
    descriptor: &PortablePrefillDescriptorV1,
    states: &[StateDeclaration],
) -> Result<ArtifactTailCoverage, StoreError> {
    let mut coverage = BTreeMap::new();
    for (family, state) in descriptor.family.states.iter().zip(states) {
        if family.token_axis_rule == TokenAxisRule::TailWindow {
            coverage.insert(
                state.key.clone(),
                u32::try_from(state.logical_count)
                    .map_err(|_| StoreError::State("Muse tail coverage exceeds u32"))?,
            );
        }
    }
    Ok(coverage)
}

fn descriptor_from_plan(
    plan: &AuthenticatedRestorePlan,
) -> Result<PortablePrefillDescriptorV1, StoreError> {
    let states = plan
        .states()
        .iter()
        .map(|state| super::ExportStateDeclaration {
            key: state.declaration.key.clone(),
            strides: state.declaration.strides.clone(),
            atomic_group: state.atomic_group,
        })
        .collect::<Vec<_>>();
    let restored_bytes = plan.states().iter().try_fold(0u64, |sum, state| {
        sum.checked_add(state.plaintext_bytes)
            .ok_or(StoreError::State("Muse restored byte total overflow"))
    })?;
    let bytes_per_state = plan
        .states()
        .iter()
        .map(|state| state.plaintext_bytes)
        .max()
        .ok_or(StoreError::Authentication(
            "Muse session has no authenticated state planes",
        ))?;
    Ok(PortablePrefillDescriptorV1 {
        semantic_model: *plan.semantic_model(),
        family: plan.family().clone(),
        states,
        bytes_per_state,
        restored_bytes,
    })
}

fn receipt(
    published: PublishedArtifact,
    input_cut: InputCutId,
    prompt_token_ids_sha256: Id32,
    tail_coverage: ArtifactTailCoverage,
) -> MuseSessionArtifactReceipt {
    MuseSessionArtifactReceipt {
        manifest_id: published.manifest_id,
        input_cut,
        prompt_token_ids_sha256,
        tail_coverage,
        restored_bytes: published.restored_bytes,
        publication_generation: published.publication_generation,
    }
}

#[cfg(test)]
mod tests;
