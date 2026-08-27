//! Handoff V2: arbitrary ordered components and atomic target+DFlash commit.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{canonical_json, sha256_hex, ExactIdentityV1, HandoffError, MacKey, Result};

pub const LIVE_HANDOFF_PROTOCOL_V2: &str = "kvpack-live-handoff-v2";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKindV2 {
    TargetKv,
    DflashContext,
    VisionContext,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComponentV2 {
    pub id: String,
    pub kind: ComponentKindV2,
    pub required: bool,
    pub identity_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HmacIdentityV2 {
    pub key_id: String,
    pub epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultimodalIdentityV2 {
    pub projector_sha256: String,
    pub preprocessing_sha256: String,
    pub image_sequence_sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SegmentRoleV2 {
    NopeKey,
    NopeValue,
    SwaKey,
    SwaValue,
    NopeTile,
    SwaTile,
    DflashKey,
    DflashValue,
    Auxiliary,
}

type SegmentRangeKey = (String, SegmentRoleV2, Option<u32>);
type SegmentRanges = BTreeMap<SegmentRangeKey, Vec<(u64, u64)>>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentDescriptorV2 {
    pub sequence: u32,
    pub component_id: String,
    pub role: SegmentRoleV2,
    pub layer: Option<u32>,
    pub logical_start: u64,
    pub logical_count: u64,
    pub element_type: String,
    pub elements_per_token: u32,
    pub byte_len: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BeginManifestV2 {
    pub protocol: String,
    pub transfer_id: String,
    pub generation: u64,
    pub created_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub identity: ExactIdentityV1,
    pub prompt_token_ids: Vec<u32>,
    pub multimodal: Option<MultimodalIdentityV2>,
    pub hmac: HmacIdentityV2,
    pub components: Vec<ComponentV2>,
    /// Streaming producers cannot know hashes for future KV tiles at begin
    /// time. In deferred mode each ordered segment frame carries its complete
    /// descriptor; the terminal seal still binds the canonical descriptor
    /// stream and all payload bytes before commit.
    #[serde(default, skip_serializing_if = "is_false")]
    pub deferred_segments: bool,
    pub segments: Vec<SegmentDescriptorV2>,
}

#[derive(Debug, Clone)]
pub struct ValidatedBeginV2 {
    manifest: BeginManifestV2,
    canonical: Vec<u8>,
}

impl ValidatedBeginV2 {
    pub fn validate(
        manifest: BeginManifestV2,
        now_unix_ms: u64,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<Self> {
        if manifest.protocol != LIVE_HANDOFF_PROTOCOL_V2 {
            return Err(validation("wrong V2 protocol"));
        }
        if manifest.transfer_id.is_empty()
            || manifest.transfer_id.len() > 128
            || !manifest
                .transfer_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        {
            return Err(validation("invalid transfer id"));
        }
        if manifest.expires_unix_ms <= manifest.created_unix_ms
            || now_unix_ms > manifest.expires_unix_ms
        {
            return Err(validation("expired or invalid handoff lifetime"));
        }
        if manifest.hmac.key_id != expected_key_id || manifest.hmac.epoch < minimum_epoch {
            return Err(validation("wrong HMAC key id or replayed epoch"));
        }
        validate_hex("model identity", &manifest.identity.model_sha256)?;
        validate_hex("adapter identity", &manifest.identity.adapter_sha256)?;
        validate_hex(
            "chat template identity",
            &manifest.identity.chat_template_sha256,
        )?;
        validate_hex(
            "context policy identity",
            &manifest.identity.context_policy_sha256,
        )?;
        validate_hex("tokenizer identity", &manifest.identity.tokenizer_sha256)?;
        if let Some(multimodal) = &manifest.multimodal {
            validate_hex("projector", &multimodal.projector_sha256)?;
            validate_hex("preprocessing", &multimodal.preprocessing_sha256)?;
            validate_hex("image sequence", &multimodal.image_sequence_sha256)?;
        }
        if manifest.components.is_empty()
            || (manifest.deferred_segments && !manifest.segments.is_empty())
            || (!manifest.deferred_segments && manifest.segments.is_empty())
        {
            return Err(validation(
                "V2 requires components and exactly one declared/deferred segment mode",
            ));
        }
        if manifest.prompt_token_ids.is_empty() {
            return Err(validation("V2 prompt token sequence is empty"));
        }
        let mut ids = BTreeSet::new();
        let mut target = false;
        let mut dflash_required = false;
        for component in &manifest.components {
            if component.id.is_empty() || !ids.insert(component.id.clone()) {
                return Err(validation("empty or duplicate component id"));
            }
            validate_hex("component identity", &component.identity_sha256)?;
            target |= component.kind == ComponentKindV2::TargetKv && component.required;
            dflash_required |=
                component.kind == ComponentKindV2::DflashContext && component.required;
        }
        if !target {
            return Err(validation("a required target KV component is mandatory"));
        }
        let mut ranges = BTreeMap::new();
        for (index, segment) in manifest.segments.iter().enumerate() {
            if segment.sequence as usize != index {
                return Err(validation(
                    "segment sequences must be contiguous in declared wire order",
                ));
            }
            validate_segment_descriptor(segment, &ids, &mut ranges)?;
        }
        if !manifest.deferred_segments
            && dflash_required
            && !manifest.segments.iter().any(|s| {
                manifest
                    .components
                    .iter()
                    .any(|c| c.id == s.component_id && c.kind == ComponentKindV2::DflashContext)
            })
        {
            return Err(validation("required DFlash component has no segments"));
        }
        let canonical = canonical_json(&manifest)?;
        Ok(Self {
            manifest,
            canonical,
        })
    }
    pub fn manifest(&self) -> &BeginManifestV2 {
        &self.manifest
    }
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }
}

#[derive(Debug, Clone)]
pub struct VerifiedSegmentV2 {
    descriptor: SegmentDescriptorV2,
    payload: Vec<u8>,
}
impl VerifiedSegmentV2 {
    fn verify(descriptor: &SegmentDescriptorV2, payload: Vec<u8>) -> Result<Self> {
        if payload.len() as u64 != descriptor.byte_len || sha256_hex(&payload) != descriptor.sha256
        {
            return Err(validation("segment length or digest mismatch"));
        }
        Ok(Self {
            descriptor: descriptor.clone(),
            payload,
        })
    }
    pub fn descriptor(&self) -> &SegmentDescriptorV2 {
        &self.descriptor
    }
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Consume the verified plane so a streaming sink can install it without
    /// a second host copy of the payload.
    pub fn into_payload(self) -> (SegmentDescriptorV2, Vec<u8>) {
        (self.descriptor, self.payload)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SealCoreV2 {
    pub transfer_id: String,
    pub generation: u64,
    pub begin_sha256: String,
    pub descriptor_sha256: String,
    pub payload_sha256: String,
    pub segment_count: u32,
    pub total_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SealManifestV2 {
    pub core: SealCoreV2,
    pub hmac_sha256: String,
}
impl SealManifestV2 {
    pub fn sign(
        begin: &ValidatedBeginV2,
        descriptors: &[SegmentDescriptorV2],
        payloads: &[Vec<u8>],
        key: &MacKey,
    ) -> Result<Self> {
        if descriptors.len() != payloads.len() {
            return Err(validation("descriptor/payload count mismatch"));
        }
        let mut descriptor_hash = Sha256::new();
        let mut payload_hash = Sha256::new();
        let mut total = 0u64;
        for (index, (descriptor, payload)) in descriptors.iter().zip(payloads).enumerate() {
            if descriptor.sequence as usize != index
                || descriptor.byte_len != payload.len() as u64
                || descriptor.sha256 != sha256_hex(payload)
            {
                return Err(validation("cannot sign mismatched segment material"));
            }
            descriptor_hash.update(canonical_json(descriptor)?);
            payload_hash.update(payload);
            total = total
                .checked_add(descriptor.byte_len)
                .ok_or_else(|| validation("total byte count overflow"))?;
        }
        let core = SealCoreV2 {
            transfer_id: begin.manifest.transfer_id.clone(),
            generation: begin.manifest.generation,
            begin_sha256: sha256_hex(begin.canonical_bytes()),
            descriptor_sha256: hex::encode(descriptor_hash.finalize()),
            payload_sha256: hex::encode(payload_hash.finalize()),
            segment_count: descriptors
                .len()
                .try_into()
                .map_err(|_| validation("segment count exceeds u32"))?,
            total_bytes: total,
        };
        let hmac_sha256 = key.tag_hex(&canonical_json(&core)?)?;
        Ok(Self { core, hmac_sha256 })
    }
}
#[derive(Debug, Clone)]
pub struct VerifiedSealV2 {
    seal: SealManifestV2,
}
impl VerifiedSealV2 {
    pub fn seal(&self) -> &SealManifestV2 {
        &self.seal
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedGeneration {
    pub transfer_id: String,
    pub generation: u64,
    pub installed_segments: u32,
    pub installed_bytes: u64,
}

/// Engine-owned detached shadow. Implementations must not mutate their live
/// generation before `commit`, and `abort` must discard the complete shadow.
pub trait HandoffSinkV2 {
    fn begin(&mut self, begin: &ValidatedBeginV2) -> Result<()>;
    fn segment_ready(&mut self, segment: VerifiedSegmentV2) -> Result<()>;
    fn prepare_commit(&mut self, seal: &VerifiedSealV2) -> Result<()>;
    fn commit(&mut self) -> Result<CommittedGeneration>;
    fn abort(&mut self);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Receiving,
    Prepared,
    Committed,
    Aborted,
}

pub struct AtomicReceiverV2<S: HandoffSinkV2> {
    begin: ValidatedBeginV2,
    sink: S,
    key: MacKey,
    next: usize,
    deferred_ranges: SegmentRanges,
    seen_components: BTreeSet<String>,
    descriptor_hash: Sha256,
    payload_hash: Sha256,
    total: u64,
    state: State,
}

impl<S: HandoffSinkV2> AtomicReceiverV2<S> {
    pub fn begin(begin: ValidatedBeginV2, key: MacKey, mut sink: S) -> Result<Self> {
        sink.begin(&begin)?;
        Ok(Self {
            begin,
            sink,
            key,
            next: 0,
            deferred_ranges: BTreeMap::new(),
            seen_components: BTreeSet::new(),
            descriptor_hash: Sha256::new(),
            payload_hash: Sha256::new(),
            total: 0,
            state: State::Receiving,
        })
    }
    pub fn segment_ready(&mut self, sequence: u32, payload: Vec<u8>) -> Result<()> {
        if self.state != State::Receiving {
            return self.fail("segment outside receiving state");
        }
        let Some(descriptor) = self.begin.manifest.segments.get(self.next) else {
            return self.fail("extra segment");
        };
        if descriptor.sequence != sequence {
            return self.fail("dropped, duplicate, or reordered segment");
        }
        self.accept_segment(descriptor.clone(), payload)
    }

    /// Accept one just-in-time descriptor from a true streaming producer.
    /// Sequence, component, geometry, overlap, payload hash, and the terminal
    /// seal are checked with the same fail-closed transaction as declared V2.
    pub fn segment_ready_deferred(
        &mut self,
        descriptor: SegmentDescriptorV2,
        payload: Vec<u8>,
    ) -> Result<()> {
        if self.state != State::Receiving || !self.begin.manifest.deferred_segments {
            return self.fail("deferred segment outside deferred receiving mode");
        }
        if descriptor.sequence as usize != self.next {
            return self.fail("dropped, duplicate, or reordered deferred segment");
        }
        let ids = self
            .begin
            .manifest
            .components
            .iter()
            .map(|component| component.id.clone())
            .collect::<BTreeSet<_>>();
        if let Err(error) =
            validate_segment_descriptor(&descriptor, &ids, &mut self.deferred_ranges)
        {
            self.abort();
            return Err(error);
        }
        self.accept_segment(descriptor, payload)
    }

    fn accept_segment(&mut self, descriptor: SegmentDescriptorV2, payload: Vec<u8>) -> Result<()> {
        let verified = match VerifiedSegmentV2::verify(&descriptor, payload) {
            Ok(v) => v,
            Err(e) => {
                self.abort();
                return Err(e);
            }
        };
        self.descriptor_hash.update(canonical_json(&descriptor)?);
        self.payload_hash.update(verified.payload());
        self.total = self
            .total
            .checked_add(descriptor.byte_len)
            .ok_or_else(|| validation("total byte count overflow"))?;
        if let Err(error) = self.sink.segment_ready(verified) {
            self.abort();
            return Err(error);
        }
        self.seen_components.insert(descriptor.component_id.clone());
        self.next += 1;
        Ok(())
    }
    pub fn prepare_commit(&mut self, seal: SealManifestV2) -> Result<()> {
        let all_segments = if self.begin.manifest.deferred_segments {
            self.next > 0
        } else {
            self.next == self.begin.manifest.segments.len()
        };
        if self.state != State::Receiving || !all_segments {
            return self.fail("seal before every segment");
        }
        if self.begin.manifest.deferred_segments
            && self.begin.manifest.components.iter().any(|component| {
                component.required && !self.seen_components.contains(&component.id)
            })
        {
            return self.fail("required component has no deferred segment");
        }
        let core = &seal.core;
        let expected = &self.begin.manifest;
        if core.transfer_id != expected.transfer_id
            || core.generation != expected.generation
            || core.segment_count as usize != self.next
            || core.total_bytes != self.total
            || core.begin_sha256 != sha256_hex(self.begin.canonical_bytes())
            || core.descriptor_sha256 != hex::encode(self.descriptor_hash.clone().finalize())
            || core.payload_sha256 != hex::encode(self.payload_hash.clone().finalize())
        {
            return self.fail("seal identity or digest mismatch");
        }
        validate_hex("seal HMAC", &seal.hmac_sha256)?;
        let stream = canonical_json(core)?;
        if let Err(error) = self.key.verify_hex(&stream, &seal.hmac_sha256) {
            self.abort();
            return Err(error);
        }
        let verified = VerifiedSealV2 { seal };
        if let Err(error) = self.sink.prepare_commit(&verified) {
            self.abort();
            return Err(error);
        }
        self.state = State::Prepared;
        Ok(())
    }
    pub fn commit(&mut self) -> Result<CommittedGeneration> {
        if self.state != State::Prepared {
            return self.fail("commit before prepared seal");
        };
        match self.sink.commit() {
            Ok(g) => {
                self.state = State::Committed;
                Ok(g)
            }
            Err(e) => {
                self.abort();
                Err(e)
            }
        }
    }
    pub fn abort(&mut self) {
        if !matches!(self.state, State::Committed | State::Aborted) {
            self.sink.abort();
            self.state = State::Aborted
        }
    }
    fn fail<T>(&mut self, message: &str) -> Result<T> {
        self.abort();
        Err(validation(message))
    }
}

impl<S: HandoffSinkV2> Drop for AtomicReceiverV2<S> {
    fn drop(&mut self) {
        self.abort();
    }
}

fn validation(message: &str) -> HandoffError {
    HandoffError::Validation(message.into())
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn validate_segment_descriptor(
    segment: &SegmentDescriptorV2,
    ids: &BTreeSet<String>,
    ranges: &mut SegmentRanges,
) -> Result<()> {
    if !ids.contains(&segment.component_id)
        || segment.logical_count == 0
        || segment.elements_per_token == 0
        || segment.byte_len == 0
    {
        return Err(validation("segment has unknown component or empty range"));
    }
    validate_hex("segment", &segment.sha256)?;
    let element_width = match segment.element_type.as_str() {
        "f16_le" => 2u64,
        "f32_le" => 4u64,
        _ => return Err(validation("unsupported segment element type")),
    };
    let expected_bytes = segment
        .logical_count
        .checked_mul(segment.elements_per_token as u64)
        .and_then(|n| n.checked_mul(element_width))
        .ok_or_else(|| validation("segment byte geometry overflow"))?;
    if segment.byte_len != expected_bytes {
        return Err(validation(
            "segment byte length disagrees with row geometry",
        ));
    }
    let end = segment
        .logical_start
        .checked_add(segment.logical_count)
        .ok_or_else(|| validation("segment logical range overflow"))?;
    let entry = ranges
        .entry((segment.component_id.clone(), segment.role, segment.layer))
        .or_default();
    if entry
        .iter()
        .any(|&(start, stop)| segment.logical_start < stop && start < end)
    {
        return Err(validation("overlapping segment logical ranges"));
    }
    entry.push((segment.logical_start, end));
    Ok(())
}
fn validate_hex(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && (!b.is_ascii_alphabetic() || b.is_ascii_lowercase()))
    {
        return Err(validation(&format!(
            "{label} digest must be lowercase SHA-256"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Sink {
        live: u64,
        shadow: Vec<VerifiedSegmentV2>,
        aborted: bool,
    }
    impl HandoffSinkV2 for Sink {
        fn begin(&mut self, _: &ValidatedBeginV2) -> Result<()> {
            Ok(())
        }
        fn segment_ready(&mut self, s: VerifiedSegmentV2) -> Result<()> {
            self.shadow.push(s);
            Ok(())
        }
        fn prepare_commit(&mut self, _: &VerifiedSealV2) -> Result<()> {
            Ok(())
        }
        fn commit(&mut self) -> Result<CommittedGeneration> {
            self.live = 7;
            Ok(CommittedGeneration {
                transfer_id: "t".into(),
                generation: 7,
                installed_segments: self.shadow.len() as u32,
                installed_bytes: self.shadow.iter().map(|s| s.payload.len() as u64).sum(),
            })
        }
        fn abort(&mut self) {
            self.shadow.clear();
            self.aborted = true
        }
    }
    fn fixture() -> (ValidatedBeginV2, MacKey, Vec<u8>) {
        let payload = vec![1, 2];
        let identity = ExactIdentityV1 {
            adapter_sha256: "d".repeat(64),
            model_sha256: "a".repeat(64),
            tokenizer_sha256: "b".repeat(64),
            chat_template_sha256: "c".repeat(64),
            context_policy_sha256: "e".repeat(64),
            model_revision: "muse-rev".into(),
            tokenizer_revision: "tok-rev".into(),
        };
        let begin = BeginManifestV2 {
            protocol: LIVE_HANDOFF_PROTOCOL_V2.into(),
            transfer_id: "t".into(),
            generation: 7,
            created_unix_ms: 10,
            expires_unix_ms: 20,
            identity,
            prompt_token_ids: vec![1],
            multimodal: None,
            hmac: HmacIdentityV2 {
                key_id: "k".into(),
                epoch: 2,
            },
            components: vec![ComponentV2 {
                id: "target".into(),
                kind: ComponentKindV2::TargetKv,
                required: true,
                identity_sha256: "f".repeat(64),
            }],
            deferred_segments: false,
            segments: vec![SegmentDescriptorV2 {
                sequence: 0,
                component_id: "target".into(),
                role: SegmentRoleV2::NopeKey,
                layer: Some(3),
                logical_start: 0,
                logical_count: 1,
                element_type: "f16_le".into(),
                elements_per_token: 1,
                byte_len: 2,
                sha256: sha256_hex(&payload),
            }],
        };
        (
            ValidatedBeginV2::validate(begin, 15, "k", 2).unwrap(),
            MacKey::from_bytes([9; 32]),
            payload,
        )
    }
    fn seal(begin: &ValidatedBeginV2, key: &MacKey, payload: &[u8]) -> SealManifestV2 {
        let descriptors = canonical_json(&begin.manifest.segments[0]).unwrap();
        let core = SealCoreV2 {
            transfer_id: "t".into(),
            generation: 7,
            begin_sha256: sha256_hex(begin.canonical_bytes()),
            descriptor_sha256: sha256_hex(&descriptors),
            payload_sha256: sha256_hex(payload),
            segment_count: 1,
            total_bytes: payload.len() as u64,
        };
        let hmac_sha256 = key.tag_hex(&canonical_json(&core).unwrap()).unwrap();
        SealManifestV2 { core, hmac_sha256 }
    }
    #[test]
    fn commits_only_after_authenticated_complete_seal() {
        let (b, k, p) = fixture();
        let s = seal(&b, &k, &p);
        let mut r = AtomicReceiverV2::begin(b, k, Sink::default()).unwrap();
        r.segment_ready(0, p).unwrap();
        r.prepare_commit(s).unwrap();
        assert_eq!(r.commit().unwrap().generation, 7)
    }
    #[test]
    fn reorder_aborts_without_changing_live_generation() {
        let (b, k, _) = fixture();
        let mut r = AtomicReceiverV2::begin(b, k, Sink::default()).unwrap();
        assert!(r.segment_ready(1, vec![1, 2]).is_err());
        assert!(r.sink.aborted);
        assert_eq!(r.sink.live, 0)
    }

    #[test]
    fn deferred_descriptor_stream_commits_only_under_terminal_seal() {
        let (declared, key, payload) = fixture();
        let descriptor = declared.manifest().segments[0].clone();
        let mut manifest = declared.manifest().clone();
        manifest.deferred_segments = true;
        manifest.segments.clear();
        let begin = ValidatedBeginV2::validate(manifest, 15, "k", 2).unwrap();
        let seal = SealManifestV2::sign(
            &begin,
            std::slice::from_ref(&descriptor),
            std::slice::from_ref(&payload),
            &key,
        )
        .unwrap();
        let mut receiver = AtomicReceiverV2::begin(begin, key, Sink::default()).unwrap();
        receiver
            .segment_ready_deferred(descriptor, payload)
            .unwrap();
        assert_eq!(receiver.sink.live, 0);
        receiver.prepare_commit(seal).unwrap();
        assert_eq!(receiver.commit().unwrap().generation, 7);
    }
}
