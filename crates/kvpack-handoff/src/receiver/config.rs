use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{
    BeginManifestV1, EndpointIdentityV1, ExactIdentityV1, FrameLimits, GeometryV1, HandoffError,
    HandoffStrategyV1, LayoutClassV2, MacKey, PrecisionV1, Result, ValidationLimits,
    LIVE_HANDOFF_PROTOCOL_V1, LIVE_HANDOFF_SCHEMA_V1, PORTABLE_KV_ABI_V1, PORTABLE_KV_ABI_V2,
    PORTABLE_KV_ABI_V2_PREROPE,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverBeginExpectationsV1 {
    pub cached_token_count: u32,
    pub endpoints: EndpointIdentityV1,
    pub geometry: GeometryV1,
    pub identity: ExactIdentityV1,
    pub portable_abi: String,
    pub precision: PrecisionV1,
    pub strategy: HandoffStrategyV1,
    pub token_ids_sha256: String,
    /// v2: the layout table the arm expects; empty = v1 (empty == empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layout_table: Vec<LayoutClassV2>,
}

impl ReceiverBeginExpectationsV1 {
    pub(super) fn validate_for(&self, begin: &BeginManifestV1) -> Result<()> {
        if self.cached_token_count == u32::MAX {
            return Err(HandoffError::Validation(
                "armed prompt token count overflows at its boundary token".into(),
            ));
        }
        if begin.cached_token_count != self.cached_token_count
            || begin.endpoints != self.endpoints
            || begin.geometry != self.geometry
            || begin.identity != self.identity
            || begin.portable_abi != self.portable_abi
            || begin.precision != self.precision
            || begin.strategy != self.strategy
            || begin.token_ids_sha256 != self.token_ids_sha256
            || begin.layout_table != self.layout_table
            || begin.protocol != LIVE_HANDOFF_PROTOCOL_V1
            || begin.schema_version != LIVE_HANDOFF_SCHEMA_V1
            // The ABI label is already exact-matched against the armed
            // expectation above; this is the build's closed known-label
            // set, so an ABI this receiver binary does not know fails
            // closed even when armed with it.
            || (begin.is_v2()
                && !matches!(
                    begin.portable_abi.as_str(),
                    PORTABLE_KV_ABI_V2 | PORTABLE_KV_ABI_V2_PREROPE
                ))
            || (!begin.is_v2() && self.portable_abi != PORTABLE_KV_ABI_V1)
            || self.strategy != HandoffStrategyV1::ConsumerLastPromptToken
        {
            return Err(HandoffError::Validation(
                "BEGIN does not match the complete armed fixture and identity tuple".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReceiverConfigV1 {
    pub bind: SocketAddr,
    pub output: PathBuf,
    pub server_cert: PathBuf,
    pub server_key: PathBuf,
    pub client_ca: PathBuf,
    pub expected_client_cert_sha256: String,
    pub expected_peer_ip: IpAddr,
    pub timeout: Duration,
    pub frame_limits: FrameLimits,
    pub validation_limits: ValidationLimits,
    pub begin: ReceiverBeginExpectationsV1,
    /// F1: optional tenant MAC key. When armed, the receiver requires every
    /// accepted artifact to carry an `artifact_hmac_sha256` that verifies
    /// under this key (and the begin to declare an `hmac_key_id`); a bundle
    /// that reached the engine outside the authenticated transport is
    /// forgeable without the key, so this is the gate that refuses it.
    /// `None` preserves the integrity-only behavior (transport-delegated
    /// authentication, the crate default in `lib.rs`).
    pub mac_key: Option<MacKey>,
}

impl ReceiverConfigV1 {
    pub(super) fn validate(&self) -> Result<()> {
        if !eligible_direct_address(self.bind.ip())
            || !eligible_direct_address(self.expected_peer_ip)
            || self.bind.ip() == self.expected_peer_ip
        {
            return Err(HandoffError::Validation(
                "receiver requires distinct, explicit, non-loopback local and peer addresses"
                    .into(),
            ));
        }
        if self.timeout.is_zero() || self.timeout > Duration::from_secs(3600) {
            return Err(HandoffError::Validation(
                "receiver timeout must be in 1ns..=3600s".into(),
            ));
        }
        require_sha256(
            &self.expected_client_cert_sha256,
            "expected client certificate",
        )?;
        require_sha256(&self.begin.token_ids_sha256, "armed fixture token IDs")?;
        let parent = self.output.parent().unwrap_or_else(|| Path::new("."));
        let parent = std::fs::canonicalize(parent)?;
        if parent.starts_with("/Volumes") {
            return Err(HandoffError::Validation(
                "verified live bundle staging must remain on internal storage".into(),
            ));
        }
        Ok(())
    }
}

fn eligible_direct_address(address: IpAddr) -> bool {
    !address.is_unspecified() && !address.is_loopback() && !address.is_multicast()
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(HandoffError::Validation(format!(
            "{label} SHA-256 must be 64 lowercase hexadecimal digits"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorRoleV1;

    #[test]
    fn explicit_route_and_complete_fixture_are_required_before_listen() {
        let temp = tempfile::tempdir().unwrap();
        let config = ReceiverConfigV1 {
            bind: "127.0.0.1:29590".parse().unwrap(),
            output: temp.path().join("bundle"),
            server_cert: PathBuf::from("server.crt"),
            server_key: PathBuf::from("server.key"),
            client_ca: PathBuf::from("ca.crt"),
            expected_client_cert_sha256: "1".repeat(64),
            expected_peer_ip: "192.0.2.2".parse().unwrap(),
            timeout: Duration::from_secs(10),
            frame_limits: FrameLimits::default(),
            validation_limits: ValidationLimits::default(),
            mac_key: None,
            begin: ReceiverBeginExpectationsV1 {
                cached_token_count: 1,
                endpoints: EndpointIdentityV1 {
                    consumer_engine_abi: "ferrite".into(),
                    consumer_node: "mac".into(),
                    producer_engine_abi: "vllm".into(),
                    producer_node: "spark".into(),
                    trust_domain: "lab".into(),
                },
                geometry: GeometryV1 {
                    head_dim: 64,
                    max_context_tokens: 32_768,
                    num_kv_heads: 2,
                    num_layers: 24,
                },
                identity: ExactIdentityV1 {
                    adapter_sha256: "2".repeat(64),
                    chat_template_sha256: "3".repeat(64),
                    context_policy_sha256: "4".repeat(64),
                    model_revision: "model".into(),
                    model_sha256: "5".repeat(64),
                    tokenizer_revision: "tokenizer".into(),
                    tokenizer_sha256: "6".repeat(64),
                },
                portable_abi: PORTABLE_KV_ABI_V1.into(),
                precision: PrecisionV1 {
                    compute: "float16".into(),
                    kv: "float16".into(),
                    weights: "q4_k_m".into(),
                },
                strategy: HandoffStrategyV1::ConsumerLastPromptToken,
                token_ids_sha256: "7".repeat(64),
                layout_table: Vec::new(),
            },
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn engine_version_drift_fails_the_arm_exact_match() {
        // F17: any drift in the endpoint identity (engine ABIs, nodes,
        // trust domain) must fail closed at arm time.
        let endpoints = EndpointIdentityV1 {
            consumer_engine_abi: "ferrite-qwen25-f16-v1".into(),
            consumer_node: "mac".into(),
            producer_engine_abi: "vllm-0.21.0-gb10-v1".into(),
            producer_node: "spark".into(),
            trust_domain: "lab".into(),
        };
        let expectations = ReceiverBeginExpectationsV1 {
            cached_token_count: 2,
            endpoints: endpoints.clone(),
            geometry: GeometryV1 {
                head_dim: 64,
                max_context_tokens: 32_768,
                num_kv_heads: 2,
                num_layers: 24,
            },
            identity: ExactIdentityV1 {
                adapter_sha256: "2".repeat(64),
                chat_template_sha256: "3".repeat(64),
                context_policy_sha256: "4".repeat(64),
                model_revision: "model".into(),
                model_sha256: "5".repeat(64),
                tokenizer_revision: "tokenizer".into(),
                tokenizer_sha256: "6".repeat(64),
            },
            portable_abi: PORTABLE_KV_ABI_V1.into(),
            precision: PrecisionV1 {
                compute: "float16".into(),
                kv: "float16".into(),
                weights: "q4_k_m".into(),
            },
            strategy: HandoffStrategyV1::ConsumerLastPromptToken,
            token_ids_sha256: "7".repeat(64),
            layout_table: Vec::new(),
        };
        let begin = BeginManifestV1 {
            cached_token_count: 2,
            created_unix_ms: 1,
            deadline_unix_ms: 2,
            endpoints,
            expected_layer_frames: 48,
            expected_payload_bytes: 12_288,
            geometry: expectations.geometry.clone(),
            identity: expectations.identity.clone(),
            portable_abi: expectations.portable_abi.clone(),
            precision: expectations.precision.clone(),
            protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            strategy: expectations.strategy,
            token_ids_sha256: expectations.token_ids_sha256.clone(),
            transfer_id: "8".repeat(64),
            layout_table: Vec::new(),
            schedule: None,
            hmac_key_id: None,
        };
        expectations.validate_for(&begin).unwrap();

        let mut drifted = begin.clone();
        drifted.endpoints.producer_engine_abi = "vllm-0.21.0-gb10-v2".into();
        assert!(expectations.validate_for(&drifted).is_err());

        let mut drifted = begin.clone();
        drifted.endpoints.consumer_engine_abi = "ferrite-qwen25-f16-v2".into();
        assert!(expectations.validate_for(&drifted).is_err());
    }

    #[test]
    fn prerope_abi_requires_an_exactly_armed_expectation() {
        // The receiver arm exact-matches the ABI label: a pre-RoPE begin
        // passes only a pre-RoPE-armed expectation, and a pre-RoPE arm
        // rejects a post-RoPE v2 begin. The representation can never sneak
        // past an arm for the other family.
        let mut expectations = ReceiverBeginExpectationsV1 {
            cached_token_count: 2,
            endpoints: EndpointIdentityV1 {
                consumer_engine_abi: "ferrite-qwen25-f16-v1".into(),
                consumer_node: "mac".into(),
                producer_engine_abi: "ferrite-qwen25-f16-v1".into(),
                producer_node: "mac".into(),
                trust_domain: "lab".into(),
            },
            geometry: GeometryV1 {
                head_dim: 64,
                max_context_tokens: 32_768,
                num_kv_heads: 2,
                num_layers: 24,
            },
            identity: ExactIdentityV1 {
                adapter_sha256: "2".repeat(64),
                chat_template_sha256: "3".repeat(64),
                context_policy_sha256: "4".repeat(64),
                model_revision: "model".into(),
                model_sha256: "5".repeat(64),
                tokenizer_revision: "tokenizer".into(),
                tokenizer_sha256: "6".repeat(64),
            },
            portable_abi: PORTABLE_KV_ABI_V2_PREROPE.into(),
            precision: PrecisionV1 {
                compute: "float16".into(),
                kv: "float16".into(),
                weights: "q4_k_m".into(),
            },
            strategy: HandoffStrategyV1::ConsumerLastPromptToken,
            token_ids_sha256: "7".repeat(64),
            layout_table: vec![LayoutClassV2 {
                class: "gqa-full".into(),
                dtype: "float16".into(),
                except: Vec::new(),
                from: 0,
                head_dim: 64,
                kv_heads: 2,
                roles: vec![TensorRoleV1::Key, TensorRoleV1::Value],
                step: 1,
                until: 24,
                window_tokens: 0,
            }],
        };
        // Pre-RoPE accounting: Key planes are f32 (2*2*64*4 = 1024 bytes),
        // Value planes f16 (512 bytes); 24 layers of each.
        let begin = BeginManifestV1 {
            cached_token_count: 2,
            created_unix_ms: 1,
            deadline_unix_ms: 2,
            endpoints: expectations.endpoints.clone(),
            expected_layer_frames: 48,
            expected_payload_bytes: 24 * (1_024 + 512),
            geometry: expectations.geometry.clone(),
            identity: expectations.identity.clone(),
            portable_abi: expectations.portable_abi.clone(),
            precision: expectations.precision.clone(),
            protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            strategy: expectations.strategy,
            token_ids_sha256: expectations.token_ids_sha256.clone(),
            transfer_id: "8".repeat(64),
            layout_table: expectations.layout_table.clone(),
            schedule: None,
            hmac_key_id: None,
        };
        expectations.validate_for(&begin).unwrap();

        // A post-RoPE v2 begin against the pre-RoPE arm fails (and vice
        // versa): the ABI label is part of the armed fixture.
        let mut post_rope = begin.clone();
        post_rope.portable_abi = PORTABLE_KV_ABI_V2.into();
        assert!(expectations.validate_for(&post_rope).is_err());
        expectations.portable_abi = PORTABLE_KV_ABI_V2.into();
        assert!(expectations.validate_for(&begin).is_err());
    }
}
