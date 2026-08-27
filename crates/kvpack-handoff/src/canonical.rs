use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{HandoffError, Result};

/// Serialize through `serde_json::Value` with object keys sorted recursively,
/// so Rust and Python (`sort_keys=True`, compact separators) emit identical
/// bytes. The recursive sort is load-bearing: when serde_json is built with
/// `preserve_order`, `Value`'s map preserves insertion order instead of
/// sorting, and the wire contract must not depend on a Cargo feature flag.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value =
        serde_json::to_value(value).map_err(|error| HandoffError::Canonical(error.to_string()))?;
    serde_json::to_vec(&sorted_value(value))
        .map_err(|error| HandoffError::Canonical(error.to_string()))
}

fn sorted_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, sorted_value(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sorted_value).collect())
        }
        other => other,
    }
}

pub fn decode_canonical_json<T: DeserializeOwned + Serialize>(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<T> {
    if bytes.is_empty() || bytes.len() > max_bytes {
        return Err(HandoffError::Validation(format!(
            "JSON length {} is outside 1..={max_bytes}",
            bytes.len()
        )));
    }
    let decoded: T = serde_json::from_slice(bytes)
        .map_err(|error| HandoffError::Canonical(error.to_string()))?;
    let encoded = canonical_json(&decoded)?;
    if encoded != bytes {
        return Err(HandoffError::Validation(
            "JSON is not in the canonical compact sorted-key encoding".into(),
        ));
    }
    Ok(decoded)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn token_ids_sha256(token_ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"kvpack-live-token-ids-v1\0");
    for token in token_ids {
        hash.update(token.to_le_bytes());
    }
    hex::encode(hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::canonical_json;

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        // Parsed with preserve_order enabled, insertion order is b-then-a; the
        // canonical wire form must still be sorted a-then-b to match Python
        // json.dumps(sort_keys=True, separators=(",", ":")).
        let value: serde_json::Value =
            serde_json::from_str(r#"{"b":1,"a":{"d":2,"c":[3,{"f":4,"e":5}]}}"#).expect("parse");
        let bytes = canonical_json(&value).expect("canonical");
        assert_eq!(
            String::from_utf8(bytes).expect("utf8"),
            r#"{"a":{"c":[3,{"e":5,"f":4}],"d":2},"b":1}"#
        );
    }
}
