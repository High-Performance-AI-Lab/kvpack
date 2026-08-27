use std::fmt;

/// A pack violates the v1 wire contract.
///
/// Variants are the API contract; `Display` output reproduces the Python
/// reference's error strings verbatim and is pinned by a conformance test.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackError {
    /// `unknown {what} {value}` — fail-closed enum decoding.
    UnknownEnum { what: &'static str, value: u64 },
    /// `{what} is outside u64 range` — numeric range violations.
    U64Range { what: &'static str },
    /// Magic/version/size identity failures.
    BadMagic(&'static str),
    /// Checksum, digest, or Merkle mismatches.
    Checksum(&'static str),
    /// Non-zero reserved bytes or flags.
    Reserved(&'static str),
    /// Truncated structures.
    Truncated(&'static str),
    /// Numeric/length bounds violations.
    Bounds(&'static str),
    /// Artifact/codec/type semantic rules.
    Semantics(&'static str),
    /// Record/commit graph rules (parents, sequences, chains, footer linkage).
    Graph(&'static str),
    /// Keyed authenticity, AEAD, digest, or epoch failure.
    Authentication(&'static str),
    /// Unsupported or malformed codec frame.
    Codec(&'static str),
    /// A writer/state machine was permanently poisoned by an earlier failure.
    Poisoned(&'static str),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackError::UnknownEnum { what, value } => write!(f, "unknown {what} {value}"),
            PackError::U64Range { what } => write!(f, "{what} is outside u64 range"),
            PackError::BadMagic(msg)
            | PackError::Checksum(msg)
            | PackError::Reserved(msg)
            | PackError::Truncated(msg)
            | PackError::Bounds(msg)
            | PackError::Semantics(msg)
            | PackError::Graph(msg)
            | PackError::Authentication(msg)
            | PackError::Codec(msg)
            | PackError::Poisoned(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for PackError {}
