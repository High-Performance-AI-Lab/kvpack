pub const ALIGNMENT: usize = 4096;
pub const CHUNK_HEADER_BYTES: usize = ALIGNMENT;
pub const PACK_HEADER_BYTES: usize = ALIGNMENT;
pub const PACK_FOOTER_BYTES: usize = ALIGNMENT;
pub const MAX_CHUNK_PLAINTEXT: usize = 4 * 1024 * 1024;
pub const CODEC_FRAME_HEADER_BYTES: usize = 16;
/// Canonical RLE can add one control byte per 128 literal bytes.
pub const MAX_CODEC_OVERHEAD: usize = CODEC_FRAME_HEADER_BYTES + MAX_CHUNK_PLAINTEXT.div_ceil(128);
/// Largest possible aligned object: header, incompressible plaintext plus the
/// AEAD tag, and alignment padding. Kept public so manifest validation and the
/// object decoder enforce one identical bound.
pub const MAX_CHUNK_OBJECT_BYTES: usize =
    CHUNK_HEADER_BYTES + MAX_CHUNK_PLAINTEXT + MAX_CODEC_OVERHEAD + 16 + ALIGNMENT;
pub const MAX_RANK: usize = 8;
/// One full base plus at most seven delta manifests.  An attempted eighth
/// append must compact to a new full manifest by reference.
pub const MAX_DELTA_DEPTH: u8 = 7;
pub const PREFIX_BLOCK_TOKENS: usize = 256;
/// Maximum number of cuts in one derived cut chain — 128k tokens at the
/// 256-token tail stride.  Derivation fails closed beyond this bound.
pub const MAX_CUT_CHAIN_CUTS: usize = 512;
pub const MAX_STATE_NAME_BYTES: usize = 255;
pub const MAX_STATES: usize = 65_536;
pub const MAX_ATOMIC_GROUPS: usize = MAX_STATES;
pub const MAX_CHUNKS_PER_STATE: usize = 65_536;
pub const MAX_DEPENDENCIES_PER_STATE: usize = 64;
pub const MAX_MANIFEST_BYTES: usize = 256 * 1024 * 1024;
pub const ZERO_ID: [u8; 32] = [0u8; 32];

/// Production-v1 magic.  It intentionally differs from every development pack.
pub const PACK_MAGIC: &[u8; 8] = b"KVPKP1\0\0";
pub const CHUNK_MAGIC: &[u8; 8] = b"KVCHK1\0\0";
pub const FOOTER_MAGIC: &[u8; 8] = b"KVCMT1\0\0";
pub const MANIFEST_MAGIC: &[u8; 8] = b"KVMNF1\0\0";
pub const FAMILY_MAGIC: &[u8; 8] = b"KVFAM1\0\0";
pub const SCHEMA_MAGIC: &[u8; 8] = b"KVRCS1\0\0";
pub const STATE_SCHEMA_MAGIC: &[u8; 8] = b"KVSTS1\0\0";
pub const STATS_SIDECAR_MAGIC: &[u8; 8] = b"KVSSC1\0\0";
pub const QUANT_K_MAGIC: &[u8; 8] = b"KVQK1\0\0\0";
pub const QUANT_V_MAGIC: &[u8; 8] = b"KVQV1\0\0\0";
pub const RAW_FRAME_MAGIC: &[u8; 8] = b"KVRAW1\0\0";
pub const LOSSLESS_FRAME_MAGIC: &[u8; 8] = b"KVRLE1\0\0";
pub const WIRE_VERSION: u16 = 1;

pub const FLAG_ENCRYPTED: u32 = 1;
pub const KNOWN_FLAGS: u32 = FLAG_ENCRYPTED;

/// Byte offset inside a chunk header where the optional statistics sidecar
/// begins (`u16` canonical length, then the canonical sidecar bytes).  The
/// sidecar deliberately has no flag bit: its presence is the nonzero length
/// prefix in what pre-sidecar readers validate as zero reserved tail, so
/// every pre-sidecar reader rejects sidecar objects as non-canonical and the
/// unknown-flag error phase ordering is unchanged.
pub const CHUNK_HEADER_SIDECAR_OFFSET: usize = 236;
/// Largest canonical sidecar that fits the chunk header tail.
pub const MAX_STATS_SIDECAR_BYTES: usize = CHUNK_HEADER_BYTES - CHUNK_HEADER_SIDECAR_OFFSET - 2;
