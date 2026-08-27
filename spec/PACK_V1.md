# Authenticated kvpack manifest envelope

> **Pre-freeze draft.** These bytes and magics remain mutable until Z1.

The file is exactly `4096-byte header || stored canonical manifest || 4096-byte
commit footer`. All integers are little-endian. Reserved bytes are zero and
checked. This object contains no record stream, external envelope, path,
catalog epoch, or extension block.

## Header

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | u8[8] | `KVPKP1\0\0` |
| 8 | u16 | draft wire version 1 |
| 10 | u16 | header bytes, 4096 |
| 12 | u32 | alignment, 4096 |
| 16 | u32 | flags; bit 0 is manifest AEAD |
| 20 | u32 | zero |
| 24 | u64 | stored manifest bytes, including AEAD tag |
| 32 | u64 | plaintext canonical manifest bytes |
| 40 | u64 | manifest key epoch |
| 48 | u64 | zero; catalog epoch is not durable identity |
| 56 | u8[32] | tenant namespace |
| 88 | u8[32] | canonical manifest ID |
| 120 | u8[16] | random AEAD salt, zero without AEAD |
| 136 | u8[12] | random AEAD nonce, zero without AEAD |
| 148 | u8[32] | SHA-256 header digest; this field is zero while hashing |
| 180 | u8[3916] | zero |

Manifest AEAD is ChaCha20-Poly1305 with the exact final header as AAD. An HKDF
data key uses the epoch manifest-encryption key, random salt, and info
`kvpack/v1/manifest-aead\0 || manifest_id`.

## Commit footer

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | u8[8] | `KVCMT1\0\0` |
| 8 | u16 | draft version 1 |
| 10 | u16 | footer bytes, 4096 |
| 12 | u32 | zero |
| 16 | u64 | exact stored manifest bytes |
| 24 | u64 | exact manifest key epoch |
| 32 | u64 | zero |
| 40 | u64 | exact complete file bytes |
| 48 | u8[32] | exact manifest ID |
| 80 | u8[32] | HMAC-SHA-256 |
| 112 | u8[3984] | zero |

The HMAC domain is `kvpack/v1/manifest-auth\0` and binds the complete header,
stored manifest/tag, stored length, key epoch, and complete file length. The
manifest ID is defined by `IDENTITY_V1.md` over canonical plaintext bytes.

## Decode order

The decoder checks outer size, header/footer magic/version/size, all flags and
reserved bytes, checked length arithmetic, padding, and exact header/footer
linkage before authenticating the header digest or HMAC. It then verifies HMAC
and optional AEAD, verifies plaintext
length and manifest ID, performs the complete `KVMNF1` canonical decode and
byte-identical re-encode, checks header/manifest tenant and epoch equality, and
only then runs semantic/graph validation. Parent-chain and chunk validation are
later phases.

`IOKVPK1`, `IOKVENC`, zlib/Q8 drafts, record magics, and every earlier
development object are invalid; migration tooling is separate from this
decoder.
