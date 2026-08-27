# Authenticated kvchunk production v1

> **Pre-freeze draft.** These bytes and magics remain mutable until Z1.

A chunk is `4096-byte header || framed payload-or-ciphertext/tag || zero
padding` to a 4096-byte boundary. Plaintext is 1 through 4 MiB and contains an
integral number of logical state tokens. The authenticated manifest supplies
the exact family, state key, token/byte span, object identity, digest, and epoch.

## Header

| Offset | Type | Meaning |
| ---: | --- | --- |
| 0 | u8[8] | `KVCHK1\0\0` |
| 8 | u16 | draft wire version 1 |
| 10 | u16 | header bytes, 4096 |
| 12 | u32 | alignment, 4096 |
| 16 | u32 | flags; bit 0 is AEAD; no other bit is accepted |
| 20 | u16 | codec: raw=1, lossless=2 |
| 22 | u16 | codec version, exactly 1 |
| 24 | u32 | decoded plaintext bytes |
| 28 | u32 | complete codec-frame bytes before AEAD tag |
| 32 | u32 | stored payload bytes including tag when encrypted |
| 36 | u32 | complete aligned object bytes |
| 40 | u64 | chunk-object key epoch |
| 48 | u8[32] | tenant namespace |
| 80 | u8[32] | static representation-family digest |
| 112 | u8[32] | keyed plaintext chunk-content ID |
| 144 | u8[32] | epoch-specific object key |
| 176 | u8[16] | random salt, zero without AEAD |
| 192 | u8[12] | random nonce, zero without AEAD |
| 204 | u8[32] | SHA-256 header digest; this field is zero while hashing |
| 236 | u8[3860] | zero |

The stored-object digest is SHA-256 over the complete header, stored payload,
tag, and padding. It is supplied by an authenticated manifest and is checked
after all framing/reserved checks and before semantic or plaintext use.

The object key is HMAC-SHA-256 under the epoch object-identity key over domain
`kvpack/v1/chunk-object\0`, tenant, family digest, content ID, epoch,
codec/version, encryption flag, salt, and nonce. It is distinct from both the
stable content ID and stored-object digest.

Encrypted chunks use ChaCha20-Poly1305. HKDF-SHA-256 derives a data key from the
epoch chunk-encryption key with the random salt and info
`kvpack/v1/chunk-aead\0 || content_id || object_key`. The exact final header is
AAD. Derived data keys are zeroized after use.

## Codec frames

Every chunk is independently framed, including raw. Both frame headers are 16
bytes:

```text
magic[8] || u16 version=1 || u16 zero || u32 decoded_bytes
```

Raw uses `KVRAW1\0\0` followed by exactly `decoded_bytes` bytes.

Lossless uses `KVRLE1\0\0` followed by canonical PackBits-style packets. A
control byte encodes `length=(control & 0x7f)+1`. If bit 7 is clear, exactly
`length` literal bytes follow. If bit 7 is set, one byte follows and is repeated
`length` times. The encoder uses a repeat packet exactly for runs of at least
three equal bytes (maximum 128); all other bytes are joined into the longest
literal packet up to 128 ending before such a run. A decoder must produce the
exact declared size and byte-identically re-encode the frame, rejecting
alternate packetizations.

Lossless worst-case expansion is bounded by the 16-byte frame plus one control
byte per 128 plaintext bytes. Q8, Q4, Q2, zlib draft frames, and every other
codec/magic are unknown production values.

## Verification order

Length/magic/version/flags/enums/size arithmetic/reserved/padding are checked
first. Next come the stored digest and header digest, then tenant/family/epoch
and object-key binding, then AEAD, canonical codec decode/re-encode, and finally
the expected state/span plaintext content ID.
