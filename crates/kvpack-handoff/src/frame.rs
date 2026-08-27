use std::io::{Read, Write};

use crate::{
    canonical_json, decode_canonical_json, AbortManifestV1, AckManifestV1, BeginManifestV1,
    HandoffError, LayerHeaderV1, Result, SealManifestV1, FRAME_MAGIC, LIVE_HANDOFF_SCHEMA_V1,
};

pub const FRAME_HEADER_BYTES: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameKind {
    Begin = 1,
    Layer = 2,
    Seal = 3,
    Abort = 4,
    Ack = 5,
}

impl TryFrom<u8> for FrameKind {
    type Error = HandoffError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Begin),
            2 => Ok(Self::Layer),
            3 => Ok(Self::Seal),
            4 => Ok(Self::Abort),
            5 => Ok(Self::Ack),
            _ => Err(HandoffError::Validation(format!(
                "unknown live handoff frame kind {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    Begin(Box<BeginManifestV1>),
    Layer(LayerHeaderV1, Vec<u8>),
    Seal(SealManifestV1),
    Abort(AbortManifestV1),
    Ack(AckManifestV1),
}

/// A decoded frame envelope whose payload has not yet been read.
///
/// Receivers use this split boundary to acquire a bounded layer permit after
/// authenticating a K-plane descriptor but before allocating or reading its
/// canonical payload.  `FrameReader` refuses to read another envelope until
/// the pending payload has been consumed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameHeader {
    Begin(Box<BeginManifestV1>),
    Layer(LayerHeaderV1),
    Seal(SealManifestV1),
    Abort(AbortManifestV1),
    Ack(AckManifestV1),
}

impl FrameHeader {
    pub const fn kind(&self) -> FrameKind {
        match self {
            Self::Begin(_) => FrameKind::Begin,
            Self::Layer(_) => FrameKind::Layer,
            Self::Seal(_) => FrameKind::Seal,
            Self::Abort(_) => FrameKind::Abort,
            Self::Ack(_) => FrameKind::Ack,
        }
    }

    pub fn layer(&self) -> Option<&LayerHeaderV1> {
        match self {
            Self::Layer(header) => Some(header),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLimits {
    pub max_json_bytes: usize,
    pub max_payload_bytes: u64,
}

impl Default for FrameLimits {
    fn default() -> Self {
        Self {
            max_json_bytes: 1024 * 1024,
            max_payload_bytes: 64 * 1024 * 1024,
        }
    }
}

fn encode_parts(frame: &Frame) -> Result<(FrameKind, Vec<u8>, &[u8])> {
    match frame {
        Frame::Begin(value) => Ok((FrameKind::Begin, canonical_json(value)?, &[])),
        Frame::Layer(header, payload) => Ok((
            FrameKind::Layer,
            canonical_json(header)?,
            payload.as_slice(),
        )),
        Frame::Seal(value) => Ok((FrameKind::Seal, canonical_json(value)?, &[])),
        Frame::Abort(value) => Ok((FrameKind::Abort, canonical_json(value)?, &[])),
        Frame::Ack(value) => Ok((FrameKind::Ack, canonical_json(value)?, &[])),
    }
}

pub fn write_frame<W: Write>(writer: &mut W, frame: &Frame, limits: FrameLimits) -> Result<()> {
    let (kind, json, payload) = encode_parts(frame)?;
    if json.is_empty() || json.len() > limits.max_json_bytes {
        return Err(HandoffError::Validation(format!(
            "frame JSON length {} is outside configured bounds",
            json.len()
        )));
    }
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| HandoffError::Validation("payload length exceeds u64".into()))?;
    if payload_len > limits.max_payload_bytes {
        return Err(HandoffError::Validation(format!(
            "frame payload length {payload_len} exceeds {}",
            limits.max_payload_bytes
        )));
    }
    let json_len = u32::try_from(json.len())
        .map_err(|_| HandoffError::Validation("frame JSON length exceeds u32".into()))?;
    let mut header = [0u8; FRAME_HEADER_BYTES];
    header[0..4].copy_from_slice(&FRAME_MAGIC);
    header[4] = LIVE_HANDOFF_SCHEMA_V1 as u8;
    header[5] = kind as u8;
    header[6..8].copy_from_slice(&0u16.to_be_bytes());
    header[8..12].copy_from_slice(&json_len.to_be_bytes());
    header[12..20].copy_from_slice(&payload_len.to_be_bytes());
    writer.write_all(&header)?;
    writer.write_all(&json)?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

pub fn read_frame<R: Read>(reader: &mut R, limits: FrameLimits) -> Result<Frame> {
    let mut reader = FrameReader::new(reader, limits);
    reader.read_header()?;
    reader.read_payload()
}

/// Stateful split reader for one ordered live-handoff byte stream.
pub struct FrameReader<R> {
    reader: R,
    limits: FrameLimits,
    pending: Option<(FrameHeader, u64)>,
}

impl<R: Read> FrameReader<R> {
    pub fn new(reader: R, limits: FrameLimits) -> Self {
        Self {
            reader,
            limits,
            pending: None,
        }
    }

    /// Decode and validate the fixed envelope plus canonical JSON, leaving a
    /// layer payload unread so the caller can acquire its memory permit.
    pub fn read_header(&mut self) -> Result<&FrameHeader> {
        if self.pending.is_some() {
            return Err(HandoffError::Validation(
                "cannot read another frame header before consuming the pending payload".into(),
            ));
        }
        let mut header = [0u8; FRAME_HEADER_BYTES];
        self.reader.read_exact(&mut header)?;
        if header[0..4] != FRAME_MAGIC
            || header[4] != LIVE_HANDOFF_SCHEMA_V1 as u8
            || header[6..8] != [0, 0]
        {
            return Err(HandoffError::Validation(
                "invalid frame magic, version, or reserved flags".into(),
            ));
        }
        let kind = FrameKind::try_from(header[5])?;
        let json_len = u32::from_be_bytes(header[8..12].try_into().expect("fixed JSON length"));
        let payload_len =
            u64::from_be_bytes(header[12..20].try_into().expect("fixed payload length"));
        let json_len = usize::try_from(json_len)
            .map_err(|_| HandoffError::Validation("frame JSON length exceeds usize".into()))?;
        if json_len == 0
            || json_len > self.limits.max_json_bytes
            || payload_len > self.limits.max_payload_bytes
            || (kind != FrameKind::Layer && payload_len != 0)
        {
            return Err(HandoffError::Validation(
                "frame lengths violate the configured bounds or frame kind".into(),
            ));
        }
        let mut json = vec![0u8; json_len];
        self.reader.read_exact(&mut json)?;
        let decoded = match kind {
            FrameKind::Begin => FrameHeader::Begin(Box::new(decode_canonical_json(
                &json,
                self.limits.max_json_bytes,
            )?)),
            FrameKind::Layer => {
                let header: LayerHeaderV1 =
                    decode_canonical_json(&json, self.limits.max_json_bytes)?;
                if header.byte_length != payload_len {
                    return Err(HandoffError::Validation(
                        "layer header length does not equal frame payload length".into(),
                    ));
                }
                FrameHeader::Layer(header)
            }
            FrameKind::Seal => {
                FrameHeader::Seal(decode_canonical_json(&json, self.limits.max_json_bytes)?)
            }
            FrameKind::Abort => {
                FrameHeader::Abort(decode_canonical_json(&json, self.limits.max_json_bytes)?)
            }
            FrameKind::Ack => {
                FrameHeader::Ack(decode_canonical_json(&json, self.limits.max_json_bytes)?)
            }
        };
        self.pending = Some((decoded, payload_len));
        Ok(&self
            .pending
            .as_ref()
            .expect("just installed pending frame")
            .0)
    }

    /// Consume the payload belonging to the most recently decoded envelope.
    pub fn read_payload(&mut self) -> Result<Frame> {
        let (header, payload_len) = self
            .pending
            .take()
            .ok_or_else(|| HandoffError::Validation("no pending frame header to consume".into()))?;
        match header {
            FrameHeader::Begin(value) => Ok(Frame::Begin(value)),
            FrameHeader::Layer(header) => {
                let payload_len_usize = usize::try_from(payload_len).map_err(|_| {
                    HandoffError::Validation("frame payload length exceeds usize".into())
                })?;
                // Reserve exactly the declared length without zero-filling a
                // buffer the read is about to overwrite; a short stream is
                // the same fail-closed unexpected-EOF as `read_exact`.
                let mut payload = Vec::with_capacity(payload_len_usize);
                self.reader
                    .by_ref()
                    .take(payload_len)
                    .read_to_end(&mut payload)?;
                if payload.len() != payload_len_usize {
                    return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
                }
                Ok(Frame::Layer(header, payload))
            }
            FrameHeader::Seal(value) => Ok(Frame::Seal(value)),
            FrameHeader::Abort(value) => Ok(Frame::Abort(value)),
            FrameHeader::Ack(value) => Ok(Frame::Ack(value)),
        }
    }

    pub fn into_inner(self) -> Result<R> {
        if self.pending.is_some() {
            return Err(HandoffError::Validation(
                "cannot release a frame reader with an unread payload".into(),
            ));
        }
        Ok(self.reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EndpointIdentityV1, ExactIdentityV1, GeometryV1, HandoffStrategyV1, PrecisionV1,
        TensorRoleV1, LIVE_HANDOFF_PROTOCOL_V1, PORTABLE_KV_ABI_V1,
    };

    fn begin() -> BeginManifestV1 {
        BeginManifestV1 {
            cached_token_count: 2,
            created_unix_ms: 100,
            deadline_unix_ms: 200,
            endpoints: EndpointIdentityV1 {
                consumer_engine_abi: "ferrite-v1".into(),
                consumer_node: "mac".into(),
                producer_engine_abi: "vllm-v1".into(),
                producer_node: "spark".into(),
                trust_domain: "lab".into(),
            },
            expected_layer_frames: 2,
            expected_payload_bytes: 32,
            geometry: GeometryV1 {
                head_dim: 2,
                max_context_tokens: 8,
                num_kv_heads: 2,
                num_layers: 1,
            },
            identity: ExactIdentityV1 {
                adapter_sha256: "0".repeat(64),
                chat_template_sha256: "1".repeat(64),
                context_policy_sha256: "2".repeat(64),
                model_revision: "repo@rev".into(),
                model_sha256: "3".repeat(64),
                tokenizer_revision: "repo@rev".into(),
                tokenizer_sha256: "4".repeat(64),
            },
            portable_abi: PORTABLE_KV_ABI_V1.into(),
            precision: PrecisionV1 {
                compute: "float16".into(),
                kv: "float16".into(),
                weights: "q4_k_m".into(),
            },
            protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            strategy: HandoffStrategyV1::ConsumerLastPromptToken,
            token_ids_sha256: "5".repeat(64),
            transfer_id: "6".repeat(64),
            layout_table: Vec::new(),
            schedule: None,
            hmac_key_id: None,
        }
    }

    #[test]
    fn begin_and_layer_round_trip() {
        let limits = FrameLimits::default();
        for frame in [
            Frame::Begin(Box::new(begin())),
            Frame::Layer(
                LayerHeaderV1 {
                    byte_length: 4,
                    layer: 0,
                    logical_token_end: 2,
                    logical_token_start: 0,
                    role: TensorRoleV1::Key,
                    schema_version: 1,
                    sequence: 0,
                    sha256: "a".repeat(64),
                    shape: [2, 1, 1],
                    transfer_id: "6".repeat(64),
                    dtype: None,
                    layout_class: None,
                },
                vec![1, 2, 3, 4],
            ),
        ] {
            let mut bytes = Vec::new();
            write_frame(&mut bytes, &frame, limits).unwrap();
            assert_eq!(read_frame(&mut bytes.as_slice(), limits).unwrap(), frame);
        }
    }

    #[test]
    fn unknown_flags_and_oversize_fail_before_allocation() {
        let mut bytes = Vec::new();
        write_frame(
            &mut bytes,
            &Frame::Begin(Box::new(begin())),
            FrameLimits::default(),
        )
        .unwrap();
        bytes[7] = 1;
        assert!(read_frame(&mut bytes.as_slice(), FrameLimits::default()).is_err());

        let mut header = [0u8; FRAME_HEADER_BYTES];
        header[0..4].copy_from_slice(&FRAME_MAGIC);
        header[4] = 1;
        header[5] = FrameKind::Layer as u8;
        header[8..12].copy_from_slice(&1u32.to_be_bytes());
        header[12..20].copy_from_slice(&(u64::MAX).to_be_bytes());
        assert!(read_frame(&mut header.as_slice(), FrameLimits::default()).is_err());
    }

    #[test]
    fn split_reader_leaves_layer_payload_unread_until_requested() {
        let frame = Frame::Layer(
            LayerHeaderV1 {
                byte_length: 4,
                layer: 0,
                logical_token_end: 2,
                logical_token_start: 0,
                role: TensorRoleV1::Key,
                schema_version: 1,
                sequence: 0,
                sha256: "a".repeat(64),
                shape: [2, 1, 1],
                transfer_id: "6".repeat(64),
                dtype: None,
                layout_class: None,
            },
            vec![1, 2, 3, 4],
        );
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &frame, FrameLimits::default()).unwrap();
        let cursor = std::io::Cursor::new(bytes);
        let mut reader = FrameReader::new(cursor, FrameLimits::default());
        assert!(matches!(
            reader.read_header().unwrap(),
            FrameHeader::Layer(_)
        ));
        assert_eq!(
            reader.reader.position() as usize + 4,
            reader.reader.get_ref().len()
        );
        assert!(reader.read_header().is_err());
        assert_eq!(reader.read_payload().unwrap(), frame);
    }
}
