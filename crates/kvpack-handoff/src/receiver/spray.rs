//! WS9 TENT-lite multi-stream spraying (receiver half, opt-in).
//!
//! When `KVPACK_HANDOFF_STREAMS` (default 1) is set to K > 1 the receiver
//! accepts K mTLS connections on the same listener; the producer must set
//! `KVPACK_SPRAY_STREAMS` to the same K. Both ends MUST agree: with the env
//! unset the receiver runs the exact single-connection code path.
//!
//! The wire protocol is unchanged: frames carry their `sequence`, BEGIN and
//! the terminal SEAL/ABORT travel on stream 0 only, every layer frame travels
//! on exactly one stream, and each stream is internally FIFO. Correctness of
//! reassembly therefore never depends on the sender's stream assignment.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::mpsc;
use std::thread;

use crate::{
    BeginManifestV1, Frame, FrameHeader, FrameLimits, FrameReader, HandoffError, LayerHeaderV1,
    LayerPermitPoolV1, LayerPermitV1, Result, TensorRoleV1,
};

pub const SPRAY_STREAMS_ENV: &str = "KVPACK_HANDOFF_STREAMS";

/// Number of parallel streams the receiver should accept; 1 (the default)
/// keeps the byte-identical single-connection path.
pub fn configured_spray_streams() -> usize {
    std::env::var(SPRAY_STREAMS_ENV)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|streams| *streams >= 1)
        .unwrap_or(1)
}

/// One fully read layer plane plus the permit acquired before its payload.
#[derive(Debug)]
pub struct SprayLayerFrame {
    pub header: LayerHeaderV1,
    pub payload: Vec<u8>,
    pub permit: Option<LayerPermitV1>,
}

/// Events produced by the per-stream reader threads.
pub enum SprayEvent {
    /// BEGIN (valid on stream 0 only, as the session's first event).
    Begin(Box<BeginManifestV1>),
    Layer(SprayLayerFrame),
    /// A non-layer frame ended a stream: SEAL/ABORT on stream 0 terminates
    /// the session; on any auxiliary stream it is a protocol error.
    Terminal {
        stream: usize,
        frame: Frame,
    },
    /// A reader thread failed; the session must abort fail-closed.
    Failed(HandoffError),
}

/// Deterministic strictly-ascending reassembly of sprayed layer frames.
///
/// Accepts arrivals in any interleaving and yields them in sequence order.
/// Duplicates and sequences at/past the declared frame count are hard errors.
pub struct SequenceMuxV1 {
    expected: u32,
    next: u32,
    stash: BTreeMap<u32, SprayLayerFrame>,
}

impl SequenceMuxV1 {
    pub fn new(expected: u32) -> Self {
        Self {
            expected,
            next: 0,
            stash: BTreeMap::new(),
        }
    }

    pub fn accept(&mut self, frame: SprayLayerFrame) -> Result<()> {
        let sequence = frame.header.sequence;
        if sequence >= self.expected {
            return Err(HandoffError::Validation(format!(
                "sprayed layer sequence {sequence} exceeds the declared frame count"
            )));
        }
        if sequence < self.next || self.stash.contains_key(&sequence) {
            return Err(HandoffError::Validation(format!(
                "duplicate sprayed layer sequence {sequence}"
            )));
        }
        self.stash.insert(sequence, frame);
        Ok(())
    }

    /// Pop the next in-order frame when it has arrived.
    pub fn pop_ready(&mut self) -> Option<SprayLayerFrame> {
        let frame = self.stash.remove(&self.next)?;
        self.next += 1;
        Some(frame)
    }

    pub fn is_complete(&self) -> bool {
        self.next == self.expected
    }
}

/// Owned per-stream reader threads plus their shared event channel.
///
/// Each thread returns the stream it owned once it observes a terminal frame
/// (so stream 0's connection can carry the receiver's ACK); it returns
/// `Ok(None)` after reporting an error or a disconnected session channel.
pub struct SprayReaders<R> {
    pub events: mpsc::Receiver<SprayEvent>,
    pub threads: Vec<thread::JoinHandle<Result<Option<R>>>>,
}

/// Spawn one blocking reader thread per stream. Threads are deliberately NOT
/// joined by the session until their sockets have been shut down: a thread
/// parked in a permit acquire wakes as in-flight pairs complete (or when the
/// coordinator drops its permits on abort), then exits on socket error or on
/// the disconnected channel, so no thread can leak or deadlock the pool.
pub fn spawn_spray_readers<R: Read + Send + 'static>(
    readers: Vec<R>,
    permit_pool: LayerPermitPoolV1,
    limits: FrameLimits,
) -> SprayReaders<R> {
    let (tx, events) = mpsc::channel();
    let threads = readers
        .into_iter()
        .enumerate()
        .map(|(stream, reader)| {
            let tx = tx.clone();
            let pool = permit_pool.clone();
            thread::Builder::new()
                .name(format!("kvpack-spray-reader-{stream}"))
                .spawn(
                    move || match spray_reader_loop(stream, reader, &pool, limits, &tx) {
                        Ok(done) => Ok(done),
                        Err(error) => {
                            let _ = tx.send(SprayEvent::Failed(error));
                            Ok(None)
                        }
                    },
                )
                .map_err(HandoffError::Io)
        })
        .collect::<Result<Vec<_>>>()
        .unwrap_or_else(|error| {
            // A spawn failure is reported through the channel so the session
            // aborts through the same fail-closed path as a reader error.
            let _ = tx.send(SprayEvent::Failed(error));
            Vec::new()
        });
    SprayReaders { events, threads }
}

fn spray_reader_loop<R: Read>(
    stream: usize,
    reader: R,
    permit_pool: &LayerPermitPoolV1,
    limits: FrameLimits,
    tx: &mpsc::Sender<SprayEvent>,
) -> Result<Option<R>> {
    let mut reader = FrameReader::new(reader, limits);
    loop {
        let header = reader.read_header()?.clone();
        match header {
            FrameHeader::Begin(_) => {
                let Frame::Begin(begin) = reader.read_payload()? else {
                    unreachable!("decoded BEGIN header must produce BEGIN frame")
                };
                if tx.send(SprayEvent::Begin(begin)).is_err() {
                    return Ok(None);
                }
            }
            FrameHeader::Layer(layer) => {
                // K planes hold a canonical-memory permit from BEFORE their
                // payload is read until their pair completes in sequence
                // order on the session thread; V planes never block. The
                // shared capacity-2 pool bounds sprayed memory exactly as in
                // the single-stream path.
                let permit = if layer.role == TensorRoleV1::Key {
                    Some(permit_pool.acquire()?)
                } else {
                    None
                };
                let Frame::Layer(header, payload) = reader.read_payload()? else {
                    unreachable!("decoded layer header must produce layer frame")
                };
                if tx
                    .send(SprayEvent::Layer(SprayLayerFrame {
                        header,
                        payload,
                        permit,
                    }))
                    .is_err()
                {
                    return Ok(None);
                }
            }
            FrameHeader::Seal(_) | FrameHeader::Abort(_) | FrameHeader::Ack(_) => {
                let frame = reader.read_payload()?;
                let stream_owner = reader.into_inner()?;
                let _ = tx.send(SprayEvent::Terminal { stream, frame });
                return Ok(Some(stream_owner));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sha256_hex, write_frame, SealManifestV1, TensorRoleV1 as Role, LIVE_HANDOFF_SCHEMA_V1,
    };
    use std::io::Write as _;
    use std::net::{TcpListener, TcpStream};

    fn layer(sequence: u32, role: Role, bytes: &[u8]) -> SprayLayerFrame {
        SprayLayerFrame {
            header: LayerHeaderV1 {
                byte_length: bytes.len() as u64,
                layer: sequence / 2,
                logical_token_end: 2,
                logical_token_start: 0,
                role,
                schema_version: LIVE_HANDOFF_SCHEMA_V1,
                sequence,
                sha256: sha256_hex(bytes),
                shape: [2, 1, 1],
                transfer_id: "spray-test".into(),
                dtype: None,
                layout_class: None,
            },
            payload: bytes.to_vec(),
            permit: None,
        }
    }

    fn drain_ready(mux: &mut SequenceMuxV1) -> Vec<u32> {
        let mut order = Vec::new();
        while let Some(frame) = mux.pop_ready() {
            order.push(frame.header.sequence);
        }
        order
    }

    #[test]
    fn mux_reassembles_interleaved_two_stream_arrival() {
        // Stream 0 carries 0,2,4 and stream 1 carries 1,3,5; arrivals
        // interleave with stream 1 running ahead.
        let mut mux = SequenceMuxV1::new(6);
        mux.accept(layer(1, Role::Value, b"a")).unwrap();
        assert!(drain_ready(&mut mux).is_empty());
        mux.accept(layer(0, Role::Key, b"b")).unwrap();
        assert_eq!(drain_ready(&mut mux), vec![0, 1]);
        mux.accept(layer(3, Role::Value, b"c")).unwrap();
        mux.accept(layer(5, Role::Value, b"d")).unwrap();
        mux.accept(layer(2, Role::Key, b"e")).unwrap();
        assert_eq!(drain_ready(&mut mux), vec![2, 3]);
        mux.accept(layer(4, Role::Key, b"f")).unwrap();
        assert_eq!(drain_ready(&mut mux), vec![4, 5]);
        assert!(mux.is_complete());
    }

    #[test]
    fn mux_rejects_duplicate_and_out_of_range_sequences() {
        let mut mux = SequenceMuxV1::new(2);
        mux.accept(layer(0, Role::Key, b"a")).unwrap();
        assert!(mux.accept(layer(0, Role::Key, b"a")).is_err());
        assert!(mux.accept(layer(2, Role::Key, b"a")).is_err());
        assert_eq!(drain_ready(&mut mux), vec![0]);
        mux.accept(layer(1, Role::Value, b"b")).unwrap();
        assert_eq!(drain_ready(&mut mux), vec![1]);
        // A sequence already emitted is a duplicate too.
        assert!(mux.accept(layer(1, Role::Value, b"b")).is_err());
        assert!(mux.is_complete());
    }

    fn frame_bytes(sequence: u32, role: Role, bytes: &[u8]) -> Vec<u8> {
        let mut buffer = Vec::new();
        write_frame(
            &mut buffer,
            &Frame::Layer(layer(sequence, role, bytes).header, bytes.to_vec()),
            FrameLimits::default(),
        )
        .unwrap();
        buffer
    }

    #[test]
    fn reader_threads_deliver_interleaved_streams_for_ordered_reassembly() {
        let listener_a = TcpListener::bind("127.0.0.1:0").unwrap();
        let listener_b = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client_a = TcpStream::connect(listener_a.local_addr().unwrap()).unwrap();
        let mut client_b = TcpStream::connect(listener_b.local_addr().unwrap()).unwrap();
        let (server_a, _) = listener_a.accept().unwrap();
        let (server_b, _) = listener_b.accept().unwrap();

        let readers = spawn_spray_readers(
            vec![server_a, server_b],
            LayerPermitPoolV1::experiment_v2(),
            FrameLimits::default(),
        );

        // K planes take pool permits (capacity 2); interleave so stream B
        // runs ahead, then finish with SEAL on stream 0.
        client_a
            .write_all(&frame_bytes(0, Role::Key, b"aa"))
            .unwrap();
        client_b
            .write_all(&frame_bytes(1, Role::Value, b"bb"))
            .unwrap();
        client_b
            .write_all(&frame_bytes(3, Role::Value, b"dd"))
            .unwrap();
        client_a
            .write_all(&frame_bytes(2, Role::Key, b"cc"))
            .unwrap();
        client_a
            .write_all(&frame_bytes(4, Role::Key, b"ee"))
            .unwrap();
        client_b
            .write_all(&frame_bytes(5, Role::Value, b"ff"))
            .unwrap();

        let mut mux = SequenceMuxV1::new(6);
        let mut order = Vec::new();
        while !mux.is_complete() {
            match readers.events.recv().unwrap() {
                SprayEvent::Layer(frame) => {
                    mux.accept(frame).unwrap();
                    while let Some(ready) = mux.pop_ready() {
                        // K planes carry the shared pool permit; V planes none.
                        assert_eq!(ready.header.role == Role::Key, ready.permit.is_some());
                        order.push((ready.header.sequence, ready.payload));
                    }
                }
                SprayEvent::Failed(error) => panic!("reader failed: {error}"),
                _ => panic!("unexpected non-layer event before completion"),
            }
        }
        assert_eq!(
            order,
            vec![
                (0, b"aa".to_vec()),
                (1, b"bb".to_vec()),
                (2, b"cc".to_vec()),
                (3, b"dd".to_vec()),
                (4, b"ee".to_vec()),
                (5, b"ff".to_vec()),
            ]
        );

        // Terminal SEAL ends stream 0 and hands its socket back via join.
        let seal = SealManifestV1 {
            artifact_sha256: "9".repeat(64),
            artifact_hmac_sha256: None,
            core: crate::SealCoreV1 {
                completed_unix_ms: 1,
                descriptor_chain_sha256: "a".repeat(64),
                frame_count: 6,
                payload_bytes: 12,
                payload_sha256: "b".repeat(64),
                prompt_token_ids: vec![1],
                protocol: crate::LIVE_HANDOFF_PROTOCOL_V1.into(),
                schema_version: LIVE_HANDOFF_SCHEMA_V1,
                strategy: crate::HandoffStrategyV1::ConsumerLastPromptToken,
                token_ids_sha256: "c".repeat(64),
                transfer_id: "spray-test".into(),
                canary: None,
            },
        };
        let mut seal_bytes = Vec::new();
        write_frame(
            &mut seal_bytes,
            &Frame::Seal(seal.clone()),
            FrameLimits::default(),
        )
        .unwrap();
        client_a.write_all(&seal_bytes).unwrap();
        match readers.events.recv().unwrap() {
            SprayEvent::Terminal { stream, frame } => {
                assert_eq!(stream, 0);
                assert_eq!(frame, Frame::Seal(seal));
            }
            _ => panic!("expected terminal event on stream 0"),
        }
        drop(readers.events);
        // Stream 1 is still mid-read; closing its peer lets the thread exit.
        drop(client_b);
        let mut threads = readers.threads.into_iter();
        let recovered = threads.next().unwrap().join().unwrap().unwrap();
        assert!(recovered.is_some(), "stream 0 socket must be handed back");
        assert!(threads.next().unwrap().join().unwrap().unwrap().is_none());
    }

    #[test]
    fn reader_threads_fail_closed_on_a_broken_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let readers = spawn_spray_readers(
            vec![server],
            LayerPermitPoolV1::experiment_v2(),
            FrameLimits::default(),
        );
        client.write_all(b"not a frame").unwrap();
        // Close the write side so the blocked header read fails with EOF
        // instead of waiting for bytes that never come.
        client.shutdown(std::net::Shutdown::Both).unwrap();
        match readers.events.recv().unwrap() {
            SprayEvent::Failed(_) => {}
            _ => panic!("garbage bytes must surface a failure event"),
        }
        drop(readers.events);
        drop(client);
        for thread in readers.threads {
            thread.join().unwrap().unwrap();
        }
    }
}
