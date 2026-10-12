use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rustls::{ServerConfig, ServerConnection, StreamOwned};
use serde::Serialize;

use super::config::ReceiverConfigV1;
use super::interrupt::{ReceiverDeadlineGuardV1, ReceiverInterruptV1, RegisteredSocketGuardV1};
use super::sink::ReceiverSinkV1;
use super::spray::{
    configured_spray_streams, spawn_spray_readers, SequenceMuxV1, SprayEvent, SPRAY_STREAMS_ENV,
};
use super::tls::{tls_server_config, verify_tls_peer};
use crate::{
    write_frame, AbortManifestV1, AckManifestV1, Frame, FrameHeader, FrameReader, HandoffError,
    LayerPermitPoolV1, Result, SealManifestV1, StreamingCoordinatorV1, ValidationLimits,
    VerifiedBundle, LIVE_HANDOFF_PROTOCOL_V1, LIVE_HANDOFF_SCHEMA_V1,
};

const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(10);
/// WS9 transport squeeze: explicit socket buffer sizing for the direct
/// 10 GbE link. The kernel default (~256 KiB autotuned) cannot fill the
/// bandwidth-delay product of the point-to-point path when the TLS record
/// loop stalls. Overridable for link experiments; 8 MiB matches the Python
/// producer's `KVPACK_SOCKET_BUFFER_BYTES` default.
const DEFAULT_SOCKET_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const SOCKET_BUFFER_BYTES_ENV: &str = "KVPACK_SOCKET_BUFFER_BYTES";

fn configured_socket_buffer_bytes() -> usize {
    std::env::var(SOCKET_BUFFER_BYTES_ENV)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(DEFAULT_SOCKET_BUFFER_BYTES)
}

/// Size the accepted socket's buffers explicitly. Linux doubles the value
/// internally, so the getsockopt readback is informational only; a sizing
/// failure is a hard error because silently running with default buffers
/// would misreport the measured configuration.
fn size_accepted_socket_buffers(socket: &std::net::TcpStream) -> Result<()> {
    let bytes = configured_socket_buffer_bytes();
    rustix::net::sockopt::set_socket_send_buffer_size(socket, bytes)
        .map_err(|error| HandoffError::Validation(format!("set SO_SNDBUF: {error}")))?;
    rustix::net::sockopt::set_socket_recv_buffer_size(socket, bytes)
        .map_err(|error| HandoffError::Validation(format!("set SO_RCVBUF: {error}")))?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverSessionStateV1 {
    AwaitBegin,
    Receiving,
    SealVerified,
    Published,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverReceiptV1 {
    pub artifact_sha256: String,
    pub cached_token_count: u32,
    pub first_frame_ns: u64,
    pub final_frame_ns: u64,
    pub layer_frames: u32,
    pub payload_bytes: u64,
    pub protocol: &'static str,
    pub publish_duration_ns: u64,
    pub schema_version: u32,
    pub state: ReceiverSessionStateV1,
    pub tls_setup_duration_ns: u64,
    pub total_duration_ns: u64,
    pub transfer_id: String,
}

/// Receive exactly one authenticated session and publish its original v1
/// bundle only after sink promotion has accepted the verified seal.
pub fn receive_one_v1(
    config: ReceiverConfigV1,
    sink: &mut impl ReceiverSinkV1,
) -> Result<ReceiverReceiptV1> {
    receive_one_v1_with_ready(config, sink, || {})
}

/// Variant used by resident qualification workers to publish their armed
/// receipt only after the exact direct-link listener is bound.
pub fn receive_one_v1_with_ready(
    config: ReceiverConfigV1,
    sink: &mut impl ReceiverSinkV1,
    ready: impl FnOnce(),
) -> Result<ReceiverReceiptV1> {
    receive_one_v1_cancellable_with_ready(config, sink, ReceiverInterruptV1::new(), ready)
}

/// Deadline-aware receiver variant used by product workers.
///
/// This is additive to the frozen qualification entry points above. The
/// supplied handle may be cancelled from another thread and is also driven by
/// the configured absolute session deadline.
pub fn receive_one_v1_cancellable(
    config: ReceiverConfigV1,
    sink: &mut impl ReceiverSinkV1,
    interrupt: ReceiverInterruptV1,
) -> Result<ReceiverReceiptV1> {
    receive_one_v1_cancellable_with_ready(config, sink, interrupt, || {})
}

/// Cancellable receiver variant with a listener-ready callback.
pub fn receive_one_v1_cancellable_with_ready(
    config: ReceiverConfigV1,
    sink: &mut impl ReceiverSinkV1,
    interrupt: ReceiverInterruptV1,
    ready: impl FnOnce(),
) -> Result<ReceiverReceiptV1> {
    let result = receive_one_v1_cancellable_inner(config, sink, interrupt, ready);
    if let Err(error) = &result {
        eprintln!("[receiver] session failed: {error}");
    }
    result
}

fn receive_one_v1_cancellable_inner(
    config: ReceiverConfigV1,
    sink: &mut impl ReceiverSinkV1,
    interrupt: ReceiverInterruptV1,
    ready: impl FnOnce(),
) -> Result<ReceiverReceiptV1> {
    config.validate()?;
    interrupt.check()?;
    let origin = Instant::now();
    let _deadline = ReceiverDeadlineGuardV1::start(config.timeout, interrupt.clone())?;
    let tls = tls_server_config(&config)?;
    let listener = TcpListener::bind(config.bind)?;
    listener.set_nonblocking(true)?;
    ready();
    // N5 containment: pre-authentication connection failures (wrong source
    // IP, failed TLS handshake, unpinned peer certificate) reject that one
    // connection, never the armed receive; the accept loop resumes until a
    // fully pinned and validated connection arrives or the deadline guard
    // fires.
    let (mut stream, _registered_socket) =
        accept_authenticated_connection(&listener, &interrupt, |socket, peer| {
            prepare_connection(socket, peer, &config, &tls, &interrupt)
        })?;
    // WS9 TENT-lite spraying (opt-in): KVPACK_HANDOFF_STREAMS > 1 accepts K
    // pinned mTLS connections and reassembles by frame sequence. With the env
    // unset the code below is byte-identical to the pre-spray receiver.
    let streams = configured_spray_streams();
    if streams > 1 {
        return receive_sprayed_v1(
            &config,
            sink,
            &interrupt,
            origin,
            &tls,
            &listener,
            stream,
            streams,
            _registered_socket,
        );
    }
    let tls_setup_duration_ns = elapsed_ns(origin);

    let permit_pool = LayerPermitPoolV1::experiment_v2();
    let mut state = ReceiverSessionStateV1::AwaitBegin;
    let mut coordinator: Option<StreamingCoordinatorV1> = None;
    let mut active_transfer_id: Option<String> = None;
    let receive_result = (|| -> Result<_> {
        interrupt.check()?;
        let mut reader = FrameReader::new(&mut stream, config.frame_limits);
        let header = reader.read_header()?.clone();
        interrupt.check()?;
        let first_frame_ns = elapsed_ns(origin);
        let FrameHeader::Begin(_) = header else {
            return Err(HandoffError::Validation(
                "first live handoff frame must be BEGIN".into(),
            ));
        };
        let Frame::Begin(begin) = reader.read_payload()? else {
            unreachable!("decoded BEGIN header must produce BEGIN frame")
        };
        let begin = *begin;
        // A one-shot receiver can be armed before the producer loads its
        // model. Validate producer creation/deadline timestamps against the
        // instant BEGIN is actually received, not listener startup.
        let session_limits = validation_limits_at(&config.validation_limits, now_unix_ms()?);
        begin.validate(&session_limits)?;
        config.begin.validate_for(&begin)?;
        active_transfer_id = Some(begin.transfer_id.clone());
        sink.begin(&begin)?;
        coordinator = Some(StreamingCoordinatorV1::create(
            &config.output,
            begin.clone(),
            session_limits.clone(),
        )?);
        state = ReceiverSessionStateV1::Receiving;

        for sequence in 0..begin.expected_layer_frames {
            interrupt.check()?;
            let header = reader.read_header()?.clone();
            let permit = match &header {
                FrameHeader::Layer(layer) if layer.sequence == sequence => {
                    coordinator
                        .as_ref()
                        .expect("coordinator created before receiving")
                        .validate_next_header(layer)?;
                    if layer.role == crate::TensorRoleV1::Key {
                        Some(permit_pool.acquire()?)
                    } else {
                        None
                    }
                }
                FrameHeader::Abort(abort) => {
                    return Err(HandoffError::Validation(format!(
                        "producer aborted handoff with code {}",
                        abort.code
                    )))
                }
                _ => {
                    return Err(HandoffError::Validation(format!(
                        "expected ordered layer frame {sequence}"
                    )))
                }
            };
            let Frame::Layer(header, payload) = reader.read_payload()? else {
                unreachable!("decoded layer header must produce layer frame")
            };
            interrupt.check()?;
            if let Some(ready) = coordinator
                .as_mut()
                .expect("coordinator created before receiving")
                .ingest_plane(header, payload, permit)?
            {
                sink.layer_ready(ready)?;
            }
        }
        let terminal = reader.read_header()?.clone();
        interrupt.check()?;
        let final_frame_ns = elapsed_ns(origin);
        let FrameHeader::Seal(_) = terminal else {
            return Err(HandoffError::Validation(
                "expected terminal SEAL after all layer frames".into(),
            ));
        };
        let Frame::Seal(seal) = reader.read_payload()? else {
            unreachable!("decoded SEAL header must produce SEAL frame")
        };
        let seal_verify_started = Instant::now();
        let verified_seal = coordinator
            .as_mut()
            .expect("coordinator created before seal")
            .verify_and_prepare_seal(seal.clone())?;
        receiver_timing!(
            "[receiver-timing] verify_and_prepare_seal {:?}",
            seal_verify_started.elapsed()
        );
        interrupt.check()?;
        // F1: when the receiver arms a tenant MAC key, the artifact must be
        // keyed-authenticated, not just integrity-hashed — a bundle that
        // reached the engine outside the authenticated transport (local
        // file, locality hop) is forgeable without the key. The begin must
        // declare an `hmac_key_id` and the seal's tag must verify under the
        // armed key, else the session aborts before publication.
        if let Some(key) = config.mac_key.as_ref() {
            if begin.hmac_key_id.is_none() {
                return Err(HandoffError::Validation(
                    "receiver armed for artifact HMAC but begin declares no hmac_key_id".into(),
                ));
            }
            coordinator
                .as_ref()
                .expect("coordinator created before seal")
                .authenticate_seal_hmac(&verified_seal, key)?;
        }
        state = ReceiverSessionStateV1::SealVerified;

        // Ferrite promotion and private first-token production are allowed at
        // this point and are deliberately independent of bundle publication.
        let sink_started = Instant::now();
        sink.seal_verified(&verified_seal)?;
        receiver_timing!(
            "[receiver-timing] sink.seal_verified {:?}",
            sink_started.elapsed()
        );
        interrupt.check()?;
        let publish_started = Instant::now();
        let committed = coordinator
            .as_mut()
            .expect("coordinator created before publication")
            .publish()?;
        receiver_timing!(
            "[receiver-timing] coordinator.publish {:?}",
            publish_started.elapsed()
        );
        let reopen_started = Instant::now();
        let verified = VerifiedBundle::open(&committed, &session_limits)?;
        receiver_timing!(
            "[receiver-timing] verified_bundle_reopen {:?}",
            reopen_started.elapsed()
        );
        let publish_duration_ns = duration_ns(publish_started.elapsed());
        let ack = AckManifestV1::committed(&begin, &seal);
        let stream = reader.into_inner()?;
        write_frame(stream, &Frame::Ack(ack), config.frame_limits)?;
        interrupt.check()?;
        state = ReceiverSessionStateV1::Published;
        Ok(ReceiverReceiptV1 {
            artifact_sha256: verified.seal().artifact_sha256.clone(),
            cached_token_count: begin.cached_token_count,
            first_frame_ns,
            final_frame_ns,
            layer_frames: begin.expected_layer_frames,
            payload_bytes: begin.expected_payload_bytes,
            protocol: LIVE_HANDOFF_PROTOCOL_V1,
            publish_duration_ns,
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            state,
            tls_setup_duration_ns,
            total_duration_ns: elapsed_ns(origin),
            transfer_id: begin.transfer_id,
        })
    })();

    match interrupt.normalize(receive_result) {
        Ok(receipt) => Ok(receipt),
        Err(error) => {
            state = ReceiverSessionStateV1::Aborted;
            let _ = state;
            sink.abort();
            if let Some(coordinator) = coordinator.as_mut() {
                let _ = coordinator.abort();
            }
            if let Some(transfer_id) = active_transfer_id {
                let abort = AbortManifestV1 {
                    code: "receiver_rejected".into(),
                    protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
                    schema_version: LIVE_HANDOFF_SCHEMA_V1,
                    transfer_id,
                };
                let _ = write_frame(&mut stream, &Frame::Abort(abort), config.frame_limits);
            }
            Err(error)
        }
    }
}

fn elapsed_ns(origin: Instant) -> u64 {
    duration_ns(origin.elapsed())
}

/// Interrupt-aware accept poll shared by the single-stream path and the
/// sprayed path's bounded wait for auxiliary connections (bounded by the
/// session deadline guard, which interrupts this loop at `config.timeout`).
fn accept_connection(
    listener: &TcpListener,
    interrupt: &ReceiverInterruptV1,
) -> Result<(TcpStream, SocketAddr)> {
    loop {
        interrupt.check()?;
        match listener.accept() {
            Ok(accepted) => return Ok(accepted),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(error) => return Err(HandoffError::Io(error)),
        }
    }
}

/// Per-connection containment (N5): accept connections until one passes
/// every per-connection policy in `prepare` (peer-IP pinning, interface
/// check, socket sizing, TLS 1.3 handshake, peer certificate pinning).
/// A rejected connection is logged, deregistered from the interrupt set,
/// and dropped; the loop resumes accepting until a connection is fully
/// prepared or the deadline guard (or a cancel) interrupts it. Only a
/// successfully prepared connection is returned — consuming the one-shot
/// receive — together with the guard that keeps it interrupt-registered.
fn accept_authenticated_connection<T>(
    listener: &TcpListener,
    interrupt: &ReceiverInterruptV1,
    mut prepare: impl FnMut(TcpStream, SocketAddr) -> Result<T>,
) -> Result<(T, RegisteredSocketGuardV1)> {
    loop {
        let (socket, peer) = accept_connection(listener, interrupt)?;
        socket.set_nonblocking(false)?;
        interrupt.register(&socket)?;
        let attempt = RegisteredSocketGuardV1 {
            interrupt: interrupt.clone(),
        };
        match prepare(socket, peer) {
            Ok(prepared) => return Ok((prepared, attempt)),
            Err(error) => {
                drop(attempt);
                // A deadline/cancel that fired while a connection was being
                // prepared reports as the interrupt, not as another
                // rejected connection.
                interrupt.check()?;
                eprintln!("[receiver] rejected connection from {peer}: {error}; resuming accept");
            }
        }
    }
}

/// Apply every per-connection policy exactly as the single-stream path does:
/// peer-IP and direct-link interface pinning, timeouts, TCP_NODELAY, explicit
/// socket buffers, the TLS 1.3 handshake, and peer certificate pinning.
fn prepare_connection(
    socket: TcpStream,
    peer: SocketAddr,
    config: &ReceiverConfigV1,
    tls: &Arc<ServerConfig>,
    interrupt: &ReceiverInterruptV1,
) -> Result<StreamOwned<ServerConnection, TcpStream>> {
    if peer.ip() != config.expected_peer_ip {
        return Err(HandoffError::Validation(format!(
            "refusing live handoff peer {}: expected {}",
            peer.ip(),
            config.expected_peer_ip
        )));
    }
    if socket.local_addr()?.ip() != config.bind.ip() {
        return Err(HandoffError::Validation(
            "accepted socket left the configured direct-link interface".into(),
        ));
    }
    socket.set_read_timeout(Some(config.timeout))?;
    socket.set_write_timeout(Some(config.timeout))?;
    socket.set_nodelay(true)?;
    size_accepted_socket_buffers(&socket)?;
    let connection = ServerConnection::new(Arc::clone(tls))
        .map_err(|error| HandoffError::Validation(format!("create TLS session: {error}")))?;
    let mut stream = StreamOwned::new(connection, socket);
    while stream.conn.is_handshaking() {
        let handshake = stream
            .conn
            .complete_io(&mut stream.sock)
            .map(|_| ())
            .map_err(HandoffError::Io);
        interrupt.normalize(handshake)?;
    }
    interrupt.check()?;
    verify_tls_peer(&stream, &config.expected_client_cert_sha256)?;
    Ok(stream)
}

/// WS9 TENT-lite sprayed session (KVPACK_HANDOFF_STREAMS > 1).
///
/// Threading model: the already-verified stream-0 connection plus `streams-1`
/// freshly accepted and identically pinned aux connections each move into a
/// blocking reader thread (same FrameReader split flow as the single-stream
/// path). Layer frames arrive on the session thread through one mpsc channel
/// and are reassembled into strict sequence order by `SequenceMuxV1` before
/// `StreamingCoordinatorV1::ingest_plane` runs exactly as today. K-plane
/// permits come from the shared capacity-2 pool BEFORE the payload read, so
/// the canonical-memory bound is unchanged; V planes never block, and pairs
/// complete in order on this thread, so a reader parked in acquire always
/// wakes. Any reader error, protocol violation (non-layer frame on an aux
/// stream, duplicate/out-of-range sequence), or channel disconnect aborts
/// fail-closed through the same cleanup path as the single-stream receiver.
/// BEGIN/SEAL/ABORT travel on stream 0 only; its reader returns the owned
/// connection at the terminal frame so this thread can write the ACK.
#[allow(clippy::too_many_arguments)]
fn receive_sprayed_v1(
    config: &ReceiverConfigV1,
    sink: &mut impl ReceiverSinkV1,
    interrupt: &ReceiverInterruptV1,
    origin: Instant,
    tls: &Arc<ServerConfig>,
    listener: &TcpListener,
    first: StreamOwned<ServerConnection, TcpStream>,
    streams: usize,
    _registered_socket: RegisteredSocketGuardV1,
) -> Result<ReceiverReceiptV1> {
    eprintln!("[receiver] spray active: {streams} streams ({SPRAY_STREAMS_ENV}={streams})");
    let mut connections = Vec::with_capacity(streams);
    connections.push(first);
    for _ in 1..streams {
        let (socket, peer) = accept_connection(listener, interrupt)?;
        socket.set_nonblocking(false)?;
        interrupt.register(&socket)?;
        connections.push(prepare_connection(socket, peer, config, tls, interrupt)?);
    }
    let tls_setup_duration_ns = elapsed_ns(origin);
    // Clones kept so every reader thread can be forced out of a blocking read
    // on both the success and abort cleanup paths.
    let shutdown_handles = connections
        .iter()
        .map(|stream| stream.sock.try_clone().map_err(HandoffError::Io))
        .collect::<Result<Vec<_>>>()?;

    let permit_pool = LayerPermitPoolV1::experiment_v2();
    let mut state = ReceiverSessionStateV1::AwaitBegin;
    let mut coordinator: Option<StreamingCoordinatorV1> = None;
    let mut active_transfer_id: Option<String> = None;
    let readers = spawn_spray_readers(connections, permit_pool, config.frame_limits);
    let events = readers.events;
    let mut threads = readers.threads;
    let receive_result = (|| -> Result<_> {
        interrupt.check()?;
        let begin = match events.recv() {
            Ok(SprayEvent::Begin(begin)) => *begin,
            Ok(SprayEvent::Failed(error)) => return Err(error),
            Ok(_) => {
                return Err(HandoffError::Validation(
                    "first sprayed frame must be BEGIN on stream 0".into(),
                ))
            }
            Err(_) => {
                return Err(HandoffError::Validation(
                    "spray readers stopped before BEGIN".into(),
                ))
            }
        };
        let first_frame_ns = elapsed_ns(origin);
        let session_limits = validation_limits_at(&config.validation_limits, now_unix_ms()?);
        begin.validate(&session_limits)?;
        config.begin.validate_for(&begin)?;
        active_transfer_id = Some(begin.transfer_id.clone());
        sink.begin(&begin)?;
        coordinator = Some(StreamingCoordinatorV1::create(
            &config.output,
            begin.clone(),
            session_limits.clone(),
        )?);
        state = ReceiverSessionStateV1::Receiving;

        let mut mux = SequenceMuxV1::new(begin.expected_layer_frames);
        let mut seal: Option<SealManifestV1> = None;
        let mut final_frame_ns = first_frame_ns;
        while !mux.is_complete() || seal.is_none() {
            interrupt.check()?;
            let event = events.recv().map_err(|_| {
                HandoffError::Validation(
                    "spray readers stopped before the session completed".into(),
                )
            })?;
            match event {
                SprayEvent::Begin(_) => {
                    return Err(HandoffError::Validation(
                        "duplicate BEGIN on a sprayed session".into(),
                    ))
                }
                SprayEvent::Layer(frame) => {
                    mux.accept(frame)?;
                    while let Some(ready) = mux.pop_ready() {
                        coordinator
                            .as_ref()
                            .expect("coordinator created before receiving")
                            .validate_next_header(&ready.header)?;
                        interrupt.check()?;
                        if let Some(ready) = coordinator
                            .as_mut()
                            .expect("coordinator created before receiving")
                            .ingest_plane(ready.header, ready.payload, ready.permit)?
                        {
                            sink.layer_ready(ready)?;
                        }
                    }
                }
                SprayEvent::Terminal { stream: 0, frame } => match frame {
                    Frame::Seal(value) => {
                        final_frame_ns = elapsed_ns(origin);
                        seal = Some(value);
                    }
                    Frame::Abort(abort) => {
                        return Err(HandoffError::Validation(format!(
                            "producer aborted handoff with code {}",
                            abort.code
                        )))
                    }
                    _ => {
                        return Err(HandoffError::Validation(
                            "unexpected terminal frame on spray stream 0".into(),
                        ))
                    }
                },
                SprayEvent::Terminal { .. } => {
                    return Err(HandoffError::Validation(
                        "non-layer frame on an auxiliary spray stream".into(),
                    ))
                }
                SprayEvent::Failed(error) => return Err(error),
            }
        }
        let seal = seal.expect("seal checked by the loop condition");
        let seal_verify_started = Instant::now();
        let verified_seal = coordinator
            .as_mut()
            .expect("coordinator created before seal")
            .verify_and_prepare_seal(seal.clone())?;
        receiver_timing!(
            "[receiver-timing] verify_and_prepare_seal {:?}",
            seal_verify_started.elapsed()
        );
        interrupt.check()?;
        // F1: when the receiver arms a tenant MAC key, the artifact must be
        // keyed-authenticated, not just integrity-hashed — a bundle that
        // reached the engine outside the authenticated transport (local
        // file, locality hop) is forgeable without the key. The begin must
        // declare an `hmac_key_id` and the seal's tag must verify under the
        // armed key, else the session aborts before publication.
        if let Some(key) = config.mac_key.as_ref() {
            if begin.hmac_key_id.is_none() {
                return Err(HandoffError::Validation(
                    "receiver armed for artifact HMAC but begin declares no hmac_key_id".into(),
                ));
            }
            coordinator
                .as_ref()
                .expect("coordinator created before seal")
                .authenticate_seal_hmac(&verified_seal, key)?;
        }
        state = ReceiverSessionStateV1::SealVerified;

        let sink_started = Instant::now();
        sink.seal_verified(&verified_seal)?;
        receiver_timing!(
            "[receiver-timing] sink.seal_verified {:?}",
            sink_started.elapsed()
        );
        interrupt.check()?;
        let publish_started = Instant::now();
        let committed = coordinator
            .as_mut()
            .expect("coordinator created before publication")
            .publish()?;
        receiver_timing!(
            "[receiver-timing] coordinator.publish {:?}",
            publish_started.elapsed()
        );
        let reopen_started = Instant::now();
        let verified = VerifiedBundle::open(&committed, &session_limits)?;
        receiver_timing!(
            "[receiver-timing] verified_bundle_reopen {:?}",
            reopen_started.elapsed()
        );
        let publish_duration_ns = duration_ns(publish_started.elapsed());
        let ack = AckManifestV1::committed(&begin, &seal);
        // Stream 0's reader observed the terminal SEAL and handed the
        // connection back through its join handle.
        let mut stream = match threads.remove(0).join() {
            Ok(Ok(Some(stream))) => stream,
            _ => {
                return Err(HandoffError::Validation(
                    "spray stream 0 reader did not return its connection".into(),
                ))
            }
        };
        write_frame(&mut stream, &Frame::Ack(ack), config.frame_limits)?;
        interrupt.check()?;
        state = ReceiverSessionStateV1::Published;
        Ok(ReceiverReceiptV1 {
            artifact_sha256: verified.seal().artifact_sha256.clone(),
            cached_token_count: begin.cached_token_count,
            first_frame_ns,
            final_frame_ns,
            layer_frames: begin.expected_layer_frames,
            payload_bytes: begin.expected_payload_bytes,
            protocol: LIVE_HANDOFF_PROTOCOL_V1,
            publish_duration_ns,
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            state,
            tls_setup_duration_ns,
            total_duration_ns: elapsed_ns(origin),
            transfer_id: begin.transfer_id,
        })
    })();

    match interrupt.normalize(receive_result) {
        Ok(receipt) => {
            // The producer closes its aux connections after the ACK; force
            // the issue so aux reader threads exit deterministically, then
            // join everything. Stream 0 stays up for the peer's close().
            for handle in &shutdown_handles[1..] {
                let _ = handle.shutdown(Shutdown::Both);
            }
            drop(events);
            for thread in threads {
                let _ = thread.join();
            }
            Ok(receipt)
        }
        Err(error) => {
            for handle in &shutdown_handles {
                let _ = handle.shutdown(Shutdown::Both);
            }
            // Dropping the channel receiver makes any reader parked in send
            // (or woken from a permit acquire) exit instead of leaking.
            drop(events);
            let mut recovered: Option<StreamOwned<ServerConnection, TcpStream>> = None;
            for (index, thread) in threads.into_iter().enumerate() {
                if index == 0 {
                    if let Ok(Ok(Some(stream))) = thread.join() {
                        recovered = Some(stream);
                    }
                } else {
                    let _ = thread.join();
                }
            }
            state = ReceiverSessionStateV1::Aborted;
            let _ = state;
            sink.abort();
            if let Some(coordinator) = coordinator.as_mut() {
                let _ = coordinator.abort();
            }
            if let (Some(transfer_id), Some(stream)) = (active_transfer_id, recovered.as_mut()) {
                let abort = AbortManifestV1 {
                    code: "receiver_rejected".into(),
                    protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
                    schema_version: LIVE_HANDOFF_SCHEMA_V1,
                    transfer_id,
                };
                let _ = write_frame(stream, &Frame::Abort(abort), config.frame_limits);
            }
            Err(error)
        }
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn now_unix_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| HandoffError::Validation("system clock precedes Unix epoch".into()))?
        .as_millis()
        .try_into()
        .map_err(|_| HandoffError::Validation("Unix time exceeds u64".into()))
}

fn validation_limits_at(limits: &ValidationLimits, now_unix_ms: u64) -> ValidationLimits {
    let mut session_limits = limits.clone();
    session_limits.now_unix_ms = now_unix_ms;
    session_limits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BeginManifestV1, EndpointIdentityV1, ExactIdentityV1, GeometryV1, HandoffStrategyV1,
        PrecisionV1, PORTABLE_KV_ABI_V1,
    };

    #[test]
    fn monotonic_receipt_clock_is_u64_nanoseconds() {
        let origin = Instant::now();
        assert!(elapsed_ns(origin) < u64::MAX);
        assert!(now_unix_ms().unwrap() > 0);
    }

    #[test]
    fn accepted_socket_buffers_are_sized_explicitly() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (accepted, _) = listener.accept().unwrap();
        size_accepted_socket_buffers(&accepted).unwrap();
        // Linux reports double the requested value; only a lower bound holds
        // portably across the lab's Linux/macOS pair.
        let recv = rustix::net::sockopt::socket_recv_buffer_size(&accepted).unwrap();
        let send = rustix::net::sockopt::socket_send_buffer_size(&accepted).unwrap();
        assert!(recv >= DEFAULT_SOCKET_BUFFER_BYTES, "recv buffer {recv}");
        assert!(send >= DEFAULT_SOCKET_BUFFER_BYTES, "send buffer {send}");
        drop(client);
    }

    #[test]
    fn rejected_first_connection_does_not_consume_the_one_shot() {
        // N5: a wrong source IP / failed TLS handshake / wrong-cert peer on
        // the FIRST connection must not kill the armed receive. The policy
        // closure stands in for prepare_connection: it rejects the first
        // connection and accepts the second; the loop must return exactly
        // the validated connection.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let bad_client = std::net::TcpStream::connect(address).unwrap();
        let good_client = std::net::TcpStream::connect(address).unwrap();
        let interrupt = ReceiverInterruptV1::new();
        let attempts = std::cell::Cell::new(0u32);
        let (stream, _guard) =
            accept_authenticated_connection(&listener, &interrupt, |socket, _| {
                attempts.set(attempts.get() + 1);
                if attempts.get() == 1 {
                    return Err(HandoffError::Validation(
                        "refusing live handoff peer: unexpected source".into(),
                    ));
                }
                Ok(socket)
            })
            .unwrap();
        assert_eq!(attempts.get(), 2);
        // The accepted connection is the second one, peer for peer.
        assert_eq!(
            stream.peer_addr().unwrap(),
            good_client.local_addr().unwrap()
        );
        drop(bad_client);
        drop(good_client);
    }

    #[test]
    fn persistent_bad_connections_fail_closed_at_the_deadline_guard() {
        // N5: containment is not an infinite pass — with every connection
        // failing pre-authentication, the session still dies on the
        // deadline guard, not on the first failure and not never.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let interrupt = ReceiverInterruptV1::new();
        let _deadline =
            ReceiverDeadlineGuardV1::start(Duration::from_millis(50), interrupt.clone()).unwrap();
        let attempts = std::cell::Cell::new(0u32);
        let bad_client = std::net::TcpStream::connect(address).unwrap();
        let result = accept_authenticated_connection(
            &listener,
            &interrupt,
            |_: TcpStream, _| -> Result<TcpStream> {
                attempts.set(attempts.get() + 1);
                Err(HandoffError::Validation("TLS handshake failed".into()))
            },
        );
        assert!(matches!(result, Err(HandoffError::DeadlineExceeded)));
        // The first failure was contained: accept resumed and took more
        // connections before the deadline fired.
        assert!(attempts.get() >= 1);
        drop(bad_client);

        // A cancelled handle reports as cancellation, same containment.
        let interrupt = ReceiverInterruptV1::new();
        interrupt.cancel();
        let result = accept_authenticated_connection(
            &listener,
            &interrupt,
            |_: TcpStream, _| -> Result<TcpStream> {
                Err(HandoffError::Validation("TLS handshake failed".into()))
            },
        );
        assert!(matches!(result, Err(HandoffError::Cancelled)));
    }

    #[test]
    fn begin_timestamp_is_validated_at_receipt_not_listener_start() {
        let configured = ValidationLimits {
            now_unix_ms: 1,
            ..ValidationLimits::default()
        };
        let begin_received_unix_ms = 90_000;
        let begin = BeginManifestV1 {
            cached_token_count: 1,
            created_unix_ms: begin_received_unix_ms,
            deadline_unix_ms: begin_received_unix_ms + 30_000,
            endpoints: EndpointIdentityV1 {
                consumer_engine_abi: "ferrite".into(),
                consumer_node: "mac".into(),
                producer_engine_abi: "vllm".into(),
                producer_node: "spark".into(),
                trust_domain: "lab".into(),
            },
            expected_layer_frames: 48,
            expected_payload_bytes: 12_288,
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
            protocol: LIVE_HANDOFF_PROTOCOL_V1.into(),
            schema_version: LIVE_HANDOFF_SCHEMA_V1,
            strategy: HandoffStrategyV1::ConsumerLastPromptToken,
            token_ids_sha256: "7".repeat(64),
            transfer_id: "8".repeat(64),
            layout_table: Vec::new(),
            schedule: None,
            hmac_key_id: None,
        };

        assert!(begin.validate(&configured).is_err());
        let received = validation_limits_at(&configured, begin_received_unix_ms);
        assert!(begin.validate(&received).is_ok());
    }
}
