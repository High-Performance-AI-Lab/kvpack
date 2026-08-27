use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::{HandoffError, Result};

const INTERRUPT_NONE: u8 = 0;
const INTERRUPT_CANCELLED: u8 = 1;
const INTERRUPT_DEADLINE: u8 = 2;

/// Cloneable cancellation handle for one receiver invocation.
///
/// Calling [`ReceiverInterruptV1::cancel`] before accept wakes the bounded
/// accept poll. Calling it after accept shuts down the registered TCP socket,
/// which unblocks TLS, frame reads, ACK writes, and a peer applying
/// backpressure. A handle is single-session: the cancellable receiver rejects
/// a handle that was already interrupted.
#[derive(Clone, Debug, Default)]
pub struct ReceiverInterruptV1 {
    inner: Arc<ReceiverInterruptInnerV1>,
}

#[derive(Debug, Default)]
struct ReceiverInterruptInnerV1 {
    reason: AtomicU8,
    // Every accepted session socket (one in the single-stream path, K in the
    // sprayed path); interruption shuts all of them down.
    sockets: Mutex<Vec<TcpStream>>,
}

impl ReceiverInterruptV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.interrupt(INTERRUPT_CANCELLED);
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.reason.load(Ordering::Acquire) == INTERRUPT_CANCELLED
    }

    pub fn is_interrupted(&self) -> bool {
        self.inner.reason.load(Ordering::Acquire) != INTERRUPT_NONE
    }

    fn deadline(&self) {
        self.interrupt(INTERRUPT_DEADLINE);
    }

    fn interrupt(&self, reason: u8) {
        let _ = self.inner.reason.compare_exchange(
            INTERRUPT_NONE,
            reason,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if let Ok(sockets) = self.inner.sockets.lock() {
            for socket in sockets.iter() {
                let _ = socket.shutdown(Shutdown::Both);
            }
        }
    }

    pub(super) fn register(&self, socket: &TcpStream) -> Result<()> {
        self.check()?;
        let cloned = socket.try_clone()?;
        let mut registered = self
            .inner
            .sockets
            .lock()
            .map_err(|_| HandoffError::Validation("receiver interrupt lock poisoned".into()))?;
        registered.push(cloned);
        drop(registered);
        self.check()
    }

    fn clear_sockets(&self) {
        if let Ok(mut sockets) = self.inner.sockets.lock() {
            sockets.clear();
        }
    }

    pub(super) fn check(&self) -> Result<()> {
        match self.inner.reason.load(Ordering::Acquire) {
            INTERRUPT_NONE => Ok(()),
            INTERRUPT_CANCELLED => Err(HandoffError::Cancelled),
            INTERRUPT_DEADLINE => Err(HandoffError::DeadlineExceeded),
            _ => Err(HandoffError::Validation(
                "receiver interrupt entered an invalid state".into(),
            )),
        }
    }

    pub(super) fn normalize<T>(&self, result: Result<T>) -> Result<T> {
        match self.check() {
            Ok(()) => result,
            Err(interrupted) => Err(interrupted),
        }
    }
}

pub(super) struct ReceiverDeadlineGuardV1 {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ReceiverDeadlineGuardV1 {
    pub(super) fn start(timeout: Duration, interrupt: ReceiverInterruptV1) -> Result<Self> {
        let (stop_tx, stop_rx) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("kvpack-receiver-deadline".into())
            .spawn(move || {
                if stop_rx.recv_timeout(timeout).is_err() {
                    interrupt.deadline();
                }
            })
            .map_err(HandoffError::Io)?;
        Ok(Self {
            stop: Some(stop_tx),
            thread: Some(thread),
        })
    }
}

impl Drop for ReceiverDeadlineGuardV1 {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(super) struct RegisteredSocketGuardV1 {
    pub(super) interrupt: ReceiverInterruptV1,
}

impl Drop for RegisteredSocketGuardV1 {
    fn drop(&mut self) {
        self.interrupt.clear_sockets();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;
    use std::net::TcpListener;

    #[test]
    fn cancellation_is_sticky_before_accept() {
        let interrupt = ReceiverInterruptV1::new();
        assert!(!interrupt.is_interrupted());
        interrupt.cancel();
        assert!(interrupt.is_cancelled());
        assert!(matches!(interrupt.check(), Err(HandoffError::Cancelled)));
        interrupt.deadline();
        assert!(matches!(interrupt.check(), Err(HandoffError::Cancelled)));
    }

    #[test]
    fn cancellation_shuts_down_an_accepted_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let (server, _) = listener.accept().unwrap();
        let interrupt = ReceiverInterruptV1::new();
        interrupt.register(&server).unwrap();
        interrupt.cancel();

        let mut byte = [0u8; 1];
        assert_eq!((&client).read(&mut byte).unwrap(), 0);
        assert!(matches!(interrupt.check(), Err(HandoffError::Cancelled)));
    }

    #[test]
    fn deadline_guard_records_deadline_and_joins() {
        let interrupt = ReceiverInterruptV1::new();
        {
            let _deadline =
                ReceiverDeadlineGuardV1::start(Duration::from_millis(5), interrupt.clone())
                    .unwrap();
            thread::sleep(Duration::from_millis(20));
        }
        assert!(matches!(
            interrupt.check(),
            Err(HandoffError::DeadlineExceeded)
        ));
    }
}
