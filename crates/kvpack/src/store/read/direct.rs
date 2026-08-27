//! Runtime-gated, cache-bypassed chunk-file read fast path (WS9 item 5).
//!
//! Setting `KVPACK_DIRECT_READ=1` routes whole-chunk reads through a direct
//! path that keeps the page cache out of the way: `O_DIRECT` with an aligned
//! buffer and `preadv2` on Linux, and `fcntl(F_NOCACHE)` with a vectored
//! `preadv` on macOS (mirroring `pod/fnocache_read.c`).  Any alignment,
//! platform, or I/O condition the fast path cannot serve returns `None`, and
//! the caller falls back to the normal cached read — a restore never fails
//! because of this path.  With the gate off the behavior is byte-identical to
//! the cached path.

// The fast path is a thin, audited wrapper over the raw `libc` syscalls
// (open/fcntl/fstat/preadv/preadv2/posix_memalign) that the safe `rustix`
// layer does not expose (O_DIRECT, F_NOCACHE, preadv2).  Every unsafe call is
// confined to the `platform` submodules; all failure modes degrade to `None`.
#![allow(unsafe_code)]

use std::path::Path;

/// Buffer/offset/length alignment required by the bypass path.  O_DIRECT
/// typically requires 512- or 4096-byte alignment; 4096 covers both, and
/// chunk lengths that are not a multiple of it fall back to the cached path
/// instead of attempting partial-aligned reads.
pub const DIRECT_READ_ALIGNMENT: usize = 4096;

/// Per-syscall read window, mirroring `pod/fnocache_read.c`.
const WINDOW_BYTES: usize = 4 * 1024 * 1024;

/// Whether the runtime gate (`KVPACK_DIRECT_READ=1`) is enabled.
pub fn direct_read_enabled() -> bool {
    std::env::var_os("KVPACK_DIRECT_READ").is_some_and(|value| value == "1")
}

/// Read one whole chunk file bypassing the page cache.  Returns `None` on any
/// condition the fast path cannot serve — zero or misaligned length,
/// unsupported platform or filesystem, open/fcntl/read failure, or a short
/// read — so the caller can fall back to the normal cached read.
pub fn read_chunk_bypass(path: &Path, expected_bytes: usize) -> Option<Vec<u8>> {
    if expected_bytes == 0 || expected_bytes % DIRECT_READ_ALIGNMENT != 0 {
        return None;
    }
    platform::read_bypass(path, expected_bytes)
}

/// Gated dispatch used by the pinned-chunk read path: with the gate off this
/// always returns `None`, so the normal cached read runs exactly as before.
pub(crate) fn maybe_read_chunk_bypass(
    path: &Path,
    expected_bytes: usize,
    enabled: bool,
) -> Option<Vec<u8>> {
    if !enabled {
        return None;
    }
    read_chunk_bypass(path, expected_bytes)
}

#[cfg(target_os = "macos")]
mod platform {
    use super::WINDOW_BYTES;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    struct FdGuard(libc::c_int);

    impl Drop for FdGuard {
        fn drop(&mut self) {
            unsafe { libc::close(self.0) };
        }
    }

    /// macOS has no O_DIRECT; F_NOCACHE is the equivalent — it keeps the
    /// unified buffer cache out of the way so large restores do not evict the
    /// working set.  F_NOCACHE needs no buffer alignment, but the read region
    /// must still be a whole number of alignment units (checked by the caller)
    /// so the fallback contract stays uniform across platforms.
    pub(super) fn read_bypass(path: &Path, expected_bytes: usize) -> Option<Vec<u8>> {
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return None;
        }
        let fd = FdGuard(fd);
        // Bypass the unified buffer cache; if the filesystem refuses, fall back.
        if unsafe { libc::fcntl(fd.0, libc::F_NOCACHE, 1) } != 0 {
            return None;
        }
        let mut status: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.0, &mut status) } != 0
            || status.st_mode & libc::S_IFMT != libc::S_IFREG
            || status.st_size != expected_bytes as libc::off_t
        {
            return None;
        }
        let mut bytes = vec![0u8; expected_bytes];
        let mut done = 0usize;
        while done < expected_bytes {
            let window = (expected_bytes - done).min(WINDOW_BYTES);
            let iov = libc::iovec {
                iov_base: bytes[done..].as_mut_ptr().cast::<libc::c_void>(),
                iov_len: window,
            };
            let n = unsafe { libc::preadv(fd.0, &iov, 1, done as libc::off_t) };
            if n <= 0 {
                return None;
            }
            done += n as usize;
        }
        Some(bytes)
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{DIRECT_READ_ALIGNMENT, WINDOW_BYTES};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    struct FdGuard(libc::c_int);

    impl Drop for FdGuard {
        fn drop(&mut self) {
            unsafe { libc::close(self.0) };
        }
    }

    struct AlignedGuard(*mut libc::c_void);

    impl Drop for AlignedGuard {
        fn drop(&mut self) {
            unsafe { libc::free(self.0) };
        }
    }

    /// O_DIRECT with a `posix_memalign`ed buffer and vectored `preadv2`.
    /// Buffer, offset, and length are all multiples of `DIRECT_READ_ALIGNMENT`
    /// (the length precondition is checked by the caller); any open or read
    /// failure — including filesystems without O_DIRECT support — falls back.
    pub(super) fn read_bypass(path: &Path, expected_bytes: usize) -> Option<Vec<u8>> {
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
        let fd = unsafe {
            libc::open(
                c_path.as_ptr(),
                libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_DIRECT,
            )
        };
        if fd < 0 {
            return None;
        }
        let fd = FdGuard(fd);
        let mut status: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.0, &mut status) } != 0
            || status.st_mode & libc::S_IFMT != libc::S_IFREG
            || status.st_size != expected_bytes as libc::off_t
        {
            return None;
        }
        let mut buffer: *mut libc::c_void = std::ptr::null_mut();
        if unsafe { libc::posix_memalign(&mut buffer, DIRECT_READ_ALIGNMENT, expected_bytes) } != 0
            || buffer.is_null()
        {
            return None;
        }
        let buffer = AlignedGuard(buffer);
        let mut done = 0usize;
        while done < expected_bytes {
            let window = (expected_bytes - done).min(WINDOW_BYTES);
            let iov = libc::iovec {
                iov_base: unsafe { (buffer.0 as *mut u8).add(done) }.cast::<libc::c_void>(),
                iov_len: window,
            };
            let n = unsafe { libc::preadv2(fd.0, &iov, 1, done as libc::off_t, 0) };
            if n <= 0 {
                return None;
            }
            done += n as usize;
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(buffer.0 as *const u8, expected_bytes) }.to_vec();
        Some(bytes)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    use std::path::Path;

    pub(super) fn read_bypass(_path: &Path, _expected_bytes: usize) -> Option<Vec<u8>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture(bytes: usize) -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chunk.kvchunk");
        let mut content = Vec::with_capacity(bytes);
        let mut state = 0x243f_6a88_85a3_08d3u64;
        while content.len() < bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            content.extend_from_slice(&state.to_le_bytes());
        }
        content.truncate(bytes);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&content).unwrap();
        file.sync_all().unwrap();
        (dir, path, content)
    }

    #[test]
    fn bypass_bytes_equal_cached_bytes() {
        let (_dir, path, content) = fixture(3 * DIRECT_READ_ALIGNMENT);
        if let Some(bypass) = read_chunk_bypass(&path, content.len()) {
            // Some filesystems (tmpfs, overlayfs) refuse the fast path; that is
            // a valid fallback outcome.  When it is served, bytes must match.
            assert_eq!(bypass, content);
        }
        assert_eq!(std::fs::read(&path).unwrap(), content);
    }

    #[test]
    fn misaligned_length_falls_back() {
        let (_dir, path, content) = fixture(3 * DIRECT_READ_ALIGNMENT + 1);
        assert!(read_chunk_bypass(&path, content.len()).is_none());
        assert!(read_chunk_bypass(&path, 0).is_none());
        // The normal path still serves the exact bytes.
        assert_eq!(std::fs::read(&path).unwrap(), content);
    }

    #[test]
    fn missing_file_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.kvchunk");
        assert!(read_chunk_bypass(&path, DIRECT_READ_ALIGNMENT).is_none());
    }

    #[test]
    fn gate_off_never_takes_bypass() {
        let (_dir, path, content) = fixture(2 * DIRECT_READ_ALIGNMENT);
        assert!(maybe_read_chunk_bypass(&path, content.len(), false).is_none());
        // With the gate explicitly on, the same aligned file is servable.
        let served = maybe_read_chunk_bypass(&path, content.len(), true);
        if let Some(bytes) = served {
            assert_eq!(bytes, content);
        }
    }

    #[test]
    fn gate_defaults_off_without_env() {
        // Test environments do not set KVPACK_DIRECT_READ; the default must be off.
        if std::env::var_os("KVPACK_DIRECT_READ").is_none() {
            assert!(!direct_read_enabled());
        }
    }
}
