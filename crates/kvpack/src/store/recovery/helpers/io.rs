use super::*;

pub(in crate::store::recovery) fn verify_file_mac(
    path: &Path,
    total_bytes: u64,
    domain: &[u8],
    key: &[u8],
) -> Result<(), StoreError> {
    if total_bytes < SIGNATURE_BYTES {
        return Err(StoreError::Codec("signed recovery artifact is truncated"));
    }
    let mut file = fs::File::open(path).map_err(io_error("open signed recovery artifact"))?;
    let signed_bytes = total_bytes - SIGNATURE_BYTES;
    let mut mac = <HmacSha256 as Mac>::new_from_slice(key)
        .map_err(|_| StoreError::Authentication("recovery MAC key is invalid"))?;
    mac.update(domain);
    let mut remaining = signed_bytes;
    let mut buffer = vec![0u8; 1024 * 1024];
    while remaining != 0 {
        let amount = remaining.min(buffer.len() as u64) as usize;
        read_exact_or_truncated(
            &mut file,
            &mut buffer[..amount],
            "signed recovery artifact is truncated",
        )?;
        mac.update(&buffer[..amount]);
        remaining -= amount as u64;
    }
    let mut signature = [0u8; SIGNATURE_BYTES as usize];
    read_exact_or_truncated(
        &mut file,
        &mut signature,
        "recovery artifact signature is truncated",
    )?;
    mac.verify_slice(&signature)
        .map_err(|_| StoreError::Authentication("recovery artifact signature mismatch"))
}

pub(in crate::store::recovery) fn validate_new_parent(path: &Path) -> Result<&Path, StoreError> {
    let parent = path
        .parent()
        .ok_or(StoreError::State("recovery destination has no parent"))?;
    let metadata = fs::metadata(parent).map_err(io_error("inspect recovery destination parent"))?;
    if !metadata.is_dir() {
        return Err(StoreError::State(
            "recovery destination parent is not a directory",
        ));
    }
    Ok(parent)
}

pub(in crate::store::recovery) fn require_new_destination(path: &Path) -> Result<(), StoreError> {
    validate_new_parent(path)?;
    match fs::symlink_metadata(path) {
        Ok(_) => return Err(StoreError::State("recovery destination already exists")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(StoreError::Io {
                op: "inspect recovery destination",
                source,
            });
        }
    }
    Ok(())
}

pub(in crate::store::recovery) fn require_bare_catalog_path(path: &Path) -> Result<(), StoreError> {
    for suffix in ["", "-wal", "-shm"] {
        let candidate = PathBuf::from(format!("{}{suffix}", path.to_string_lossy()));
        match fs::symlink_metadata(candidate) {
            Ok(_) => return Err(StoreError::State("catalog restore destination is not bare")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(StoreError::Io {
                    op: "inspect catalog restore destination",
                    source,
                });
            }
        }
    }
    Ok(())
}

pub(in crate::store::recovery) fn hash_file(
    path: &Path,
    maximum_bytes: u64,
) -> Result<(u64, Id32), StoreError> {
    let mut file = fs::File::open(path).map_err(io_error("open recovery hash input"))?;
    let metadata = file
        .metadata()
        .map_err(io_error("inspect recovery hash input"))?;
    if !metadata.is_file() || metadata.len() > maximum_bytes {
        return Err(StoreError::Quota("recovery hash input exceeds its bound"));
    }
    let mut digest = Sha256::new();
    let mut read_bytes = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let amount = file
            .read(&mut buffer)
            .map_err(io_error("read recovery hash input"))?;
        if amount == 0 {
            break;
        }
        digest.update(&buffer[..amount]);
        read_bytes = read_bytes
            .checked_add(amount as u64)
            .ok_or(StoreError::Quota("recovery hash byte total overflow"))?;
        if read_bytes > maximum_bytes {
            return Err(StoreError::Quota("recovery hash input exceeds its bound"));
        }
    }
    if read_bytes != metadata.len() {
        return Err(StoreError::Authentication(
            "recovery hash input changed while reading",
        ));
    }
    Ok((read_bytes, digest.finalize().into()))
}

pub(in crate::store::recovery) fn write_mac(
    file: &mut fs::File,
    mac: &mut HmacSha256,
    bytes: &[u8],
) -> Result<(), StoreError> {
    file.write_all(bytes)
        .map_err(io_error("write signed recovery artifact"))?;
    mac.update(bytes);
    Ok(())
}

pub(in crate::store::recovery) fn read_exact_or_truncated(
    reader: &mut impl Read,
    destination: &mut [u8],
    message: &'static str,
) -> Result<(), StoreError> {
    reader.read_exact(destination).map_err(|error| {
        if error.kind() == std::io::ErrorKind::UnexpectedEof {
            StoreError::Codec(message)
        } else {
            StoreError::Io {
                op: "read recovery artifact",
                source: error,
            }
        }
    })
}

pub(in crate::store::recovery) fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}
