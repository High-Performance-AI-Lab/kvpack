use std::io::{Read, Seek, SeekFrom};

use kvpack_core::{inspect_pack_header, Id32, PackError, PACK_FOOTER_BYTES, PACK_HEADER_BYTES};

#[derive(Debug, Clone, Copy)]
pub struct InspectionBounds {
    pub maximum_file_bytes: u64,
}

impl Default for InspectionBounds {
    fn default() -> Self {
        Self {
            maximum_file_bytes: 256 * 1024 * 1024
                + (PACK_HEADER_BYTES + PACK_FOOTER_BYTES + 16) as u64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UntrustedInspection {
    pub claimed_manifest_id: Id32,
    pub claimed_tenant_namespace: Id32,
    pub claimed_key_epoch: u64,
    pub claimed_manifest_bytes: u64,
    pub encrypted: bool,
}

/// Bounded structural inspection only.  This type deliberately carries no
/// store/key handle and has no conversion into `AuthenticatedArtifact`.
pub fn inspect_untrusted(
    mut source: impl Read + Seek,
    bounds: InspectionBounds,
) -> Result<UntrustedInspection, PackError> {
    let length = source
        .seek(SeekFrom::End(0))
        .map_err(|_| PackError::Truncated("untrusted source length unavailable"))?;
    if length < (PACK_HEADER_BYTES + PACK_FOOTER_BYTES) as u64 || length > bounds.maximum_file_bytes
    {
        return Err(PackError::Bounds(
            "untrusted source length is outside bounds",
        ));
    }
    source
        .seek(SeekFrom::Start(0))
        .map_err(|_| PackError::Truncated("untrusted source is not seekable"))?;
    let mut header = vec![0u8; PACK_HEADER_BYTES];
    source
        .read_exact(&mut header)
        .map_err(|_| PackError::Truncated("truncated untrusted pack header"))?;
    let parsed = inspect_pack_header(&header)?;
    if PACK_HEADER_BYTES as u64 + parsed.manifest_bytes + PACK_FOOTER_BYTES as u64 != length {
        return Err(PackError::Bounds("untrusted pack length fields disagree"));
    }
    Ok(UntrustedInspection {
        claimed_manifest_id: parsed.manifest_id,
        claimed_tenant_namespace: parsed.tenant_namespace,
        claimed_key_epoch: parsed.key_epoch,
        claimed_manifest_bytes: parsed.manifest_bytes,
        encrypted: parsed.encrypted,
    })
}
