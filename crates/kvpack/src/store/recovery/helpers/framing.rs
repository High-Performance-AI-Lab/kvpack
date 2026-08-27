use super::*;

pub(in crate::store::recovery) fn encode_backup_header(
    header: BackupHeader,
) -> [u8; BACKUP_HEADER_BYTES] {
    let mut encoded = [0u8; BACKUP_HEADER_BYTES];
    encoded[..8].copy_from_slice(BACKUP_MAGIC);
    put_u16(&mut encoded, 8, RECOVERY_VERSION);
    put_u16(&mut encoded, 10, BACKUP_FLAGS);
    put_u32(&mut encoded, 12, BACKUP_HEADER_BYTES as u32);
    encoded[16..48].copy_from_slice(&header.tenant);
    put_u64(&mut encoded, 48, header.catalog_schema);
    put_u64(&mut encoded, 56, header.catalog_epoch);
    put_u64(&mut encoded, 64, header.key_epoch);
    put_u64(&mut encoded, 72, header.created_ns);
    put_u64(&mut encoded, 80, header.plaintext_bytes);
    put_u32(&mut encoded, 88, BACKUP_BLOCK_BYTES as u32);
    put_u64(&mut encoded, 96, header.block_count);
    encoded[104..136].copy_from_slice(&header.plaintext_digest);
    encoded[136..152].copy_from_slice(&header.salt);
    encoded[152..156].copy_from_slice(&header.nonce_prefix);
    encoded
}

pub(in crate::store::recovery) fn decode_backup_header(
    encoded: &[u8; BACKUP_HEADER_BYTES],
) -> Result<BackupHeader, StoreError> {
    if &encoded[..8] != BACKUP_MAGIC
        || get_u16(encoded, 8) != RECOVERY_VERSION
        || get_u16(encoded, 10) != BACKUP_FLAGS
        || get_u32(encoded, 12) as usize != BACKUP_HEADER_BYTES
        || get_u32(encoded, 88) as usize != BACKUP_BLOCK_BYTES
        || encoded[92..96].iter().any(|value| *value != 0)
        || encoded[156..].iter().any(|value| *value != 0)
    {
        return Err(StoreError::Codec(
            "catalog backup framing, flags, or reserved bytes are invalid",
        ));
    }
    Ok(BackupHeader {
        tenant: encoded[16..48].try_into().unwrap(),
        catalog_schema: get_u64(encoded, 48),
        catalog_epoch: get_u64(encoded, 56),
        key_epoch: get_u64(encoded, 64),
        created_ns: get_u64(encoded, 72),
        plaintext_bytes: get_u64(encoded, 80),
        block_count: get_u64(encoded, 96),
        plaintext_digest: encoded[104..136].try_into().unwrap(),
        salt: encoded[136..152].try_into().unwrap(),
        nonce_prefix: encoded[152..156].try_into().unwrap(),
    })
}

pub(in crate::store::recovery) fn encode_inventory_header(
    header: InventoryHeader,
) -> [u8; INVENTORY_HEADER_BYTES] {
    let mut encoded = [0u8; INVENTORY_HEADER_BYTES];
    encoded[..8].copy_from_slice(INVENTORY_MAGIC);
    put_u16(&mut encoded, 8, RECOVERY_VERSION);
    put_u16(&mut encoded, 10, INVENTORY_FLAGS);
    put_u32(&mut encoded, 12, INVENTORY_HEADER_BYTES as u32);
    encoded[16..48].copy_from_slice(&header.tenant);
    put_u64(&mut encoded, 48, header.catalog_schema);
    put_u64(&mut encoded, 56, header.catalog_epoch);
    put_u64(&mut encoded, 64, header.key_epoch);
    put_u64(&mut encoded, 72, header.created_ns);
    put_u64(&mut encoded, 80, header.entry_count);
    put_u64(&mut encoded, 88, header.payload_bytes);
    encoded[96..128].copy_from_slice(&header.payload_digest);
    encoded[128..144].copy_from_slice(&header.salt);
    encoded
}

pub(in crate::store::recovery) fn decode_inventory_header(
    encoded: &[u8; INVENTORY_HEADER_BYTES],
) -> Result<InventoryHeader, StoreError> {
    if &encoded[..8] != INVENTORY_MAGIC
        || get_u16(encoded, 8) != RECOVERY_VERSION
        || get_u16(encoded, 10) != INVENTORY_FLAGS
        || get_u32(encoded, 12) as usize != INVENTORY_HEADER_BYTES
        || encoded[144..].iter().any(|value| *value != 0)
    {
        return Err(StoreError::Codec(
            "inventory framing, flags, or reserved bytes are invalid",
        ));
    }
    Ok(InventoryHeader {
        tenant: encoded[16..48].try_into().unwrap(),
        catalog_schema: get_u64(encoded, 48),
        catalog_epoch: get_u64(encoded, 56),
        key_epoch: get_u64(encoded, 64),
        created_ns: get_u64(encoded, 72),
        entry_count: get_u64(encoded, 80),
        payload_bytes: get_u64(encoded, 88),
        payload_digest: encoded[96..128].try_into().unwrap(),
        salt: encoded[128..144].try_into().unwrap(),
    })
}

pub(in crate::store::recovery) fn encode_inventory_entry(
    entry: &InventoryEntry,
) -> [u8; INVENTORY_ENTRY_BYTES] {
    let mut encoded = [0u8; INVENTORY_ENTRY_BYTES];
    encoded[0] = match entry.kind {
        InventoryObjectKind::Manifest => 0,
        InventoryObjectKind::Chunk => 1,
    };
    encoded[8..40].copy_from_slice(&entry.object_id);
    encoded[40..72].copy_from_slice(&entry.object_digest);
    put_u64(&mut encoded, 72, entry.object_bytes);
    put_u64(&mut encoded, 80, entry.publication_generation);
    put_u64(&mut encoded, 88, entry.key_epoch);
    encoded
}

pub(in crate::store::recovery) fn decode_inventory_entry(
    encoded: &[u8; INVENTORY_ENTRY_BYTES],
) -> Result<InventoryEntry, StoreError> {
    if encoded[1..8].iter().any(|value| *value != 0) {
        return Err(StoreError::Codec(
            "inventory entry reserved bytes are nonzero",
        ));
    }
    let kind = match encoded[0] {
        0 => InventoryObjectKind::Manifest,
        1 => InventoryObjectKind::Chunk,
        _ => return Err(StoreError::Codec("inventory object kind is unknown")),
    };
    let entry = InventoryEntry {
        kind,
        object_id: encoded[8..40].try_into().unwrap(),
        object_digest: encoded[40..72].try_into().unwrap(),
        object_bytes: get_u64(encoded, 72),
        publication_generation: get_u64(encoded, 80),
        key_epoch: get_u64(encoded, 88),
    };
    if entry.object_id == [0; 32]
        || entry.object_digest == [0; 32]
        || entry.object_bytes == 0
        || entry.key_epoch == 0
        || (kind == InventoryObjectKind::Manifest && entry.publication_generation == 0)
        || (kind == InventoryObjectKind::Chunk && entry.publication_generation != 0)
    {
        return Err(StoreError::Codec(
            "inventory entry contains invalid canonical values",
        ));
    }
    Ok(entry)
}

pub(in crate::store::recovery) fn backup_block_aad(
    header: &[u8; BACKUP_HEADER_BYTES],
    ordinal: u64,
    plaintext_bytes: u32,
    stored_bytes: u32,
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(BACKUP_AEAD_DOMAIN.len() + BACKUP_HEADER_BYTES + 16);
    aad.extend_from_slice(BACKUP_AEAD_DOMAIN);
    aad.extend_from_slice(header);
    aad.extend_from_slice(&ordinal.to_le_bytes());
    aad.extend_from_slice(&plaintext_bytes.to_le_bytes());
    aad.extend_from_slice(&stored_bytes.to_le_bytes());
    aad
}

pub(in crate::store::recovery) fn block_nonce(prefix: &[u8; 4], ordinal: u64) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[..4].copy_from_slice(prefix);
    nonce[4..].copy_from_slice(&ordinal.to_le_bytes());
    nonce
}

pub(in crate::store::recovery) fn encode_record_header(
    plaintext_bytes: u32,
    stored_bytes: u32,
) -> [u8; 8] {
    let mut encoded = [0u8; 8];
    put_u32(&mut encoded, 0, plaintext_bytes);
    put_u32(&mut encoded, 4, stored_bytes);
    encoded
}

pub(in crate::store::recovery) fn expected_backup_bytes(
    plaintext_bytes: u64,
    block_count: u64,
) -> Result<u64, StoreError> {
    (BACKUP_HEADER_BYTES as u64)
        .checked_add(plaintext_bytes)
        .and_then(|value| {
            block_count
                .checked_mul(RECORD_HEADER_BYTES + AEAD_TAG_BYTES)
                .and_then(|records| value.checked_add(records))
        })
        .and_then(|value| value.checked_add(SIGNATURE_BYTES))
        .ok_or(StoreError::Codec("catalog backup length overflow"))
}

pub(in crate::store::recovery) fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

pub(in crate::store::recovery) fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

pub(in crate::store::recovery) fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

pub(in crate::store::recovery) fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

pub(in crate::store::recovery) fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

pub(in crate::store::recovery) fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
