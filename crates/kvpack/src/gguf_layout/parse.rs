use super::*;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_SUPPORTED_VERSION: u32 = 3;
// Fail-closed parse bounds: the reader consumes untrusted files, so every
// variable-length field is capped before allocation.
const GGUF_MAX_METADATA_PAIRS: u64 = 1 << 20;
const GGUF_MAX_STRING_BYTES: u64 = 1 << 24;
const GGUF_MAX_ARRAY_ELEMENTS: u64 = 1 << 20;

/// Parse the GGUF header and key/value metadata section from `path`. Tensor
/// data is never touched; the reader stops after the metadata pairs.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata, StoreError> {
    let file = std::fs::File::open(path).map_err(io_error("open gguf metadata"))?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .map_err(io_error("read gguf header"))?;
    if &magic != GGUF_MAGIC {
        return Err(StoreError::Expectation("gguf metadata has a bad magic"));
    }
    let version = read_u32(&mut reader)?;
    if version != GGUF_SUPPORTED_VERSION {
        return Err(StoreError::Expectation(
            "gguf metadata version is outside the supported v3 grammar",
        ));
    }
    // tensor_count is part of the header but tensor infos are not read.
    let _tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;
    if metadata_count > GGUF_MAX_METADATA_PAIRS {
        return Err(StoreError::Expectation(
            "gguf metadata pair count is outside the parse bounds",
        ));
    }
    let mut metadata = GgufMetadata::new();
    for _ in 0..metadata_count {
        let key = read_gguf_string(&mut reader)?;
        let value = read_gguf_value(&mut reader)?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

fn read_exact<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], StoreError> {
    let mut bytes = [0u8; N];
    reader
        .read_exact(&mut bytes)
        .map_err(io_error("read gguf metadata"))?;
    Ok(bytes)
}

fn read_u32(reader: &mut impl Read) -> Result<u32, StoreError> {
    Ok(u32::from_le_bytes(read_exact(reader)?))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, StoreError> {
    Ok(u64::from_le_bytes(read_exact(reader)?))
}

fn read_gguf_string(reader: &mut impl Read) -> Result<String, StoreError> {
    let length = read_u64(reader)?;
    if length > GGUF_MAX_STRING_BYTES {
        return Err(StoreError::Expectation(
            "gguf metadata string is outside the parse bounds",
        ));
    }
    let mut bytes = vec![0u8; length as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(io_error("read gguf metadata"))?;
    String::from_utf8(bytes)
        .map_err(|_| StoreError::Expectation("gguf metadata string is not valid utf-8"))
}

fn read_gguf_scalar(reader: &mut impl Read, value_type: u32) -> Result<GgufValue, StoreError> {
    match value_type {
        0 => Ok(GgufValue::Uint8(read_exact::<1>(reader)?[0])),
        1 => Ok(GgufValue::Int8(read_exact::<1>(reader)?[0] as i8)),
        2 => Ok(GgufValue::Uint16(u16::from_le_bytes(read_exact(reader)?))),
        3 => Ok(GgufValue::Int16(i16::from_le_bytes(read_exact(reader)?))),
        4 => Ok(GgufValue::Uint32(read_u32(reader)?)),
        5 => Ok(GgufValue::Int32(i32::from_le_bytes(read_exact(reader)?))),
        6 => Ok(GgufValue::Float32(f32::from_le_bytes(read_exact(reader)?))),
        7 => Ok(GgufValue::Bool(read_exact::<1>(reader)?[0] != 0)),
        8 => Ok(GgufValue::String(read_gguf_string(reader)?)),
        10 => Ok(GgufValue::Uint64(read_u64(reader)?)),
        11 => Ok(GgufValue::Int64(i64::from_le_bytes(read_exact(reader)?))),
        12 => Ok(GgufValue::Float64(f64::from_le_bytes(read_exact(reader)?))),
        _ => Err(StoreError::Expectation(
            "gguf metadata value type is outside the v3 grammar",
        )),
    }
}

fn read_gguf_value(reader: &mut impl Read) -> Result<GgufValue, StoreError> {
    let value_type = read_u32(reader)?;
    if value_type != 9 {
        return read_gguf_scalar(reader, value_type);
    }
    let element_type = read_u32(reader)?;
    if element_type == 9 || element_type > 12 {
        return Err(StoreError::Expectation(
            "gguf metadata array element type is outside the v3 grammar",
        ));
    }
    let length = read_u64(reader)?;
    if length > GGUF_MAX_ARRAY_ELEMENTS {
        return Err(StoreError::Expectation(
            "gguf metadata array is outside the parse bounds",
        ));
    }
    let mut elements = Vec::with_capacity(length as usize);
    for _ in 0..length {
        elements.push(read_gguf_scalar(reader, element_type)?);
    }
    Ok(GgufValue::Array(elements))
}
