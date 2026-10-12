use crate::PackError;

pub(crate) struct Encoder {
    bytes: Vec<u8>,
}

impl Encoder {
    pub(crate) fn new(magic: &[u8; 8]) -> Self {
        Self {
            bytes: magic.to_vec(),
        }
    }
    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    pub(crate) fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub(crate) fn bytes(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }
    pub(crate) fn id(&mut self, value: &[u8; 32]) {
        self.bytes(value);
    }
    pub(crate) fn string(&mut self, value: &str) -> Result<(), PackError> {
        let len =
            u16::try_from(value.len()).map_err(|_| PackError::Bounds("string is too long"))?;
        self.u16(len);
        self.bytes(value.as_bytes());
        Ok(())
    }
    pub(crate) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

pub(crate) struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    pub(crate) fn new(bytes: &'a [u8], magic: &[u8; 8]) -> Result<Self, PackError> {
        if bytes.len() < 8 {
            return Err(PackError::Truncated("truncated canonical object"));
        }
        if &bytes[..8] != magic {
            return Err(PackError::BadMagic("invalid canonical object magic"));
        }
        Ok(Self { bytes, offset: 8 })
    }
    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], PackError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(PackError::Bounds("canonical offset overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(PackError::Truncated("truncated canonical object"))?;
        self.offset = end;
        Ok(value)
    }
    pub(crate) fn u8(&mut self) -> Result<u8, PackError> {
        Ok(self.take(1)?[0])
    }
    pub(crate) fn u16(&mut self) -> Result<u16, PackError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, PackError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, PackError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn id(&mut self) -> Result<[u8; 32], PackError> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    pub(crate) fn string(&mut self) -> Result<String, PackError> {
        let len = self.u16()? as usize;
        let bytes = self.take(len)?;
        let value = std::str::from_utf8(bytes)
            .map_err(|_| PackError::Semantics("state name is not UTF-8"))?;
        Ok(value.to_owned())
    }
    pub(crate) fn finish(self) -> Result<(), PackError> {
        if self.offset != self.bytes.len() {
            return Err(PackError::Reserved("trailing canonical bytes"));
        }
        Ok(())
    }
}
