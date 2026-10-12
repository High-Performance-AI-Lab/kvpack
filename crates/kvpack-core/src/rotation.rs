//! Versioned, authenticated Rotation ABI family reservation.
//!
//! The descriptor is identity material only. It does not execute rotation or
//! enable a runtime route. An absent [`RotationFamilyHook`] is byte-for-byte
//! neutral; a present hook derives a distinct engine-cache ABI.

use sha2::{Digest, Sha256};

use crate::canonical::{Decoder, Encoder};
use crate::{Id32, PackError, RepresentationFamilyId, ZERO_ID};

pub const ROTATION_FAMILY_MAGIC: &[u8; 8] = b"KVRTA1\0\0";
pub const ROTATION_FAMILY_VERSION: u16 = 1;
pub const MAX_ROTARY_DIMENSION: u32 = 4_096;
pub const QWEN25_ROTARY_DIMENSION: u32 = 128;
pub const QWEN25_FRAC64_TABLE_SHA256: Id32 = [
    0xc5, 0xd5, 0x12, 0x3c, 0x8a, 0xf3, 0xba, 0x4f, 0x7c, 0xe7, 0x78, 0xf4, 0x3f, 0xe5, 0xdf, 0xb4,
    0x16, 0x0d, 0xba, 0x70, 0x2f, 0x29, 0xc2, 0x0e, 0x88, 0xc9, 0xa0, 0x2d, 0xd5, 0x24, 0x38, 0x13,
];
pub const FIXED_Q30_D7_D6_COEFFICIENT_SHA256: Id32 = [
    0xd2, 0xe8, 0xc7, 0xc6, 0x2e, 0x2c, 0xa3, 0x28, 0xb1, 0xcd, 0x7e, 0x4d, 0x84, 0x62, 0xe4, 0x0e,
    0xf8, 0x8e, 0xed, 0xe5, 0x33, 0x5e, 0xbf, 0x21, 0x22, 0xd6, 0x19, 0x52, 0xd7, 0xdb, 0xd9, 0x50,
];

/// Pinned full-precision Qwen2.5 base-1,000,000 increments. Consumers serialize
/// these constants little-endian; they never regenerate frequencies with libm.
pub const QWEN25_FRAC64: [u64; 64] = [
    0x28be60db9391054a,
    0x20d53d2925f67455,
    0x1a754bcdf1528959,
    0x1552352f8941b08d,
    0x112e743e936615ee,
    0x0dd875b948b28ead,
    0x0b28467931a8dd3d,
    0x08fdb50634ad64e9,
    0x073ed00d7f5d2561,
    0x05d6afb33f733a7b,
    0x04b47b368f3b0d41,
    0x03ca9f315c34f772,
    0x030e2b042f8045f4,
    0x02764dd2da302666,
    0x01fbecf10dbdd0de,
    0x01994ece8966ef84,
    0x0149d66800919d8c,
    0x0109cc07b015a4eb,
    0x00d630c00191eafb,
    0x00ac9a8b11cb9296,
    0x008b176173e0158c,
    0x007015edf6892b10,
    0x005a52c769d29a9f,
    0x0048c94f8f79e94e,
    0x003aa782077acd6f,
    0x002f442137d117bb,
    0x002616cb87806cce,
    0x001eb19a1cf78363,
    0x0018bbfcb87996fd,
    0x0013ee9518d1aadc,
    0x00100fe04cd6d8fa,
    0x000cf185f65ff0fe,
    0x000a6e2d468c2d51,
    0x000867bdbcb5c01e,
    0x0006c5f6bbe3aaa2,
    0x0005754d1a81587f,
    0x000466011629d480,
    0x00038b61b8a6110a,
    0x0002db34db71109b,
    0x00024d3cad9ee4b5,
    0x0001dad5016848ff,
    0x00017ea3c3643613,
    0x00013458e2f59024,
    0x0000f87aac61553d,
    0x0000c83c2a081074,
    0x0000a15b9a548735,
    0x0000820768b7bde8,
    0x000068c8660680c0,
    0x000054703b647f64,
    0x0000440b458d556a,
    0x000036d52f1b06a3,
    0x00002c2fc14c3c4e,
    0x0000239b7d4cc4a1,
    0x00001cb1a55dd54b,
    0x0000171f6e711b4a,
    0x000012a220a7cd4a,
    0x00000f03f853d5b6,
    0x00000c19a21a93bc,
    0x000009c0341a7f16,
    0x000007db8bce124f,
    0x00000654fd1ccb29,
    0x0000051a42d73fea,
    0x0000041ca3eab5fa,
    0x00000350430ff7cb,
];

macro_rules! wire_enum {
    ($name:ident, $what:literal, $variant:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $variant,
        }
        impl $name {
            fn wire(self) -> u16 {
                1
            }
            fn from_wire(value: u16) -> Result<Self, PackError> {
                match value {
                    1 => Ok(Self::$variant),
                    other => Err(PackError::UnknownEnum {
                        what: $what,
                        value: other as u64,
                    }),
                }
            }
        }
    };
}

wire_enum!(
    CoefficientSet,
    "rotation coefficient set",
    FixedQ30Degree7Degree6V1
);
wire_enum!(SincosOrder, "rotation sincos order", FixedQ30HornerV1);
wire_enum!(PhaseOrigin, "rotation phase origin", Zero);
wire_enum!(
    PositionConvention,
    "rotation position convention",
    ZeroBasedAbsolute
);
wire_enum!(RopePairing, "rotation pairing", NeoxHalfSplit);
wire_enum!(DenormalPolicy, "rotation denormal policy", PreserveIeee);
wire_enum!(F32Rounding, "rotation f32 rounding", NearestTiesEven);
wire_enum!(
    F16CacheCast,
    "rotation f16 cache cast",
    NearestTiesEvenPreserveSubnormals
);
wire_enum!(
    RotationOrder,
    "rotation operation order",
    FmaWithRoundedCrossProductV1
);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RotationFamilyDescriptorV1 {
    pub coefficient_set: CoefficientSet,
    pub coefficient_set_sha256: Id32,
    pub sincos_order: SincosOrder,
    pub phase_origin: PhaseOrigin,
    pub position_convention: PositionConvention,
    pub pairing: RopePairing,
    pub denormal_policy: DenormalPolicy,
    pub f32_rounding: F32Rounding,
    pub f16_cache_cast: F16CacheCast,
    pub rotation_order: RotationOrder,
    pub rotary_dimension: u32,
    pub model_representation_id: Id32,
    pub frac64_le: Vec<u8>,
    pub frac64_sha256: Id32,
}

impl RotationFamilyDescriptorV1 {
    pub fn new(
        rotary_dimension: u32,
        model_representation_id: Id32,
        frac64_le: Vec<u8>,
        coefficient_set_sha256: Id32,
    ) -> Result<Self, PackError> {
        let frac64_sha256 = Sha256::digest(&frac64_le).into();
        let value = Self {
            coefficient_set: CoefficientSet::FixedQ30Degree7Degree6V1,
            coefficient_set_sha256,
            sincos_order: SincosOrder::FixedQ30HornerV1,
            phase_origin: PhaseOrigin::Zero,
            position_convention: PositionConvention::ZeroBasedAbsolute,
            pairing: RopePairing::NeoxHalfSplit,
            denormal_policy: DenormalPolicy::PreserveIeee,
            f32_rounding: F32Rounding::NearestTiesEven,
            f16_cache_cast: F16CacheCast::NearestTiesEvenPreserveSubnormals,
            rotation_order: RotationOrder::FmaWithRoundedCrossProductV1,
            rotary_dimension,
            model_representation_id,
            frac64_le,
            frac64_sha256,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn pinned_qwen25(model_representation_id: Id32) -> Result<Self, PackError> {
        let mut bytes = Vec::with_capacity(QWEN25_FRAC64.len() * 8);
        for increment in QWEN25_FRAC64 {
            bytes.extend_from_slice(&increment.to_le_bytes());
        }
        let value = Self::new(
            QWEN25_ROTARY_DIMENSION,
            model_representation_id,
            bytes,
            FIXED_Q30_D7_D6_COEFFICIENT_SHA256,
        )?;
        if value.frac64_sha256 != QWEN25_FRAC64_TABLE_SHA256 {
            return Err(PackError::Semantics(
                "pinned qwen2.5 rotation table hash does not match its family constant",
            ));
        }
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), PackError> {
        if self.rotary_dimension < 2
            || self.rotary_dimension > MAX_ROTARY_DIMENSION
            || self.rotary_dimension % 2 != 0
        {
            return Err(PackError::Bounds(
                "rotation rotary dimension is outside the v1 bounds",
            ));
        }
        let expected = usize::try_from(self.rotary_dimension / 2)
            .ok()
            .and_then(|pairs| pairs.checked_mul(8))
            .ok_or(PackError::Bounds("rotation table byte count overflow"))?;
        if self.frac64_le.len() != expected {
            return Err(PackError::Bounds(
                "rotation frac64 table does not match rotary dimension",
            ));
        }
        if self.model_representation_id == ZERO_ID || self.coefficient_set_sha256 == ZERO_ID {
            return Err(PackError::Semantics(
                "rotation descriptor contains a zero identity",
            ));
        }
        let actual: Id32 = Sha256::digest(&self.frac64_le).into();
        if actual != self.frac64_sha256 {
            return Err(PackError::Semantics(
                "rotation frac64 table hash does not match authenticated bytes",
            ));
        }
        let mut any_low_bits = false;
        for bytes in self.frac64_le.chunks_exact(8) {
            let increment = u64::from_le_bytes(bytes.try_into().expect("eight-byte chunk"));
            if increment == 0 {
                return Err(PackError::Semantics(
                    "rotation frac64 table contains a zero increment",
                ));
            }
            any_low_bits |= increment as u32 != 0;
        }
        if !any_low_bits {
            return Err(PackError::Semantics(
                "rotation frac64 table is only a lifted 32-bit frequency table",
            ));
        }
        Ok(())
    }

    pub fn encode_canonical(&self) -> Result<Vec<u8>, PackError> {
        self.validate()?;
        let mut out = Encoder::new(ROTATION_FAMILY_MAGIC);
        out.u16(ROTATION_FAMILY_VERSION);
        out.u16(0);
        out.u16(self.coefficient_set.wire());
        out.u16(self.sincos_order.wire());
        out.u16(self.phase_origin.wire());
        out.u16(self.position_convention.wire());
        out.u16(self.pairing.wire());
        out.u16(self.denormal_policy.wire());
        out.u16(self.f32_rounding.wire());
        out.u16(self.f16_cache_cast.wire());
        out.u16(self.rotation_order.wire());
        out.u16(0);
        out.u32(self.rotary_dimension);
        out.id(&self.coefficient_set_sha256);
        out.id(&self.model_representation_id);
        out.u32(
            u32::try_from(self.frac64_le.len())
                .map_err(|_| PackError::Bounds("rotation table exceeds u32"))?,
        );
        out.bytes(&self.frac64_le);
        out.id(&self.frac64_sha256);
        Ok(out.finish())
    }

    pub fn decode_canonical(bytes: &[u8]) -> Result<Self, PackError> {
        let mut input = Decoder::new(bytes, ROTATION_FAMILY_MAGIC)?;
        if input.u16()? != ROTATION_FAMILY_VERSION {
            return Err(PackError::BadMagic("unsupported rotation family version"));
        }
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "rotation family reserved field is nonzero",
            ));
        }
        let coefficient_set = CoefficientSet::from_wire(input.u16()?)?;
        let sincos_order = SincosOrder::from_wire(input.u16()?)?;
        let phase_origin = PhaseOrigin::from_wire(input.u16()?)?;
        let position_convention = PositionConvention::from_wire(input.u16()?)?;
        let pairing = RopePairing::from_wire(input.u16()?)?;
        let denormal_policy = DenormalPolicy::from_wire(input.u16()?)?;
        let f32_rounding = F32Rounding::from_wire(input.u16()?)?;
        let f16_cache_cast = F16CacheCast::from_wire(input.u16()?)?;
        let rotation_order = RotationOrder::from_wire(input.u16()?)?;
        if input.u16()? != 0 {
            return Err(PackError::Reserved(
                "rotation family secondary reserved field is nonzero",
            ));
        }
        let rotary_dimension = input.u32()?;
        let coefficient_set_sha256 = input.id()?;
        let model_representation_id = input.id()?;
        let table_len = input.u32()? as usize;
        let frac64_le = input.take(table_len)?.to_vec();
        let frac64_sha256 = input.id()?;
        input.finish()?;
        let value = Self {
            coefficient_set,
            coefficient_set_sha256,
            sincos_order,
            phase_origin,
            position_convention,
            pairing,
            denormal_policy,
            f32_rounding,
            f16_cache_cast,
            rotation_order,
            rotary_dimension,
            model_representation_id,
            frac64_le,
            frac64_sha256,
        };
        value.validate()?;
        if value.encode_canonical()? != bytes {
            return Err(PackError::Reserved(
                "rotation family encoding is not canonical",
            ));
        }
        Ok(value)
    }

    pub fn identity(&self) -> Result<Id32, PackError> {
        let mut hash = Sha256::new();
        hash.update(b"kvpack/v1/rotation-family\0");
        hash.update(self.encode_canonical()?);
        Ok(hash.finalize().into())
    }
}

/// Optional family hook. `None` is deliberately neutral for compatibility.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RotationFamilyHook {
    pub descriptor_id: Option<Id32>,
}

impl RotationFamilyHook {
    pub fn from_descriptor(value: &RotationFamilyDescriptorV1) -> Result<Self, PackError> {
        Ok(Self {
            descriptor_id: Some(value.identity()?),
        })
    }

    pub fn bind_engine_cache_abi(&self, base: &Id32) -> Id32 {
        let Some(descriptor_id) = self.descriptor_id else {
            return *base;
        };
        let mut hash = Sha256::new();
        hash.update(b"kvpack/v1/rotation-bound-engine-cache-abi\0");
        hash.update(base);
        hash.update(descriptor_id);
        hash.finalize().into()
    }
}

/// Apply the optional hook to a family clone. Every other family field remains
/// byte-identical; only the already-authenticated engine-cache ABI changes.
pub fn bind_representation_family(
    family: &RepresentationFamilyId,
    hook: RotationFamilyHook,
) -> RepresentationFamilyId {
    let mut bound = family.clone();
    bound.engine_cache_abi = hook.bind_engine_cache_abi(&family.engine_cache_abi);
    bound
}

#[cfg(test)]
mod tests;
