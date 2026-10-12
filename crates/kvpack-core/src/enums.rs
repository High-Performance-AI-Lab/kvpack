use crate::error::PackError;

macro_rules! wire_enum {
    ($(#[$doc:meta])* $name:ident, $what:literal, { $($variant:ident = $value:literal),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u16)]
        pub enum $name {
            $($variant = $value),+
        }

        impl $name {
            /// Fail-closed decode from the wire value.
            pub fn from_wire(value: u16) -> Result<Self, PackError> {
                match value {
                    $($value => Ok(Self::$variant),)+
                    other => Err(PackError::UnknownEnum { what: $what, value: other as u64 }),
                }
            }

            pub fn wire(self) -> u16 {
                self as u16
            }
        }
    };
}

wire_enum!(CacheKind, "cache kind", {
    OrdinaryKv = 1,
});

wire_enum!(DType, "dtype", {
    U8 = 1,
    U32 = 2,
    I8 = 3,
    I16 = 4,
    I32 = 5,
    F16 = 6,
    Bf16 = 7,
    F32 = 8,
    F64 = 9,
});

wire_enum!(Codec, "codec", {
    Raw = 1,
    Lossless = 2,
});

wire_enum!(Layout, "layout", {
    Contiguous = 1,
    Strided = 2,
});

wire_enum!(RepresentationMode, "representation mode", {
    Native = 1,
    Portable = 2,
});

wire_enum!(
/// How an engine exposes the logical token axis to an export session.
/// `TailWindow` exports only the trailing in-window tokens of the prefix
/// (sliding-window attention layers).
TokenAxisRule, "token axis rule", {
    Direct = 1,
    Gather = 2,
    TailWindow = 3,
});

impl DType {
    /// Fixed element width in bytes; `None` for `None`/`Opaque`.
    pub fn width_bytes(self) -> Option<u64> {
        match self {
            DType::U8 | DType::I8 => Some(1),
            DType::I16 | DType::F16 | DType::Bf16 => Some(2),
            DType::U32 | DType::I32 | DType::F32 => Some(4),
            DType::F64 => Some(8),
        }
    }
}
