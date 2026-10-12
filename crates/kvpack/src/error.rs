use std::fmt;

use kvpack_core::PackError;

#[derive(Debug)]
#[non_exhaustive]
pub enum StoreError {
    Pack(PackError),
    Io {
        op: &'static str,
        source: std::io::Error,
    },
    Catalog(rusqlite::Error),
    Integrity(&'static str),
    Authentication(&'static str),
    Codec(&'static str),
    Poisoned(&'static str),
    Expectation(&'static str),
    Quota(&'static str),
    Endurance(&'static str),
    State(&'static str),
    NotFound,
    Busy,
    Cancelled,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pack(error) => write!(f, "{error}"),
            Self::Io { op, source } => write!(f, "{op}: {source}"),
            Self::Catalog(error) => write!(f, "catalog: {error}"),
            Self::Integrity(message)
            | Self::Authentication(message)
            | Self::Codec(message)
            | Self::Poisoned(message)
            | Self::Expectation(message)
            | Self::Quota(message)
            | Self::Endurance(message)
            | Self::State(message) => f.write_str(message),
            Self::NotFound => f.write_str("artifact not found"),
            Self::Busy => f.write_str("resource is busy"),
            Self::Cancelled => f.write_str("operation was cancelled"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pack(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            Self::Catalog(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PackError> for StoreError {
    fn from(error: PackError) -> Self {
        match error {
            PackError::Authentication(message) => Self::Authentication(message),
            PackError::Codec(message) => Self::Codec(message),
            PackError::Poisoned(message) => Self::Poisoned(message),
            other => Self::Pack(other),
        }
    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Catalog(error)
    }
}

pub(crate) fn io_error(op: &'static str) -> impl FnOnce(std::io::Error) -> StoreError {
    move |source| StoreError::Io { op, source }
}
