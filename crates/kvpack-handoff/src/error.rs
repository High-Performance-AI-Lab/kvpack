use std::io;

#[derive(Debug, thiserror::Error)]
pub enum HandoffError {
    #[error("canonical JSON error: {0}")]
    Canonical(String),
    #[error("live handoff cancelled")]
    Cancelled,
    #[error("live handoff deadline exceeded")]
    DeadlineExceeded,
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("live handoff validation failed: {0}")]
    Validation(String),
}

pub type Result<T> = std::result::Result<T, HandoffError>;
