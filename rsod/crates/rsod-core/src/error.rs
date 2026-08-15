/// Unified error type for the rsod project.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum RsodError {
    #[error("missing rate {rate:.1}% exceeds threshold {threshold:.1}%")]
    MissingRateTooHigh { rate: f64, threshold: f64 },

    #[error("insufficient data points: need at least {need}, got {got}")]
    InsufficientData { need: usize, got: usize },

    #[error("data is empty")]
    EmptyData,

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model serialization failed: {0}")]
    ModelSerialize(String),

    #[error("model deserialization failed: {0}")]
    ModelDeserialize(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("detection failed: {0}")]
    Detection(String),

    #[error("preprocessing failed: {0}")]
    Preprocessing(String),

    #[error("{0}")]
    Other(String),
}

impl From<String> for RsodError {
    fn from(s: String) -> Self {
        RsodError::Other(s)
    }
}

impl From<&str> for RsodError {
    fn from(s: &str) -> Self {
        RsodError::Other(s.to_string())
    }
}

impl From<std::io::Error> for RsodError {
    fn from(e: std::io::Error) -> Self {
        RsodError::Storage(e.to_string())
    }
}

impl From<serde_json::Error> for RsodError {
    fn from(e: serde_json::Error) -> Self {
        RsodError::InvalidConfig(e.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for RsodError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        RsodError::Other(e.to_string())
    }
}

impl RsodError {
    /// Wrap any displayable error as [`RsodError::Other`]. Handy for adapting
    /// third-party error types (polars, perpetual, rusqlite, …) at `?` sites
    /// without pulling those crates into `rsod-core`.
    pub fn other<E: std::fmt::Display>(e: E) -> Self {
        RsodError::Other(e.to_string())
    }
}

/// Unified Result type for the rsod project.
pub type Result<T> = std::result::Result<T, RsodError>;
