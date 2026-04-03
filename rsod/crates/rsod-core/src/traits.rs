use crate::error::Result;
use crate::types::{DetectionResult, TimeSeriesData};

/// Unified detector interface.
///
/// All anomaly detection strategies (baseline, outlier, forecast)
/// implement this trait to provide a consistent API for FFI and orchestration.
pub trait Detector: Send + Sync {
    /// Train the detector on historical data.
    fn fit(&mut self, data: &TimeSeriesData) -> Result<()>;

    /// Run anomaly detection on the given data.
    fn predict(&self, data: &TimeSeriesData) -> Result<DetectionResult>;

    /// Convenience: train then predict in one step.
    fn fit_predict(&mut self, data: &TimeSeriesData) -> Result<DetectionResult> {
        self.fit(data)?;
        self.predict(data)
    }
}

/// Ability to serialize/deserialize a trained model for persistence.
///
/// Used by `rsod-storage` to save models to SQLite/PostgreSQL
/// without knowing the concrete detector type.
pub trait ModelSerializable {
    /// Serialize the trained model state to bytes.
    fn serialize_model(&self) -> Result<Vec<u8>>;

    /// Restore a detector from previously serialized bytes.
    fn deserialize_model(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

/// Unified storage interface for model persistence.
///
/// Abstracts over different backends (SQLite, PostgreSQL, etc.)
/// so that detectors and the FFI layer do not depend on concrete storage.
pub trait ModelStorage: Send + Sync {
    /// Persist a model's serialized artifacts by its unique identifier.
    fn save(&self, uuid: &str, artifacts: &[u8]) -> Result<()>;

    /// Load a model's serialized artifacts by its unique identifier.
    ///
    /// Returns `Ok(None)` if no model with the given uuid exists.
    fn load(&self, uuid: &str) -> Result<Option<Vec<u8>>>;

    /// Delete a model by its unique identifier.
    fn delete(&self, uuid: &str) -> Result<()>;

    /// Check if a model with the given uuid exists.
    fn exists(&self, uuid: &str) -> Result<bool> {
        Ok(self.load(uuid)?.is_some())
    }
}
