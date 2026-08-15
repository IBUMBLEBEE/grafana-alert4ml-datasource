use crate::error::Result;

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
