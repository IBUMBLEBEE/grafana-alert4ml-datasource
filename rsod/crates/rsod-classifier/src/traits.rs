/// Stage-based detection pipeline traits
use crate::types::ClassificationResult;
use rsod_core::Result;

/// Input for classification pipeline
#[derive(Debug, Clone)]
pub struct ClassifierInput<'a> {
    pub timestamps: &'a [f64],
    pub values: &'a [f64],
}

impl<'a> ClassifierInput<'a> {
    pub fn new(timestamps: &'a [f64], values: &'a [f64]) -> Self {
        Self { timestamps, values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Single detection stage in the pipeline
///
/// Each stage performs one specific analysis and contributes to the final classification
pub trait ClassificationStage: Send + Sync {
    /// Get stage name for logging/debugging
    fn name(&self) -> &str;

    /// Execute this stage's detection logic
    fn detect(&self, input: &ClassifierInput) -> Result<()>;

    /// Get mutable access to result for writing outputs
    /// This is a simplified API; full implementation would use interior mutability
    fn stage_name(&self) -> String {
        self.name().to_string()
    }
}

/// Complete pipeline for time series classification
pub trait TimeSeriesClassifier {
    /// Classify time series based on all stages
    fn classify(&self, input: &ClassifierInput) -> Result<ClassificationResult>;

    /// Get stages
    fn stages(&self) -> Vec<&dyn ClassificationStage>;
}
