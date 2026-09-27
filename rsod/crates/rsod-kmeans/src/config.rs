use serde::{Deserialize, Serialize};

pub const FEATURE_WIDTH: usize = 6;
pub const FEATURE_CONTEXT_POINTS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct KmeansOptions {
    pub train_window_secs: i64,
    pub retrain_every_secs: i64,
    pub max_models: usize,
    pub min_models_for_alert: usize,
    pub anomaly_threshold: f64,
    pub min_train_vectors: usize,
    pub max_train_vectors: usize,
    pub max_iterations: usize,
    pub seed: u64,
    pub persist_model: bool,
}

impl Default for KmeansOptions {
    fn default() -> Self {
        Self {
            train_window_secs: 6 * 60 * 60,
            retrain_every_secs: 3 * 60 * 60,
            max_models: 18,
            min_models_for_alert: 18,
            anomaly_threshold: 99.0,
            min_train_vectors: 120,
            max_train_vectors: 1440,
            max_iterations: 1000,
            seed: 0x4b4d_4541_4e53,
            persist_model: true,
        }
    }
}

impl KmeansOptions {
    pub fn validate(&self) -> rsod_core::Result<()> {
        if self.train_window_secs <= 0 || self.retrain_every_secs <= 0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "trainWindowSecs and retrainEverySecs must be > 0".into(),
            ));
        }
        if self.max_models == 0 || self.min_models_for_alert == 0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "maxModels and minModelsForAlert must be > 0".into(),
            ));
        }
        if self.min_models_for_alert > self.max_models {
            return Err(rsod_core::RsodError::InvalidConfig(
                "minModelsForAlert must be <= maxModels".into(),
            ));
        }
        if !(0.0..=100.0).contains(&self.anomaly_threshold) {
            return Err(rsod_core::RsodError::InvalidConfig(
                "anomalyThreshold must be between 0 and 100".into(),
            ));
        }
        if self.min_train_vectors < 2 || self.max_train_vectors < self.min_train_vectors {
            return Err(rsod_core::RsodError::InvalidConfig(
                "training vector limits are inconsistent".into(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(rsod_core::RsodError::InvalidConfig(
                "maxIterations must be > 0".into(),
            ));
        }
        Ok(())
    }

    pub fn history_duration_ms(&self, _interval_ms: i64) -> i64 {
        let coverage_secs = self.train_window_secs
            + self.retrain_every_secs * self.max_models.saturating_sub(1) as i64;
        // Add one alignment period because the newest training window ends on
        // the prior retrain grid boundary, not necessarily at panel start.
        (coverage_secs + self.retrain_every_secs) * 1000
    }
}
