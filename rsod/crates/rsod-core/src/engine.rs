//! Pluggable anomaly-detection engine interface.
//!
//! Each detect type (`outlier`, `dynamics`, `forecast`, `funnel`) is an
//! engine implementing [`Detector`]. Engines self-register via `inventory`,
//! so the backend orchestrates purely by `detector_by_name(detect_type)`
//! and never references a concrete algorithm's options type — replacing a
//! model means adding a new engine crate + registering it, nothing more.

use serde_json::Value;

use crate::config::DetectionMethod;
use crate::types::{DetectionResult, TimeSeriesInput};

/// How the engine consumes the upstream frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// The whole upstream frame is the evaluation window (no history split).
    WholeFrame,
    /// The frame is split into `current` (eval window) and `history` (training).
    HistoryCurrent,
}

/// How the backend must fetch upstream data for this engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryKind {
    /// Single extended-range query; `split_frames` produces current/history.
    Single,
    /// Funnel dual query (profile history + panel current).
    FunnelDual,
}

/// Data + parameters handed to an engine for one series evaluation.
#[derive(Debug)]
pub struct DetectRequest<'a> {
    /// Evaluation-window series (epoch seconds).
    pub current: TimeSeriesInput<'a>,
    /// History/training series (epoch seconds); empty for [`InputKind::WholeFrame`].
    pub history: TimeSeriesInput<'a>,
    /// Raw frontend `hyperParams` JSON, parsed and defaulted by the engine.
    pub hyper_params: Value,
    /// Deterministic base model key (UUID v5) computed by the backend.
    pub uuid: String,
    /// Query interval in milliseconds (for `periods` string → point count).
    pub interval_ms: i64,
}

/// A completed detection plus the method that produced it (drives rendering).
#[derive(Debug, Clone)]
pub struct DetectOutput {
    pub result: DetectionResult,
    pub method: DetectionMethod,
}

/// A pluggable, stateless anomaly-detection engine.
pub trait Detector: Send + Sync {
    /// Stable registry key (`"outlier"`, `"dynamics"`, `"forecast"`, `"funnel"`).
    fn name(&self) -> &'static str;

    /// How the engine consumes upstream data.
    fn input_kind(&self) -> InputKind;

    /// How the backend must fetch upstream data for this engine.
    fn query_kind(&self) -> QueryKind {
        QueryKind::Single
    }

    /// Default history lookback in milliseconds when the panel omits
    /// `historyTimeRange` (or sends `durationMs: 0`). Return `0` to leave the
    /// unset range unchanged — the backend must not hard-code detect-type
    /// names for this.
    fn default_history_duration_ms(&self) -> i64 {
        0
    }

    /// Run detection for one series.
    fn detect(&self, req: &DetectRequest) -> crate::Result<DetectOutput>;
}

inventory::collect!(&'static dyn Detector);

/// Iterate every self-registered engine (order is unspecified).
pub fn iter_detectors() -> impl Iterator<Item = &'static dyn Detector> {
    inventory::iter::<&'static dyn Detector>
        .into_iter()
        .copied()
}

/// Look up an engine by its detect-type name.
pub fn detector_by_name(name: &str) -> Option<&'static dyn Detector> {
    iter_detectors().find(|d| d.name() == name)
}
