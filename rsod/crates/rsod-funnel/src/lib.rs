//! Funnel anomaly detection: L1 statistical pre-filter + L2 ML escalation.
//!
//! ## Architecture
//!
//! 1. **Profile** — seasonal bucket statistics built from history (offline / cold-start).
//! 2. **L1** — O(1) per-point three-state filter (Normal / Uncertain / Anomaly).
//! 3. **L2** — optional ML detectors for uncertain points only.

mod alert_output;
mod config;
mod engine;
mod l1;
mod l2;
mod metrics;
mod model;
mod pipeline;
mod profile;
mod stats;
mod storage;

pub use config::{AlertOutputMode, FunnelOptions};
pub use engine::{force_link, parse_funnel_hyper_params, FunnelEngine, FunnelHyperParams};
pub use l1::{FilterVerdict, L1Result, L1Stats};
pub use metrics::{FunnelMetrics, FunnelRun};
pub use model::{FunnelModel, FUNNEL_MODEL_VERSION};
pub use pipeline::{eval_window_start, funnel_detect, funnel_detect_with_metrics};
pub use profile::{compute_trend, SeasonalProfile};
pub use stats::{BucketStat, ThresholdMethod};
pub use storage::{delete_funnel_model, load_funnel_model, save_funnel_model};
