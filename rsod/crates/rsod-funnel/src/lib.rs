//! Funnel anomaly detection: L1 statistical pre-filter + L2 ML escalation.
//!
//! ## Architecture
//!
//! 1. **Profile** — seasonal bucket statistics built from history (offline / cold-start).
//! 2. **L1** — O(1) per-point three-state filter (Normal / Uncertain / Anomaly).
//! 3. **L2** — optional ML detectors for uncertain points only.

mod alert_output;
mod config;
mod l1;
mod l2;
mod pipeline;
mod profile;
mod stats;
mod storage;

pub use config::{AlertOutputMode, FunnelOptions};
pub use l1::{FilterVerdict, L1Result, L1Stats};
pub use pipeline::{eval_window_start, funnel_detect};
pub use profile::{SeasonalProfile, compute_trend};
pub use stats::{BucketStat, ThresholdMethod};
