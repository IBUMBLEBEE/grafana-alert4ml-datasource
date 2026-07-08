use rsod_core::{DetectionMethod, DetectionResult};

/// Runtime metrics that make the funnel split measurable.
#[derive(Debug, Clone)]
pub struct FunnelMetrics {
    pub total_points: usize,
    pub l1_normal: usize,
    pub l1_anomaly: usize,
    pub l1_uncertain: usize,
    pub l1_coverage_rate: f64,
    pub l2_escalation_rate: f64,
    pub l2_enabled: bool,
    pub l2_triggered: bool,
    pub l2_method: Option<DetectionMethod>,
    pub l1_elapsed_ms: u128,
    pub l2_elapsed_ms: u128,
    pub total_elapsed_ms: u128,
}

/// Detection result plus funnel split metrics.
#[derive(Debug, Clone)]
pub struct FunnelRun {
    pub result: DetectionResult,
    pub metrics: FunnelMetrics,
}
