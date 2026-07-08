use serde::{Deserialize, Serialize};

use crate::profile::SeasonalProfile;

/// On-disk layout version for a persisted [`FunnelModel`].
///
/// Bump when the serialized structure changes in a backward-incompatible way.
pub const FUNNEL_MODEL_VERSION: u32 = 1;

/// Serializable snapshot of a trained funnel detector.
///
/// The funnel is a combined **L1** (statistical pre-filter) + **L2** (ML escalation)
/// detector. The L2 sub-models (outlier / forecaster / baseline) persist themselves
/// under their own UUIDs, so a `FunnelModel` captures the L1 seasonal profile plus a
/// version tag. Reloading a `FunnelModel` restores L1 behaviour without recomputing
/// the profile from history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunnelModel {
    /// On-disk format version.
    #[serde(default)]
    pub version: u32,
    /// L1 seasonal profile (trained statistical state).
    pub profile: SeasonalProfile,
}

impl FunnelModel {
    /// Wrap a seasonal profile into a versioned funnel model.
    pub fn new(profile: SeasonalProfile) -> Self {
        Self {
            version: FUNNEL_MODEL_VERSION,
            profile,
        }
    }

    /// Serialize the model to JSON bytes for backend storage.
    pub fn to_bytes(&self) -> rsod_core::Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| rsod_core::RsodError::Storage(e.to_string()))
    }

    /// Deserialize a model from JSON bytes.
    ///
    /// Falls back to parsing a legacy bare [`SeasonalProfile`] blob (persisted before
    /// `FunnelModel` existed) and wraps it with `version = 0`.
    pub fn from_bytes(bytes: &[u8]) -> rsod_core::Result<Self> {
        if let Ok(model) = serde_json::from_slice::<FunnelModel>(bytes) {
            return Ok(model);
        }
        let profile: SeasonalProfile = serde_json::from_slice(bytes)
            .map_err(|e| rsod_core::RsodError::Storage(e.to_string()))?;
        Ok(Self {
            version: 0,
            profile,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FunnelOptions;
    use crate::profile::build_profile;
    use rsod_core::TrendType;

    fn sample_profile() -> SeasonalProfile {
        let ts: Vec<f64> = (0..96).map(|i| (1_700_000_000 + i * 3600) as f64).collect();
        let vals: Vec<f64> = (0..96).map(|i| 100.0 + (i % 24) as f64).collect();
        let opts = FunnelOptions {
            trend: Some(TrendType::Daily),
            ..Default::default()
        };
        build_profile(&ts, &vals, &opts)
    }

    #[test]
    fn model_bytes_round_trip_preserves_profile() {
        let model = FunnelModel::new(sample_profile());
        let bytes = model.to_bytes().unwrap();
        let restored = FunnelModel::from_bytes(&bytes).unwrap();
        assert_eq!(restored.version, FUNNEL_MODEL_VERSION);
        assert_eq!(restored, model);
    }

    #[test]
    fn from_bytes_accepts_legacy_bare_profile() {
        let profile = sample_profile();
        let legacy_bytes = serde_json::to_vec(&profile).unwrap();
        let restored = FunnelModel::from_bytes(&legacy_bytes).unwrap();
        assert_eq!(restored.version, 0, "legacy blobs decode as version 0");
        assert_eq!(restored.profile, profile);
    }
}
