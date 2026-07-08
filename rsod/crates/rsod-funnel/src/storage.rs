use rsod_core::ModelStorage;
use rsod_storage::model::Model;
use rsod_storage::GlobalStorage;

use crate::model::FunnelModel;
use crate::profile::SeasonalProfile;

const PROFILE_PREFIX: &str = "funnel_profile:";

fn storage_key(uuid: &str) -> String {
    format!("{}{}", PROFILE_PREFIX, uuid)
}

/// Load a persisted funnel model (L1 profile + version) from backend storage.
///
/// Falls back to legacy bare-profile blobs via [`FunnelModel::from_bytes`].
pub fn load_funnel_model(uuid: &str) -> Option<FunnelModel> {
    if uuid.is_empty() {
        return None;
    }
    let mut model = Model::new(storage_key(uuid), vec![]);
    if model.read().is_err() || model.artifacts.is_empty() {
        return None;
    }
    FunnelModel::from_bytes(&model.artifacts).ok()
}

/// Persist a funnel model to backend storage.
pub fn save_funnel_model(uuid: &str, funnel: &FunnelModel) -> rsod_core::Result<()> {
    if uuid.is_empty() {
        return Ok(());
    }
    let bytes = funnel.to_bytes()?;
    let model = Model::new(storage_key(uuid), bytes);
    model
        .write()
        .map_err(|e| rsod_core::RsodError::Storage(e.to_string()))
}

/// Delete a persisted funnel model from backend storage.
pub fn delete_funnel_model(uuid: &str) -> rsod_core::Result<()> {
    if uuid.is_empty() {
        return Ok(());
    }
    GlobalStorage::new().delete(&storage_key(uuid))
}

/// Load just the L1 seasonal profile (unwraps the persisted funnel model).
pub fn load_profile(uuid: &str) -> Option<SeasonalProfile> {
    load_funnel_model(uuid).map(|m| m.profile)
}

/// Persist the L1 seasonal profile inside a versioned funnel model.
pub fn save_profile(uuid: &str, profile: &SeasonalProfile) -> rsod_core::Result<()> {
    save_funnel_model(uuid, &FunnelModel::new(profile.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FunnelOptions;
    use crate::profile::build_profile;
    use rsod_core::TrendType;
    use rsod_storage::init_db_with_config;

    fn init_storage() {
        let _ = init_db_with_config(true, "");
    }

    fn sample_model() -> FunnelModel {
        let ts: Vec<f64> = (0..96).map(|i| (1_700_000_000 + i * 3600) as f64).collect();
        let vals: Vec<f64> = (0..96).map(|i| 100.0 + (i % 24) as f64).collect();
        let opts = FunnelOptions {
            trend: Some(TrendType::Daily),
            ..Default::default()
        };
        FunnelModel::new(build_profile(&ts, &vals, &opts))
    }

    #[test]
    fn save_then_load_funnel_model_round_trips() {
        init_storage();
        let uuid = "funnel_model_round_trip";
        let model = sample_model();

        save_funnel_model(uuid, &model).unwrap();
        let restored = load_funnel_model(uuid).expect("model must load");

        assert_eq!(restored, model);
    }

    #[test]
    fn delete_funnel_model_removes_persisted_state() {
        init_storage();
        let uuid = "funnel_model_delete";
        save_funnel_model(uuid, &sample_model()).unwrap();
        assert!(load_funnel_model(uuid).is_some());

        delete_funnel_model(uuid).unwrap();
        assert!(load_funnel_model(uuid).is_none());
    }

    #[test]
    fn load_funnel_model_reads_legacy_bare_profile() {
        init_storage();
        let uuid = "funnel_model_legacy";
        let profile = sample_model().profile;

        // Write a legacy bare-profile blob directly (pre-FunnelModel format).
        let legacy_bytes = serde_json::to_vec(&profile).unwrap();
        Model::new(storage_key(uuid), legacy_bytes).write().unwrap();

        let restored = load_funnel_model(uuid).expect("legacy blob must load");
        assert_eq!(restored.version, 0);
        assert_eq!(restored.profile, profile);
    }
}
