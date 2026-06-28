use rsod_storage::model::Model;

use crate::profile::SeasonalProfile;

const PROFILE_PREFIX: &str = "funnel_profile:";

fn storage_key(uuid: &str) -> String {
    format!("{}{}", PROFILE_PREFIX, uuid)
}

/// Load a persisted seasonal profile.
pub fn load_profile(uuid: &str) -> Option<SeasonalProfile> {
    if uuid.is_empty() {
        return None;
    }
    let mut model = Model::new(storage_key(uuid), vec![]);
    if model.read().is_err() || model.artifacts.is_empty() {
        return None;
    }
    serde_json::from_slice(&model.artifacts).ok()
}

/// Persist seasonal profile.
pub fn save_profile(uuid: &str, profile: &SeasonalProfile) -> rsod_core::Result<()> {
    if uuid.is_empty() {
        return Ok(());
    }
    let bytes = serde_json::to_vec(profile)
        .map_err(|e| rsod_core::RsodError::Storage(e.to_string()))?;
    let model = Model::new(storage_key(uuid), bytes);
    model
        .write()
        .map_err(|e| rsod_core::RsodError::Storage(e.to_string()))
}
