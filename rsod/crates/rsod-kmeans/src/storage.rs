use rsod_storage::model::Model;

use crate::ensemble::KmeansEnsemble;

fn legacy_storage_key(uuid: &str) -> String {
    format!("kmeans:v1:{uuid}")
}

fn snapshot_storage_key(uuid: &str, anchor: i64) -> String {
    format!("kmeans:v1:{uuid}:anchor:{anchor}")
}

fn read_ensemble(key: String) -> Option<KmeansEnsemble> {
    let mut model = Model::new(key, Vec::new());
    model.read().ok()?;
    if model.artifacts.is_empty() {
        return None;
    }
    serde_json::from_slice(&model.artifacts).ok()
}

pub fn load_ensemble(uuid: &str, anchor: i64) -> Option<KmeansEnsemble> {
    if uuid.is_empty() {
        return None;
    }
    read_ensemble(snapshot_storage_key(uuid, anchor))
        // Compatibility with ensembles written before snapshots were keyed by
        // window anchor. The caller still validates the loaded anchor.
        .or_else(|| read_ensemble(legacy_storage_key(uuid)))
}

pub fn save_ensemble(uuid: &str, anchor: i64, ensemble: &KmeansEnsemble) -> rsod_core::Result<()> {
    if uuid.is_empty() {
        return Ok(());
    }
    let artifacts = serde_json::to_vec(ensemble)
        .map_err(|e| rsod_core::RsodError::Storage(format!("serialize K-means model: {e}")))?;
    // Each window is an immutable snapshot. Publishing a newer anchor never
    // overwrites the previous one, so concurrent readers see complete blobs
    // and a rollback/restart can still load the exact requested window.
    Model::new(snapshot_storage_key(uuid, anchor), artifacts).write()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::KmeansModel;

    #[test]
    fn persisted_ensemble_round_trips_under_kmeans_prefix() {
        let _ = rsod_storage::init_db_with_config(true, "");
        let key = "kmeans-storage-round-trip";
        let ensemble = KmeansEnsemble {
            version: 1,
            interval_ms: 60_000,
            models: vec![KmeansModel {
                window_start: 0,
                window_end: 60,
                centers: [[0.0; 6], [1.0; 6]],
                min_distance: 0.0,
                max_distance: 1.0,
            }],
        };
        save_ensemble(key, 60, &ensemble).unwrap();
        assert_eq!(load_ensemble(key, 60), Some(ensemble));
    }

    #[test]
    fn snapshots_for_previous_anchors_are_retained() {
        let _ = rsod_storage::init_db_with_config(true, "");
        let key = "kmeans-storage-snapshots";
        let mut first = KmeansEnsemble {
            version: 1,
            interval_ms: 60_000,
            models: vec![],
        };
        let mut second = first.clone();
        first.models.push(KmeansModel {
            window_start: 0,
            window_end: 60,
            centers: [[0.0; 6], [1.0; 6]],
            min_distance: 0.0,
            max_distance: 1.0,
        });
        second.models.push(KmeansModel {
            window_start: 60,
            window_end: 120,
            centers: [[2.0; 6], [3.0; 6]],
            min_distance: 0.0,
            max_distance: 1.0,
        });
        save_ensemble(key, 60, &first).unwrap();
        save_ensemble(key, 120, &second).unwrap();
        assert_eq!(load_ensemble(key, 60), Some(first));
        assert_eq!(load_ensemble(key, 120), Some(second));
    }
}
