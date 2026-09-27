use serde::{Deserialize, Serialize};

use crate::features::FeatureVector;
use crate::model::KmeansModel;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KmeansEnsemble {
    pub version: u32,
    pub interval_ms: i64,
    pub models: Vec<KmeansModel>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsembleScore {
    pub score: f64,
    pub anomaly: bool,
}

impl KmeansEnsemble {
    pub fn score(&self, vector: &FeatureVector, threshold: f64) -> Option<EnsembleScore> {
        if self.models.is_empty() {
            return None;
        }
        let score = self
            .models
            .iter()
            .map(|model| model.score(vector))
            .fold(f64::INFINITY, f64::min);
        Some(EnsembleScore {
            score,
            anomaly: score >= threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_score_is_equivalent_to_unanimous_consensus() {
        let normal = KmeansModel {
            window_start: 0,
            window_end: 1,
            centers: [[0.0; 6], [0.0; 6]],
            min_distance: 0.0,
            max_distance: 1.0,
        };
        let anomalous = KmeansModel {
            window_start: 0,
            window_end: 1,
            centers: [[10.0; 6], [10.0; 6]],
            min_distance: 0.0,
            max_distance: 1.0,
        };
        let ensemble = KmeansEnsemble {
            version: 1,
            interval_ms: 60_000,
            models: vec![normal, anomalous],
        };
        let score = ensemble.score(&[0.0; 6], 99.0).unwrap();
        assert!(!score.anomaly);
        assert_eq!(score.score, 0.0);
    }

    #[test]
    fn threshold_is_inclusive_at_99() {
        let model = KmeansModel {
            window_start: 0,
            window_end: 1,
            centers: [[0.0; 6], [0.0; 6]],
            min_distance: 0.0,
            max_distance: 1.0,
        };
        let ensemble = KmeansEnsemble {
            version: 1,
            interval_ms: 60_000,
            models: vec![model],
        };
        assert!(
            ensemble
                .score(&[0.99, 0.0, 0.0, 0.0, 0.0, 0.0], 99.0)
                .unwrap()
                .anomaly
        );
        assert!(
            !ensemble
                .score(&[0.989, 0.0, 0.0, 0.0, 0.0, 0.0], 99.0)
                .unwrap()
                .anomaly
        );
    }

    #[test]
    fn all_eighteen_models_must_vote_anomalous() {
        let anomalous = KmeansModel {
            window_start: 0,
            window_end: 1,
            centers: [[0.0; 6], [0.0; 6]],
            min_distance: 0.0,
            max_distance: 1.0,
        };
        let normal = KmeansModel {
            window_start: 0,
            window_end: 1,
            centers: [[0.99, 0.0, 0.0, 0.0, 0.0, 0.0]; 2],
            min_distance: 0.0,
            max_distance: 1.0,
        };
        let vector = [0.99, 0.0, 0.0, 0.0, 0.0, 0.0];

        let unanimous = KmeansEnsemble {
            version: 1,
            interval_ms: 60_000,
            models: vec![anomalous.clone(); 18],
        };
        let unanimous_score = unanimous.score(&vector, 99.0).unwrap();
        assert_eq!(unanimous_score.score, 99.0);
        assert!(unanimous_score.anomaly);

        let mut divided_models = vec![anomalous; 17];
        divided_models.push(normal);
        let divided = KmeansEnsemble {
            version: 1,
            interval_ms: 60_000,
            models: divided_models,
        };
        let divided_score = divided.score(&vector, 99.0).unwrap();
        assert_eq!(divided_score.score, 0.0);
        assert!(!divided_score.anomaly);
    }
}
