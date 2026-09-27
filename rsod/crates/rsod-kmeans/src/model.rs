use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::FEATURE_WIDTH;
use crate::features::FeatureVector;

const SCORE_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KmeansModel {
    pub window_start: i64,
    pub window_end: i64,
    pub centers: [FeatureVector; 2],
    pub min_distance: f64,
    pub max_distance: f64,
}

impl KmeansModel {
    pub fn train(
        vectors: &[FeatureVector],
        window_start: i64,
        window_end: i64,
        seed: u64,
        max_iterations: usize,
    ) -> rsod_core::Result<Self> {
        Self::train_with_cancel(
            vectors,
            window_start,
            window_end,
            seed,
            max_iterations,
            None,
        )
    }

    pub fn train_with_cancel(
        vectors: &[FeatureVector],
        window_start: i64,
        window_end: i64,
        seed: u64,
        max_iterations: usize,
        cancelled: Option<&AtomicBool>,
    ) -> rsod_core::Result<Self> {
        if vectors.len() < 2 {
            return Err(rsod_core::RsodError::InsufficientData {
                need: 2,
                got: vectors.len(),
            });
        }
        let mut centers = initialize_centers(vectors, seed);
        for _ in 0..max_iterations {
            if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Err(rsod_core::RsodError::Cancelled);
            }
            let mut sums = [[0.0; FEATURE_WIDTH]; 2];
            let mut counts = [0usize; 2];
            for vector in vectors {
                let cluster = nearest_center(vector, &centers);
                counts[cluster] += 1;
                for (sum, value) in sums[cluster].iter_mut().zip(vector) {
                    *sum += *value;
                }
            }
            let mut next = centers;
            for cluster in 0..2 {
                if counts[cluster] == 0 {
                    next[cluster] = farthest_vector(vectors, &centers[1 - cluster]);
                } else {
                    for (value, sum) in next[cluster].iter_mut().zip(sums[cluster]) {
                        *value = sum / counts[cluster] as f64;
                    }
                }
            }
            let movement = euclidean(&centers[0], &next[0]) + euclidean(&centers[1], &next[1]);
            centers = next;
            if movement <= SCORE_EPSILON {
                break;
            }
        }

        let mut min_distance = f64::INFINITY;
        let mut max_distance = f64::NEG_INFINITY;
        for vector in vectors {
            let distance = mean_center_distance(vector, &centers);
            min_distance = min_distance.min(distance);
            max_distance = max_distance.max(distance);
        }
        Ok(Self {
            window_start,
            window_end,
            centers,
            min_distance,
            max_distance,
        })
    }

    pub fn score(&self, vector: &FeatureVector) -> f64 {
        let distance = mean_center_distance(vector, &self.centers);
        let range = self.max_distance - self.min_distance;
        if range.abs() <= SCORE_EPSILON * self.max_distance.abs().max(1.0) {
            if (distance - self.min_distance).abs()
                <= SCORE_EPSILON * self.min_distance.abs().max(1.0)
            {
                0.0
            } else {
                100.0
            }
        } else {
            (100.0 * ((distance - self.min_distance) / range).abs()).min(100.0)
        }
    }
}

fn initialize_centers(vectors: &[FeatureVector], seed: u64) -> [FeatureVector; 2] {
    let mut rng = StdRng::seed_from_u64(seed);
    let first_index = rng.gen_range(0..vectors.len());
    let first = vectors[first_index];
    let weights: Vec<f64> = vectors
        .iter()
        .map(|v| euclidean(v, &first).powi(2))
        .collect();
    let total: f64 = weights.iter().sum();
    let second_index = if total <= SCORE_EPSILON {
        (first_index + vectors.len() / 2).min(vectors.len() - 1)
    } else {
        let mut target = rng.gen::<f64>() * total;
        let mut selected = vectors.len() - 1;
        for (index, weight) in weights.iter().enumerate() {
            target -= *weight;
            if target <= 0.0 {
                selected = index;
                break;
            }
        }
        selected
    };
    [first, vectors[second_index]]
}

fn nearest_center(vector: &FeatureVector, centers: &[FeatureVector; 2]) -> usize {
    if euclidean(vector, &centers[0]) <= euclidean(vector, &centers[1]) {
        0
    } else {
        1
    }
}

fn farthest_vector(vectors: &[FeatureVector], center: &FeatureVector) -> FeatureVector {
    *vectors
        .iter()
        .max_by(|a, b| {
            euclidean(a, center)
                .partial_cmp(&euclidean(b, center))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(center)
}

fn mean_center_distance(vector: &FeatureVector, centers: &[FeatureVector; 2]) -> f64 {
    (euclidean(vector, &centers[0]) + euclidean(vector, &centers[1])) / 2.0
}

fn euclidean(a: &FeatureVector, b: &FeatureVector) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_is_deterministic() {
        let vectors = vec![[0.0; 6], [1.0; 6], [10.0; 6], [11.0; 6]];
        let a = KmeansModel::train(&vectors, 0, 1, 7, 100).unwrap();
        let b = KmeansModel::train(&vectors, 0, 1, 7, 100).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn degenerate_training_range_distinguishes_a_change() {
        let vectors = vec![[1.0; 6], [1.0; 6]];
        let model = KmeansModel::train(&vectors, 0, 1, 7, 100).unwrap();
        assert_eq!(model.score(&[1.0; 6]), 0.0);
        assert_eq!(model.score(&[2.0; 6]), 100.0);
    }

    #[test]
    fn constant_raw_series_has_explainable_degenerate_scores() {
        let timestamps: Vec<f64> = (0..12).map(|i| i as f64 * 60.0).collect();
        let values = vec![42.0; timestamps.len()];
        let vectors: Vec<_> = crate::features::build_features(&timestamps, &values, 60_000)
            .unwrap()
            .into_iter()
            .flatten()
            .map(|point| point.values)
            .collect();

        assert_eq!(vectors, vec![[0.0; 6]; 4]);
        let model = KmeansModel::train(&vectors, 0, 1, 7, 100).unwrap();
        assert_eq!(model.min_distance, model.max_distance);
        assert_eq!(model.score(&[0.0; 6]), 0.0);
        // Unlike the pinned Netdata zero-range behavior, this is the project's
        // explicit adaptation: a non-matching vector scores as anomalous.
        assert_eq!(model.score(&[1.0; 6]), 100.0);
    }

    #[test]
    fn training_observes_cancellation_at_iteration_boundary() {
        let cancelled = AtomicBool::new(true);
        let err =
            KmeansModel::train_with_cancel(&[[0.0; 6], [1.0; 6]], 0, 1, 7, 100, Some(&cancelled))
                .unwrap_err();
        assert_eq!(err, rsod_core::RsodError::Cancelled);
    }
}
