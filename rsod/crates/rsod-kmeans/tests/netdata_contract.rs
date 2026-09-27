use rsod_kmeans::{build_features, FeatureVector, KmeansEnsemble, KmeansModel};
use serde::Deserialize;

const REFERENCE_COMMIT: &str = "4464a9ed71c8a7351b540ce4ca207e1418faffea";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Fixture {
    reference: Reference,
    feature: FeatureFixture,
    score: ScoreFixture,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Reference {
    repository: String,
    commit: String,
    feature_source: String,
    score_source: String,
    feature_contract: String,
    score_contract: String,
    initialization_compatibility: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FeatureFixture {
    interval_ms: i64,
    timestamps: Vec<f64>,
    values: Vec<f64>,
    expected: Vec<ExpectedFeature>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExpectedFeature {
    source_index: usize,
    timestamp: f64,
    vector: FeatureVector,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScoreFixture {
    model: KmeansModel,
    threshold: f64,
    cases: Vec<ScoreCase>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScoreCase {
    vector: FeatureVector,
    expected_score: f64,
    expected_anomaly: bool,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/netdata-4464a9ed-contract.json"))
        .expect("the checked-in Netdata contract fixture must remain valid")
}

fn assert_approx_eq(actual: f64, expected: f64) {
    let tolerance = 1e-10 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn fixture_pins_the_audited_netdata_contract_not_center_initialization() {
    let reference = fixture().reference;
    assert_eq!(reference.repository, "https://github.com/netdata/netdata");
    assert_eq!(reference.commit, REFERENCE_COMMIT);
    assert_eq!(reference.commit.len(), 40);
    assert_eq!(reference.feature_source, "src/ml/ml_features.cc");
    assert_eq!(reference.score_source, "src/ml/ml_kmeans.cc");
    assert_eq!(reference.feature_contract, "diff(1) -> smooth(3) -> lag(5)");
    assert!(reference.score_contract.contains("mean_center_distance"));
    assert!(reference
        .initialization_compatibility
        .contains("contract-only"));
    assert!(reference
        .initialization_compatibility
        .contains("seeded k-means++"));
    assert!(reference
        .initialization_compatibility
        .contains("dlib pick_initial_centers"));
}

#[test]
fn feature_transform_matches_the_pinned_netdata_formula() {
    let feature = fixture().feature;
    let actual = build_features(&feature.timestamps, &feature.values, feature.interval_ms).unwrap();

    assert!(actual[..8].iter().all(Option::is_none));
    let emitted: Vec<_> = actual.into_iter().flatten().collect();
    assert_eq!(emitted.len(), feature.expected.len());
    for (actual, expected) in emitted.iter().zip(feature.expected) {
        assert_eq!(actual.source_index, expected.source_index);
        assert_eq!(actual.timestamp, expected.timestamp);
        assert_eq!(actual.values, expected.vector);
    }
}

#[test]
fn score_and_inclusive_threshold_match_the_pinned_netdata_formula() {
    let score = fixture().score;
    let ensemble = KmeansEnsemble {
        version: 1,
        interval_ms: 60_000,
        models: vec![score.model],
    };

    for case in score.cases {
        let actual = ensemble.score(&case.vector, score.threshold).unwrap();
        assert_approx_eq(actual.score, case.expected_score);
        assert_eq!(actual.anomaly, case.expected_anomaly);
    }
}
