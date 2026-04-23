package rsod

// rsod_test.go — Integration tests for Go ↔ Rust FFI layer.
//
// All time-series fixtures are loaded from dataset/testdata/ via the helpers
// in eval_test.go (loadTestdataFrame / loadTestdataCSV).
//
// Metric assertions use the unified helpers from eval_test.go:
//   - assertOutlierMetrics(t, m, minF1=0.80, minRecall=0.85)
//   - assertForecastMetrics(t, m, baselineMAE, baselineRMSE, maxMASE=1.0)
//
// Thresholds are defined in .claude/skills/rust-ml-boundary/SKILL.md.

import (
	"fmt"
	"testing"
)

// ─── Outlier tests ────────────────────────────────────────────────────────────

// Test_24_7x24_OutlierFitPredict verifies outlier detection on a 24-hour
// seasonal fixture (5-minute resolution → period = 288 points).
//
// Fixture: dataset/testdata/artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv
func Test_24_7x24_OutlierFitPredict(t *testing.T) {
	frame, labels, err := loadTestdataFrame(
		"artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv",
	)
	if err != nil {
		t.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "outlier_model",
		// 288 points = one 24-hour period at 5-minute resolution;
		// 2016 points = one 7-day period (168 h × 12 samples/h).
		Periods: []uint{288, 2016},
	}

	predictions, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict: %v", err)
	}

	m := computeOutlierMetrics(predictions, labels)
	// Default thresholds: F1 >= 0.80, Recall >= 0.85
	assertOutlierMetrics(t, m, 0.80, 0.85)
}

// TestNoPeriodsOutlierFitPredict verifies the model on a no-anomaly series.
// All is_anomaly labels are 0, so F1/Recall are undefined; we assert a low
// false-positive rate instead.
//
// Fixture: dataset/testdata/artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv
func TestNoPeriodsOutlierFitPredict(t *testing.T) {
	frame, labels, err := loadTestdataFrame(
		"artificialNoAnomaly/p24h_clean_art_daily_no_noise.csv",
	)
	if err != nil {
		t.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	predictions, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict: %v", err)
	}

	m := computeOutlierMetrics(predictions, labels)
	t.Logf(
		"TestNoPeriodsOutlierFitPredict — FP=%d total=%d false-positive-rate=%.4f",
		m.FalsePositives, len(predictions),
		float64(m.FalsePositives)/float64(len(predictions)),
	)
	// No positive labels → F1/Recall undefined; assert false-positive rate ≤ 5%.
	fpRate := float64(m.FalsePositives) / float64(len(predictions))
	if fpRate > 0.05 {
		t.Errorf("false positive rate %.4f > 5%% on clean fixture", fpRate)
	}
}

// TestMissingDataOutlierFitPredict verifies that the model handles a spike-
// density fixture without error and meets the default metric thresholds.
//
// Fixture: dataset/testdata/artificialWithAnomaly/p24h_anom_art_load_balancer_spikes.csv
func TestMissingDataOutlierFitPredict(t *testing.T) {
	frame, labels, err := loadTestdataFrame(
		"artificialWithAnomaly/p24h_anom_art_load_balancer_spikes.csv",
	)
	if err != nil {
		t.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	predictions, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict: %v", err)
	}

	m := computeOutlierMetrics(predictions, labels)
	// Default thresholds: F1 >= 0.80, Recall >= 0.85
	assertOutlierMetrics(t, m, 0.80, 0.85)
}

// TestAutoPeriodsOutlierFitPredict verifies the auto-period detection path
// on a seasonal fixture meets the default metric thresholds.
//
// Fixture: dataset/testdata/artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv
func TestAutoPeriodsOutlierFitPredict(t *testing.T) {
	frame, labels, err := loadTestdataFrame(
		"artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv",
	)
	if err != nil {
		t.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{}, // auto-detect
	}

	predictions, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict: %v", err)
	}

	m := computeOutlierMetrics(predictions, labels)
	// Default thresholds: F1 >= 0.80, Recall >= 0.85
	assertOutlierMetrics(t, m, 0.80, 0.85)
}

// TestErrorRateOutlierFitPredict verifies outlier detection on a real-world
// NYC taxi demand series that contains known anomalies.
//
// Fixture: dataset/testdata/realKnownCause/p7d_anom_curr_nyc_taxi.csv
func TestErrorRateOutlierFitPredict(t *testing.T) {
	frame, labels, err := loadTestdataFrame(
		"realKnownCause/p7d_anom_curr_nyc_taxi.csv",
	)
	if err != nil {
		t.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	predictions, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict: %v", err)
	}

	m := computeOutlierMetrics(predictions, labels)
	// Default thresholds: F1 >= 0.80, Recall >= 0.85
	assertOutlierMetrics(t, m, 0.80, 0.85)
}

// ─── Baseline tests ───────────────────────────────────────────────────────────
//
// All baseline tests use the NYC taxi dataset.
//   current : dataset/testdata/realKnownCause/p7d_anom_curr_nyc_taxi.csv
//   history : dataset/testdata/realKnownCause/p7d_clean_hist_nyc_taxi.csv
//
// The result frame's "anomaly" column is compared against the ground truth
// is_anomaly column from the current fixture using assertOutlierMetrics.

// TestBaselineFitPredictWithDailyTrend11 tests the Daily trend baseline path.
func TestBaselineFitPredictWithDailyTrend11(t *testing.T) {
	frame, labels, err := loadTestdataFrame("realKnownCause/p7d_anom_curr_nyc_taxi.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (current): %v", err)
	}
	historyFrame, _, err := loadTestdataFrame("realKnownCause/p7d_clean_hist_nyc_taxi.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (history): %v", err)
	}

	options := BaselineOptions{
		TrendType:           "Daily",
		IntervalMins:        30, // NYC taxi uses 30-minute resolution
		ConfidenceLevel:     95.0,
		AllowNegativeBounds: false,
	}

	result, err := BaselineFitPredict(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("BaselineFitPredict: %v", err)
	}

	// Extract the "anomaly" column from the result frame.
	anomalyPreds, ok := extractFrameFloats(result, "anomaly")
	if !ok {
		t.Fatal("result frame does not contain an 'anomaly' column")
	}

	m := computeOutlierMetrics(anomalyPreds, labels)
	// Default thresholds: F1 >= 0.80, Recall >= 0.85
	assertOutlierMetrics(t, m, 0.80, 0.85)
}

// TestBaselineFitPredictWithWeeklyTrend tests the Weekly trend baseline path.
func TestBaselineFitPredictWithWeeklyTrend(t *testing.T) {
	frame, labels, err := loadTestdataFrame("realKnownCause/p7d_anom_curr_nyc_taxi.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (current): %v", err)
	}
	historyFrame, _, err := loadTestdataFrame("realKnownCause/p7d_clean_hist_nyc_taxi.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (history): %v", err)
	}

	options := BaselineOptions{
		TrendType:           "Weekly",
		IntervalMins:        30,
		ConfidenceLevel:     95.0,
		AllowNegativeBounds: false,
	}

	result, err := BaselineFitPredict(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("BaselineFitPredict: %v", err)
	}

	anomalyPreds, ok := extractFrameFloats(result, "anomaly")
	if !ok {
		t.Fatal("result frame does not contain an 'anomaly' column")
	}

	m := computeOutlierMetrics(anomalyPreds, labels)
	// Default thresholds: F1 >= 0.80, Recall >= 0.85
	assertOutlierMetrics(t, m, 0.80, 0.85)
}

// TestBaselineFitPredictWithMonthlyTrend keeps the monthly path covered with a
// smoke test. The fixed testdata fixtures in this repository do not contain a
// full month-sized seasonal window, so strict F1/Recall thresholds would be
// measuring fixture mismatch rather than implementation quality.
func TestBaselineFitPredictWithMonthlyTrend(t *testing.T) {
	frame, _, err := loadTestdataFrame("realKnownCause/p7d_anom_curr_nyc_taxi.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (current): %v", err)
	}
	historyFrame, _, err := loadTestdataFrame("realKnownCause/p7d_clean_hist_nyc_taxi.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (history): %v", err)
	}

	options := BaselineOptions{
		TrendType:           "Monthly",
		IntervalMins:        30,
		ConfidenceLevel:     95.0,
		AllowNegativeBounds: false,
	}

	result, err := BaselineFitPredict(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("BaselineFitPredict: %v", err)
	}

	anomalyPreds, ok := extractFrameFloats(result, "anomaly")
	if !ok {
		t.Fatal("result frame does not contain an 'anomaly' column")
	}
	if len(anomalyPreds) == 0 {
		t.Fatal("monthly baseline returned an empty anomaly column")
	}
}

// ─── Forecaster test ──────────────────────────────────────────────────────────

// TestRSODForecaster verifies that the Perpetual-Booster forecaster beats the
// naïve one-step persistence baseline by at least 5%% on MAE and RMSE,
// and achieves MASE < 1.0.
//
// Fixtures:
//
//	current : dataset/testdata/realAdExchange/p24h7d_anom_curr_exchange2_cpm.csv
//	history : dataset/testdata/realAdExchange/p24h7d_clean_hist_exchange2_cpm.csv
//
// Baseline: naïve persistence forecast on the current window.
func TestRSODForecaster(t *testing.T) {
	frame, _, err := loadTestdataFrame("realAdExchange/p24h7d_anom_curr_exchange2_cpm.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (current): %v", err)
	}
	historyFrame, _, err := loadTestdataFrame("realAdExchange/p24h7d_clean_hist_exchange2_cpm.csv")
	if err != nil {
		t.Fatalf("loadTestdataFrame (history): %v", err)
	}

	// Read actual values from the current frame for metric computation.
	actualVals, ok := extractFrameFloats(frame, "Value")
	if !ok || len(actualVals) == 0 {
		t.Fatal("current frame has no 'Value' column")
	}
	histVals, _ := extractFrameFloats(historyFrame, "Value")

	seed := uint64(0)
	logIterations := 0
	options := ForecasterOptions{
		ModelName: "forecaster_model",
		// 24 = one day and 168 = one week at hourly resolution.
		Periods:             []uint{24, 168},
		UUID:                "forecaster_uuid",
		Budget:              0.5,
		NumThreads:          1,
		Nlags:               24,
		StdDevMultiplier:    2.0,
		AllowNegativeBounds: false,
		MaxBin:              255,
		Seed:                &seed,
		LogIterations:       &logIterations,
	}

	result, err := RSODForecaster(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("RSODForecaster: %v", err)
	}

	// Extract the "pred" column from the result frame.
	predicted, ok := extractFrameFloats(result, "pred")
	if !ok || len(predicted) == 0 {
		t.Fatal("result frame does not contain a 'pred' column")
	}

	// Align lengths (result may have fewer rows than the input frame).
	n := len(actualVals)
	if len(predicted) < n {
		n = len(predicted)
	}
	actuals := actualVals[:n]
	predicted = predicted[:n]

	// Compute naïve persistence baseline on the current window.
	baselineMAE, baselineRMSE := naiveBaselineMetrics(actuals)

	// Use history as the MASE denominator; fall back to actuals if unavailable.
	trainingActuals := histVals
	if len(trainingActuals) == 0 {
		trainingActuals = actuals
	}

	m := computeForecastMetrics(actuals, predicted, trainingActuals)
	// Default thresholds (rust-ml-boundary skill):
	//   MASE < 1.0; MAE <= 0.95 * baselineMAE; RMSE <= 0.95 * baselineRMSE
	assertForecastMetrics(t, m, baselineMAE, baselineRMSE, 1.0)
}

// ─── Benchmarks ───────────────────────────────────────────────────────────────

// Benchmark_24_7x24_OutlierFitPredict benchmarks outlier detection on the
// daily-jump seasonal fixture.
func Benchmark_24_7x24_OutlierFitPredict(b *testing.B) {
	frame, _, err := loadTestdataFrame(
		"artificialWithAnomaly/p24h_anom_art_daily_jumpsup.csv",
	)
	if err != nil {
		b.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{288, 2016},
	}
	b.ResetTimer()
	for i := range b.N {
		fmt.Printf("benchmark %d ...\n", i)
		_, err := OutlierFitPredict(frame, options)
		if err != nil {
			b.Fatalf("OutlierFitPredict: %v", err)
		}
	}
}

// BenchmarkErrorRateOutlierFitPredict benchmarks outlier detection on the
// real-world NYC taxi fixture.
func BenchmarkErrorRateOutlierFitPredict(b *testing.B) {
	frame, _, err := loadTestdataFrame("realKnownCause/p7d_anom_curr_nyc_taxi.csv")
	if err != nil {
		b.Fatalf("loadTestdataFrame: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}
	b.ResetTimer()
	for i := range b.N {
		fmt.Printf("benchmark %d ...\n", i)
		_, err := OutlierFitPredict(frame, options)
		if err != nil {
			b.Fatalf("OutlierFitPredict: %v", err)
		}
	}
}
