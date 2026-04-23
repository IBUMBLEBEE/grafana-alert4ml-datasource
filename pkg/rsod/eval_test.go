package rsod

// eval_test.go — Unified ML evaluation metric helpers for test assertions.
//
// Thresholds (from .claude/skills/rust-ml-boundary/SKILL.md):
//   Outlier  : F1 >= 0.80, Recall >= 0.85
//   Forecast : MASE < 1.0; MAE and RMSE <= 0.95 * respective baseline value
//
// Usage pattern:
//   frame, labels, err := loadTestdataFrame("artificialWithAnomaly/p24h_anom_art_load_balancer_spikes.csv")
//   predictions, err := OutlierFitPredict(frame, opts)
//   m := computeOutlierMetrics(predictions, labels)
//   assertOutlierMetrics(t, m, 0.80, 0.85)

import (
	"encoding/csv"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/grafana/grafana-plugin-sdk-go/data"
)

// testdataRoot is the path to dataset/testdata/ relative to the pkg/rsod/ working
// directory that Go's test runner uses.
const testdataRoot = "../../dataset/testdata"

// ─── Outlier metrics ──────────────────────────────────────────────────────────

// OutlierMetrics holds the confusion-matrix based evaluation metrics for
// anomaly / outlier detection tasks.
//
// Default thresholds (rust-ml-boundary skill): F1 >= 0.80, Recall >= 0.85.
type OutlierMetrics struct {
	TruePositives  int
	FalsePositives int
	FalseNegatives int
	Precision      float64
	Recall         float64
	F1             float64
}

// computeOutlierMetrics computes Precision, Recall and F1 from binary scores.
//
// Both slices must have the same length and use 0.0 (normal) / 1.0 (anomaly)
// encoding. A score >= 0.5 is treated as a positive (anomaly) prediction.
func computeOutlierMetrics(predictions, labels []float64) OutlierMetrics {
	if len(predictions) != len(labels) {
		panic("computeOutlierMetrics: predictions and labels must have the same length")
	}
	var tp, fp, fn int
	for i, p := range predictions {
		pred := p >= 0.5
		actual := labels[i] >= 0.5
		switch {
		case pred && actual:
			tp++
		case pred && !actual:
			fp++
		case !pred && actual:
			fn++
		}
	}
	precision := 0.0
	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp)
	}
	recall := 0.0
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn)
	}
	f1 := 0.0
	if precision+recall > 0 {
		f1 = 2 * precision * recall / (precision + recall)
	}
	return OutlierMetrics{
		TruePositives:  tp,
		FalsePositives: fp,
		FalseNegatives: fn,
		Precision:      precision,
		Recall:         recall,
		F1:             f1,
	}
}

// assertOutlierMetrics fails the test if F1 or Recall is below the required minimum.
//
// Call with the default thresholds (rust-ml-boundary skill):
//
//	assertOutlierMetrics(t, m, 0.80, 0.85)
func assertOutlierMetrics(t *testing.T, m OutlierMetrics, minF1, minRecall float64) {
	t.Helper()
	t.Logf(
		"OutlierMetrics: F1=%.4f Recall=%.4f Precision=%.4f TP=%d FP=%d FN=%d",
		m.F1, m.Recall, m.Precision,
		m.TruePositives, m.FalsePositives, m.FalseNegatives,
	)
	if m.F1 < minF1 {
		t.Errorf(
			"outlier F1 %.4f < required minimum %.4f (TP=%d FP=%d FN=%d)",
			m.F1, minF1, m.TruePositives, m.FalsePositives, m.FalseNegatives,
		)
	}
	if m.Recall < minRecall {
		t.Errorf(
			"outlier Recall %.4f < required minimum %.4f (TP=%d FN=%d)",
			m.Recall, minRecall, m.TruePositives, m.FalseNegatives,
		)
	}
}

// ─── Forecast metrics ─────────────────────────────────────────────────────────

// ForecastMetrics holds MAE, RMSE and MASE for forecast evaluation.
//
// Default thresholds (rust-ml-boundary skill):
//   - MASE < 1.0
//   - MAE  <= 0.95 * baselineMAE
//   - RMSE <= 0.95 * baselineRMSE
type ForecastMetrics struct {
	MAE  float64
	RMSE float64
	// MASE < 1.0 means the model beats naïve one-step-ahead persistence.
	MASE float64
}

// computeForecastMetrics computes MAE, RMSE and MASE.
//
//   - actuals         – ground truth values for the evaluation window.
//   - predicted       – model predictions aligned with actuals.
//   - trainingActuals – in-sample values used to compute the naïve persistence
//     denominator (mean |y[i] − y[i-1]|). May be identical to actuals when no
//     separate training data is available; a finite result is always produced.
func computeForecastMetrics(actuals, predicted, trainingActuals []float64) ForecastMetrics {
	if len(actuals) != len(predicted) {
		panic("computeForecastMetrics: actuals and predicted must have the same length")
	}
	n := float64(len(actuals))
	var sumAE, sumSE float64
	for i, a := range actuals {
		diff := a - predicted[i]
		sumAE += math.Abs(diff)
		sumSE += diff * diff
	}
	mae := sumAE / n
	rmse := math.Sqrt(sumSE / n)

	naiveDenom := 1.0
	if len(trainingActuals) > 1 {
		var sumNaive float64
		for i := 1; i < len(trainingActuals); i++ {
			sumNaive += math.Abs(trainingActuals[i] - trainingActuals[i-1])
		}
		naiveDenom = sumNaive / float64(len(trainingActuals)-1)
	}
	mase := 0.0
	if naiveDenom > 0 {
		mase = mae / naiveDenom
	}
	return ForecastMetrics{MAE: mae, RMSE: rmse, MASE: mase}
}

// assertForecastMetrics fails the test if any threshold is violated.
//
// Call with the default thresholds (rust-ml-boundary skill):
//
//	assertForecastMetrics(t, m, naiveMAE, naiveRMSE, 1.0)
func assertForecastMetrics(
	t *testing.T,
	m ForecastMetrics,
	baselineMAE, baselineRMSE, maxMASE float64,
) {
	t.Helper()
	t.Logf(
		"ForecastMetrics: MAE=%.4f RMSE=%.4f MASE=%.4f (baseline MAE=%.4f RMSE=%.4f)",
		m.MAE, m.RMSE, m.MASE, baselineMAE, baselineRMSE,
	)
	if m.MASE >= maxMASE {
		t.Errorf("forecast MASE %.4f >= required max %.4f", m.MASE, maxMASE)
	}
	if m.MAE > 0.95*baselineMAE {
		t.Errorf(
			"forecast MAE %.4f does not beat baseline by 5%% (baseline=%.4f threshold=%.4f)",
			m.MAE, baselineMAE, 0.95*baselineMAE,
		)
	}
	if m.RMSE > 0.95*baselineRMSE {
		t.Errorf(
			"forecast RMSE %.4f does not beat baseline by 5%% (baseline=%.4f threshold=%.4f)",
			m.RMSE, baselineRMSE, 0.95*baselineRMSE,
		)
	}
}

// ─── Fixture helpers ──────────────────────────────────────────────────────────

// loadTestdataCSV reads a fixture from dataset/testdata/<relPath>.
//
// The CSV must have a header row with three columns:
//  1. timestamp_s (Unix seconds) or timestamp_ms (Unix milliseconds)
//  2. value
//  3. is_anomaly  (0 or 1)
//
// Returns (timestamps_sec, values, labels) as float64 slices.
// Timestamps are always returned in seconds.
func loadTestdataCSV(relPath string) (timestamps, values, labels []float64, err error) {
	path := filepath.Join(testdataRoot, relPath)
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	header, err := r.Read()
	if err != nil {
		return nil, nil, nil, err
	}
	tsIsMs := strings.HasSuffix(strings.ToLower(header[0]), "_ms")

	for {
		record, rerr := r.Read()
		if rerr != nil {
			break
		}
		rawTs, _ := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		ts := rawTs
		if tsIsMs {
			ts = rawTs / 1000.0
		}
		val, _ := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		lab, _ := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		timestamps = append(timestamps, ts)
		values = append(values, val)
		labels = append(labels, lab)
	}
	return timestamps, values, labels, nil
}

// loadTestdataFrame loads a testdata CSV as a *data.Frame plus the label column.
//
// The returned frame has float64 "Time" and "Value" fields compatible with
// OutlierFitPredict, BaselineFitPredict, and RSODForecaster.
func loadTestdataFrame(relPath string) (frame *data.Frame, labels []float64, err error) {
	timestamps, values, labs, err := loadTestdataCSV(relPath)
	if err != nil {
		return nil, nil, err
	}
	frame, err = buildTestFrame(timestamps, values)
	return frame, labs, err
}

// buildTestFrame creates a *data.Frame with float64 "Time" and "Value" fields.
func buildTestFrame(timestamps, values []float64) (*data.Frame, error) {
	timeField := data.NewField("Time", nil, make([]float64, len(timestamps)))
	valueField := data.NewField("Value", nil, make([]float64, len(values)))
	for i := range timestamps {
		timeField.Set(i, timestamps[i])
		valueField.Set(i, values[i])
	}
	frame := data.NewFrame("test", timeField, valueField)
	frame.Meta = &data.FrameMeta{Type: data.FrameTypeTimeSeriesWide}
	return frame, nil
}

// naiveBaselineMetrics computes MAE and RMSE for naïve one-step-ahead persistence
// on the given actuals slice. Used as the comparison baseline for assertForecastMetrics.
func naiveBaselineMetrics(actuals []float64) (mae, rmse float64) {
	if len(actuals) < 2 {
		return 0, 0
	}
	var sumAE, sumSE float64
	n := float64(len(actuals) - 1)
	for i := 1; i < len(actuals); i++ {
		diff := actuals[i] - actuals[i-1]
		sumAE += math.Abs(diff)
		sumSE += diff * diff
	}
	return sumAE / n, math.Sqrt(sumSE / n)
}

// extractFrameFloats extracts all float64 values from a named field in a data.Frame.
func extractFrameFloats(frame *data.Frame, fieldName string) ([]float64, bool) {
	field, _ := frame.FieldByName(fieldName)
	if field == nil {
		return nil, false
	}
	out := make([]float64, field.Len())
	for i := range field.Len() {
		v := field.At(i)
		switch fv := v.(type) {
		case float64:
			out[i] = fv
		case *float64:
			if fv != nil {
				out[i] = *fv
			}
		}
	}
	return out, true
}
