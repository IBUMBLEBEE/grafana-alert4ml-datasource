package rsod

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/grafana/grafana-plugin-sdk-go/data"
)

func Test_24_7x24_OutlierFitPredict(t *testing.T) {
	frame, err := getTrendSeasonalDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "outlier_model",
		Periods:   []uint{24, 168},
	}

	result, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict failed: %v", err)
	}
	for idx, res := range result {
		if res == 1 {
			fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
		}
	}
}

func TestNoPeriodsOutlierFitPredict(t *testing.T) {
	frame, err := getNoSeasonalDataFromCSV()
	if err != nil {
		t.Fatalf("getNoSeasonalDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	result, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("NoSeasonalOutlierFitPredict failed: %v", err)
	}
	for idx, res := range result {
		if res == 1 {
			fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
		}
	}
}

func TestMissingDataOutlierFitPredict(t *testing.T) {
	frame, err := getMissingDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	result, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict failed: %v", err)
	}
	for idx, res := range result {
		if res == 1 {
			fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
		}
	}
}

func TestAutoPeriodsOutlierFitPredict(t *testing.T) {
	frame, err := getTrendSeasonalDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	result, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict failed: %v", err)
	}
	for idx, res := range result {
		if res == 1 {
			fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
		}
	}
}

func TestErrorRateOutlierFitPredict(t *testing.T) {
	frame, err := getErrorRateDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}

	result, err := OutlierFitPredict(frame, options)
	if err != nil {
		t.Fatalf("OutlierFitPredict failed: %v", err)
	}
	for idx, res := range result {
		if res == 1 {
			fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
		}
	}
}

func Benchmark_24_7x24_OutlierFitPredict(b *testing.B) {
	frame, err := getTrendSeasonalDataFromCSV()
	if err != nil {
		b.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{24, 168},
	}
	for i := range b.N {
		fmt.Printf("benchmark %d ...\n", i)
		result, err := OutlierFitPredict(frame, options)
		if err != nil {
			b.Fatalf("OutlierFitPredict failed: %v", err)
		}
		for idx, res := range result {
			if res == 1 {
				fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
			}
		}
	}
}

func BenchmarkErrorRateOutlierFitPredict(b *testing.B) {
	frame, err := getErrorRateDataFromCSV()
	if err != nil {
		b.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := OutlierOptions{
		ModelName: "test_model",
		Periods:   []uint{},
	}
	for i := range b.N {
		fmt.Printf("benchmark %d ...\n", i)
		result, err := OutlierFitPredict(frame, options)
		if err != nil {
			b.Fatalf("OutlierFitPredict failed: %v", err)
		}
		for idx, res := range result {
			if res == 1 {
				fmt.Printf("index: %d, outlier: %.1f\n", idx, res)
			}
		}
	}
}

func getErrorRateDataFromCSV() (*data.Frame, error) {
	return GetDataFromCSV("data/error_rate.csv")
}

func getTrendSeasonalDataFromCSV() (*data.Frame, error) {
	return GetDataFromCSV("data/seasonal.csv")
}

func getNoSeasonalDataFromCSV() (*data.Frame, error) {
	return GetDataFromCSV("data/no_seasonal.csv")
}

func getMissingDataFromCSV() (*data.Frame, error) {
	return GetDataFromCSV("data/missing_data.csv")
}

func getErrorRate1DataFromCSV() (*data.Frame, error) {
	return GetDataFromCSV("data/record1.csv")
}

func getErrorRate1HistoryDataFromCSV() (*data.Frame, error) {
	return GetDataFromCSV("data/historyRecord1.csv")
}

func GetDataFromCSV(path string) (*data.Frame, error) {
	csvFile, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer csvFile.Close()

	reader := csv.NewReader(csvFile)
	reader.Comma = ','
	reader.LazyQuotes = true
	reader.TrimLeadingSpace = true
	reader.FieldsPerRecord = -1
	reader.ReuseRecord = true

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	timeseries_data := records[1:]

	timeField := data.NewField("Time", nil, make([]float64, len(timeseries_data)))
	valueField := data.NewField("Value", nil, make([]float64, len(timeseries_data)))

	frame := data.NewFrame("test", timeField, valueField)

	for idx, tsd := range timeseries_data {
		vts, err := strconv.ParseFloat(tsd[0], 64)
		if err != nil {
			return nil, err
		}
		timeField.Set(idx, vts)
		vval, err := strconv.ParseFloat(tsd[1], 64)
		if err != nil {
			return nil, err
		}
		valueField.Set(idx, vval)
	}
	frame.Meta = &data.FrameMeta{
		Type: data.FrameTypeTimeSeriesWide,
	}
	return frame, nil
}

func TestBaselineFitPredictWithDailyTrend11(t *testing.T) {
	frame, err := getErrorRate1DataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}
	historyFrame, err := getErrorRate1HistoryDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}
	options := BaselineOptions{
		TrendType:           "Daily",
		IntervalMins:        60,
		ConfidenceLevel:     95.0,
		AllowNegativeBounds: false,
	}

	result, err := BaselineFitPredict(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("BaselineFitPredict failed: %v", err)
	}
	jsonResult, err := result.MarshalJSON()
	if err != nil {
		t.Fatalf("result.MarshalJSON failed: %v", err)
	}
	fmt.Printf("result: %+v\n", string(jsonResult))
	// for idx, res := range result {
	// 	fmt.Printf("index: %d, baseline: %.6f\n", idx, res)
	// }
}

func TestBaselineFitPredictWithWeeklyTrend(t *testing.T) {
	frame, err := getErrorRate1DataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}
	historyFrame, err := getErrorRate1HistoryDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := BaselineOptions{
		TrendType:           "Weekly",
		IntervalMins:        60,
		ConfidenceLevel:     95.0,
		AllowNegativeBounds: false,
	}

	result, err := BaselineFitPredict(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("BaselineFitPredict failed: %v", err)
	}
	fmt.Printf("result: %+v\n", result)
	// for idx, res := range result.Fields[1].At(idx).(float64) {
	// 	fmt.Printf("index: %d, baseline: %.6f\n", idx, res)
	// }
}

func TestBaselineFitPredictWithMonthlyTrend(t *testing.T) {
	frame, err := getErrorRate1DataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}
	historyFrame, err := getErrorRate1HistoryDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := BaselineOptions{
		TrendType:           "Monthly",
		IntervalMins:        60,
		ConfidenceLevel:     95.0,
		AllowNegativeBounds: false,
	}

	result, err := BaselineFitPredict(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("BaselineFitPredict failed: %v", err)
	}
	fmt.Printf("result: %+v\n", result)
	// for idx, res := range result {
	// 	fmt.Printf("index: %d, baseline: %.6f\n", idx, res)
	// }
}

func TestRSODForecaster(t *testing.T) {
	frame, err := getErrorRate1DataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}
	historyFrame, err := getErrorRate1HistoryDataFromCSV()
	if err != nil {
		t.Fatalf("GetDataFromCSV failed: %v", err)
	}

	options := ForecasterOptions{
		ModelName:           "forecaster_model",
		Periods:             []uint{24, 168},
		UUID:                "forecaster_uuid",
		Budget:              0.5,
		NumThreads:          1,
		Nlags:               24,
		StdDevMultiplier:    2.0,
		AllowNegativeBounds: false,
	}

	result, err := RSODForecaster(frame, historyFrame, options)
	if err != nil {
		t.Fatalf("RSODForecaster failed: %v", err)
	}
	jsonResult, err := result.MarshalJSON()
	if err != nil {
		t.Fatalf("result.MarshalJSON failed: %v", err)
	}
	fmt.Printf("result: %+v\n", string(jsonResult))
}
