//go:build linux || darwin || windows

package rsod

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo linux LDFLAGS: -L${SRCDIR}/../target/x86_64-unknown-linux-musl/release -lrsod_go
#cgo darwin LDFLAGS: -L${SRCDIR}/../target/x86_64-apple-darwin/release -lrsod_go
#cgo windows LDFLAGS: -L${SRCDIR}/../target/release -lrsod_go -lws2_32 -luserenv -lbcrypt

#include "rsod_go.h"
*/
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/cdata"
	arrowcsv "github.com/apache/arrow-go/v18/arrow/csv"
	"github.com/grafana/grafana-plugin-sdk-go/data"
)

const MISSDATA_THRESHOLD float64 = 30

type OutlierOptions struct {
	ModelName string `json:"model_name"`
	Periods   []uint `json:"periods"`
	UUID      string `json:"uuid"`
}

type BaselineOptions struct {
	TrendType           string  `json:"trend_type"`
	IntervalMins        int     `json:"interval_mins"`
	ConfidenceLevel     float64 `json:"confidence_level"`
	AllowNegativeBounds bool    `json:"allow_negative_bounds"`
	StdDevMultiplier    float64 `json:"std_dev_multiplier,omitempty"`
	UUID                string  `json:"uuid"`
}

type LLMOptions struct {
	ModelName   string `json:"model_name"`
	Temperature int    `json:"temperature"`
	MaxTokens   int    `json:"max_tokens"`
	UUID        string `json:"uuid"`
}

type ForecasterOptions struct {
	ModelName           string  `json:"model_name"`
	Periods             []uint  `json:"periods"`
	UUID                string  `json:"uuid"`
	Budget              float32 `json:"budget,omitempty"`
	NumThreads          int     `json:"num_threads,omitempty"`
	Nlags               int     `json:"n_lags,omitempty"`
	StdDevMultiplier    float64 `json:"std_dev_multiplier,omitempty"`
	AllowNegativeBounds bool    `json:"allow_negative_bounds,omitempty"`
}

func tableToRecord(table arrow.Table) arrow.Record {
	col_ts := table.Column(0)
	col_value := table.Column(1)
	return array.NewRecord(table.Schema(), []arrow.Array{col_ts.Data().Chunk(0), col_value.Data().Chunk(0)}, int64(col_ts.Len()))
}

func OutlierFitPredict(frame *data.Frame, options OutlierOptions) ([]float64, error) {
	if frame == nil {
		return nil, errors.New("frame is null")
	}

	table, err := data.FrameToArrowTable(frame)
	if err != nil {
		return nil, err
	}

	ok, err := calculateMissingRate(frame, frame.Fields[1].Name)
	if err != nil {
		return nil, err
	}

	if ok {
		// 生成一个全部为0的数组
		field, _ := frame.FieldByName(frame.Fields[0].Name)
		normalResult := make([]float64, field.Len())
		for idx := range field.Len() {
			normalResult[idx] = 0
		}
		return normalResult, nil
	}

	record := tableToRecord(table)
	optsJson, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}
	// 通过cdata 封装成 C ArrowArray 和 C ArrowSchema
	var inArray cdata.CArrowArray
	var inSchema cdata.CArrowSchema
	cdata.ExportArrowRecordBatch(record, &inArray, &inSchema)
	defer cdata.ReleaseCArrowArray(&inArray)
	defer cdata.ReleaseCArrowSchema(&inSchema)

	// Rust 输出
	var outSchema cdata.CArrowSchema
	var outArray cdata.CArrowArray
	defer cdata.ReleaseCArrowArray(&outArray)
	defer cdata.ReleaseCArrowSchema(&outSchema)

	success := C.outlier_fit_predict(
		C.to_arrow_schema(unsafe.Pointer(&inSchema)),
		C.to_arrow_array(unsafe.Pointer(&inArray)),
		C.CString(string(optsJson)),
		C.to_arrow_schema(unsafe.Pointer(&outSchema)),
		C.to_arrow_array(unsafe.Pointer(&outArray)),
	)
	if !success {
		return nil, errors.New("outlier fit predict failed")
	}
	// 从 Rust 输出转换回 Go RecordBatch
	imp, err := cdata.ImportCRecordBatch(&outArray, &outSchema)
	if err != nil {
		return nil, err
	}
	defer imp.Release()
	col1 := imp.Column(0).(*array.Float64)
	col2 := imp.Column(1).(*array.Float64)
	result := make([]float64, col1.Len())
	for i := range col1.Len() {
		result[i] = col2.Value(i)
	}

	return result, nil
}

func calculateMissingRate(frame *data.Frame, valueField string) (bool, error) {
	vfield, _ := frame.FieldByName(valueField)
	if vfield == nil {
		return false, errors.New("value filed is null")
	}

	newValues := make([]float64, vfield.Len())
	var zeroCount float64 = 0
	for i := range vfield.Len() {
		value, err := vfield.NullableFloatAt(i)
		if err != nil {
			return false, err
		}
		if *value == 0 {
			zeroCount++
		}
		newValues[i] = *value
	}
	return zeroCount/float64(vfield.Len())*100 > MISSDATA_THRESHOLD, nil
}

func BaselineFitPredict(frame *data.Frame, historyFrame *data.Frame, options BaselineOptions) (*data.Frame, error) {
	if frame == nil {
		return nil, errors.New("frame is null")
	}

	if historyFrame == nil {
		return nil, errors.New("historyFrame is null")
	}

	table, err := data.FrameToArrowTable(frame)
	if err != nil {
		return nil, err
	}

	historyTable, err := data.FrameToArrowTable(historyFrame)
	if err != nil {
		return nil, err
	}

	record := tableToRecord(table)
	historyRecord := tableToRecord(historyTable)

	// // // 将record和historyRecord写入csv文件
	// recordCsv := "record1.csv"
	// historyRecordCsv := "historyRecord1.csv"

	// WriteArrowRecordToCSV(record, recordCsv)
	// WriteArrowRecordToCSV(historyRecord, historyRecordCsv)

	optsJson, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}

	var inArray cdata.CArrowArray
	var inSchema cdata.CArrowSchema
	cdata.ExportArrowRecordBatch(record, &inArray, &inSchema)
	defer cdata.ReleaseCArrowArray(&inArray)
	defer cdata.ReleaseCArrowSchema(&inSchema)

	var historyInArray cdata.CArrowArray
	var historyInSchema cdata.CArrowSchema
	cdata.ExportArrowRecordBatch(historyRecord, &historyInArray, &historyInSchema)
	defer cdata.ReleaseCArrowArray(&historyInArray)
	defer cdata.ReleaseCArrowSchema(&historyInSchema)

	var outSchema cdata.CArrowSchema
	var outArray cdata.CArrowArray
	defer cdata.ReleaseCArrowArray(&outArray)
	defer cdata.ReleaseCArrowSchema(&outSchema)

	tnow := time.Now()

	// 验证数据有效性（frame 和 historyFrame 已在函数开头检查过 nil）
	if len(frame.Fields) < 2 {
		return nil, errors.New("frame has insufficient fields")
	}
	if len(historyFrame.Fields) < 2 {
		return nil, errors.New("historyFrame has insufficient fields")
	}
	if frame.Fields[0].Len() == 0 {
		return nil, fmt.Errorf("frame has no rows")
	}
	if historyFrame.Fields[0].Len() == 0 {
		return nil, fmt.Errorf("historyFrame has no rows (filtered out by time range)")
	}
	// 创建 CString 并确保释放
	cOptsJson := C.CString(string(optsJson))
	defer C.free(unsafe.Pointer(cOptsJson))

	success := C.baseline_fit_predict(
		C.to_arrow_schema(unsafe.Pointer(&inSchema)),
		C.to_arrow_array(unsafe.Pointer(&inArray)),
		C.to_arrow_array(unsafe.Pointer(&historyInArray)),
		C.to_arrow_schema(unsafe.Pointer(&historyInSchema)),
		cOptsJson,
		C.to_arrow_schema(unsafe.Pointer(&outSchema)),
		C.to_arrow_array(unsafe.Pointer(&outArray)),
	)

	duration := time.Since(tnow)
	// var success bool = true
	if !success {
		return nil, fmt.Errorf("baseline fit predict failed (duration: %v)", duration)
	}

	imp, err := cdata.ImportCRecordBatch(&outArray, &outSchema)
	if err != nil {
		return nil, err
	}
	defer imp.Release()

	// fmt.Println("imp: ", imp)
	dfData, err := data.FromArrowRecord(imp)
	if err != nil {
		return nil, err
	}
	// dfJson, err := dfData.MarshalJSON()
	// if err != nil {
	// 	return nil, err
	// }
	// fmt.Println("dfData: ", string(dfJson))

	// // 使用更优雅的方式访问多列数据
	// // 根据 Arrow 最佳实践，直接使用 Column() 和类型断言
	// if imp.NumCols() < 3 {
	// 	return nil, fmt.Errorf("expected at least 3 columns, got %d", imp.NumCols())
	// }

	// // 获取各列数据
	// timestampCol := imp.Column(0).(*array.Int64)
	// fmt.Println("timestampCol: ", timestampCol)
	// baselineCol := imp.Column(1).(*array.Float64)
	// lowerBoundCol := imp.Column(2).(*array.Float64)
	// upperBoundCol := imp.Column(3).(*array.Float64)

	// // 验证列长度一致
	// length := baselineCol.Len()
	// if lowerBoundCol.Len() != length || upperBoundCol.Len() != length {
	// 	return nil, fmt.Errorf("column length mismatch: baseline=%d, lower_bound=%d, upper_bound=%d",
	// 		length, lowerBoundCol.Len(), upperBoundCol.Len())
	// }

	// // 提取 baseline 值（保持向后兼容，只返回 baseline）
	// result := make([]float64, length)
	// for i := 0; i < length; i++ {
	// 	result[i] = baselineCol.Value(i)
	// }

	// // 如果需要返回置信区间，可以在这里添加日志或返回结构体
	// // 目前保持向后兼容，只返回 baseline 值
	// // 如果需要返回所有数据，可以考虑返回结构体：
	// type BaselineResult struct {
	// 	Baseline   []float64
	// 	LowerBound []float64
	// 	UpperBound []float64
	// }

	return dfData, nil
}

func WriteArrowRecordToCSV(record arrow.Record, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating file: %v\n", err)
		return err
	}
	defer f.Close()
	w := arrowcsv.NewWriter(f, record.Schema())
	err = w.Write(record)
	if err != nil {
		return err
	}

	err = w.Flush()
	if err != nil {
		return err
	}

	err = w.Error()
	if err != nil {
		return err
	}
	return nil
}

func RSODStorageInit() bool {
	success := C.rsod_storage_init()
	if success {
		return true
	}
	// 记录错误但不 panic，避免插件崩溃
	// 错误信息已经在 Rust 代码中通过 eprintln! 输出
	return false
}

func RSODForecaster(frame *data.Frame, historyFrame *data.Frame, options ForecasterOptions) (*data.Frame, error) {
	if frame == nil {
		return nil, errors.New("frame is null")
	}

	if historyFrame == nil {
		return nil, errors.New("historyFrame is null")
	}

	table, err := data.FrameToArrowTable(frame)
	if err != nil {
		return nil, err
	}

	historyTable, err := data.FrameToArrowTable(historyFrame)
	if err != nil {
		return nil, err
	}

	record := tableToRecord(table)
	historyRecord := tableToRecord(historyTable)

	optsJson, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}

	var inArray cdata.CArrowArray
	var inSchema cdata.CArrowSchema
	cdata.ExportArrowRecordBatch(record, &inArray, &inSchema)
	defer cdata.ReleaseCArrowArray(&inArray)
	defer cdata.ReleaseCArrowSchema(&inSchema)

	var historyInArray cdata.CArrowArray
	var historyInSchema cdata.CArrowSchema
	cdata.ExportArrowRecordBatch(historyRecord, &historyInArray, &historyInSchema)
	defer cdata.ReleaseCArrowArray(&historyInArray)
	defer cdata.ReleaseCArrowSchema(&historyInSchema)

	var outSchema cdata.CArrowSchema
	var outArray cdata.CArrowArray
	defer cdata.ReleaseCArrowArray(&outArray)
	defer cdata.ReleaseCArrowSchema(&outSchema)

	tnow := time.Now()

	success := C.rsod_forecaster(
		C.to_arrow_schema(unsafe.Pointer(&inSchema)),
		C.to_arrow_array(unsafe.Pointer(&inArray)),
		C.to_arrow_array(unsafe.Pointer(&historyInArray)),
		C.to_arrow_schema(unsafe.Pointer(&historyInSchema)),
		C.CString(string(optsJson)),
		C.to_arrow_schema(unsafe.Pointer(&outSchema)),
		C.to_arrow_array(unsafe.Pointer(&outArray)),
	)

	duration := time.Since(tnow)
	if !success {
		return nil, fmt.Errorf("forecaster failed (duration: %v)", duration)
	}

	imp, err := cdata.ImportCRecordBatch(&outArray, &outSchema)
	if err != nil {
		return nil, err
	}
	defer imp.Release()

	dfData, err := data.FromArrowRecord(imp)
	if err != nil {
		return nil, err
	}
	return dfData, nil
}
