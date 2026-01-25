package plugin

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/constant"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/grafana/grafana-plugin-sdk-go/data"
	str2duration "github.com/xhit/go-str2duration/v2"
)

func UniqueSlice[T comparable](s []T) []T {
	seen := make(map[T]struct{})
	result := make([]T, 0, len(s))
	for _, v := range s {
		if _, exists := seen[v]; !exists {
			seen[v] = struct{}{}
			result = append(result, v)
		}
	}
	return result
}

func ParsePeriods(durations string, intervalMs int64) ([]uint, error) {
	periods := make([]uint, 0)
	for _, dStr := range strings.FieldsFunc(durations, func(r rune) bool { return r == ',' }) {
		d, err := str2duration.ParseDuration(dStr)
		if err != nil {
			return nil, err
		}
		periods = append(periods, uint(d.Milliseconds()/intervalMs))
	}
	return periods, nil
}

func GetRecalculateTimeRange(from, to time.Time, htr HistoryTimeRange) (time.Time, time.Time) {
	duration := htr.To - htr.From
	return from.Add(time.Duration(duration) * time.Second), to
}

func GetHistoryTimeRange(currentFrom time.Time, htr HistoryTimeRange) (time.Time, time.Time) {
	duration := htr.To - htr.From
	historyTo := currentFrom.Add(-time.Duration(duration) * time.Second)
	return currentFrom, historyTo
}

func splitFrames(frame *data.Frame, currentFrom, currentTo time.Time, htr HistoryTimeRange) (currentFrame *data.Frame, historyFrame *data.Frame, err error) {
	if frame == nil || len(frame.Fields) == 0 {
		return nil, nil, errors.New("frame is nil or has no fields")
	}
	historyFrom, historyTo := GetHistoryTimeRange(currentFrom, htr)
	historyFrame, err = splitFrameByTime(frame, historyFrom, historyTo)
	if err != nil {
		return nil, nil, err
	}

	currentFrame, err = splitFrameByTime(frame, historyTo, currentTo)
	if err != nil {
		return nil, nil, err
	}
	return currentFrame, historyFrame, nil
}

func splitFrameByTime(frame *data.Frame, from, to time.Time) (*data.Frame, error) {
	if frame == nil || len(frame.Fields) == 0 {
		return nil, errors.New("frame is nil or has no fields")
	}

	filteredFrame, err := frame.FilterRowsByField(0, func(i any) (bool, error) {
		tv, ok := i.(time.Time)
		if !ok {
			return false, nil
		}
		return (tv.Equal(from) || tv.After(from)) && tv.Before(to), nil
	})
	if err != nil {
		return nil, err
	}

	return filteredFrame, nil
}

func DebugFrameHead(frame *data.Frame, n int) string {
	if frame == nil {
		return "Frame is nil\n"
	}
	if len(frame.Fields) == 0 {
		return fmt.Sprintf("Frame '%s' has no fields\n", frame.Name)
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("=== Frame: %s ===\n", frame.Name))
	sb.WriteString(fmt.Sprintf("RefID: %s\n", frame.RefID))
	sb.WriteString(fmt.Sprintf("Fields: %d, Rows: %d\n\n", len(frame.Fields), frame.Fields[0].Len()))

	rowCount := frame.Fields[0].Len()
	if n > 0 && n < rowCount {
		rowCount = n
	}

	for i, field := range frame.Fields {
		if i > 0 {
			sb.WriteString(" | ")
		}
		sb.WriteString(fmt.Sprintf("%s (%s)", field.Name, getFieldTypeName(field)))
	}
	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("-", 80))
	sb.WriteString("\n")

	for row := 0; row < rowCount; row++ {
		for col, field := range frame.Fields {
			if col > 0 {
				sb.WriteString(" | ")
			}
			value := formatFieldValue(field, row)
			sb.WriteString(fmt.Sprintf("%-20s", value))
		}
		sb.WriteString("\n")
	}

	if n > 0 && n < frame.Fields[0].Len() {
		sb.WriteString(fmt.Sprintf("... (showing %d of %d rows)\n", n, frame.Fields[0].Len()))
	}

	return sb.String()
}

func DebugFrameInfo(frame *data.Frame) string {
	if frame == nil {
		return "Frame is nil\n"
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("=== Frame Info: %s ===\n", frame.Name))
	sb.WriteString(fmt.Sprintf("RefID: %s\n", frame.RefID))
	sb.WriteString(fmt.Sprintf("Number of fields: %d\n", len(frame.Fields)))
	if len(frame.Fields) > 0 {
		sb.WriteString(fmt.Sprintf("Number of rows: %d\n", frame.Fields[0].Len()))
	}

	if frame.Meta != nil {
		sb.WriteString(fmt.Sprintf("Meta Type: %s\n", frame.Meta.Type))
	}

	sb.WriteString("\nColumn Details:\n")
	sb.WriteString(fmt.Sprintf("%-20s %-15s %-10s %-15s\n", "Column", "Type", "Non-Null", "Dtype"))
	sb.WriteString(strings.Repeat("-", 70))
	sb.WriteString("\n")

	for _, field := range frame.Fields {
		nonNull := countNonNull(field)
		dtype := getFieldTypeName(field)
		sb.WriteString(fmt.Sprintf("%-20s %-15s %-10d %-15s\n",
			field.Name, dtype, nonNull, dtype))
	}

	return sb.String()
}

func DebugFrame(frame *data.Frame, headRows int) string {
	var sb strings.Builder
	sb.WriteString(DebugFrameInfo(frame))
	sb.WriteString("\n")
	sb.WriteString(DebugFrameHead(frame, headRows))
	return sb.String()
}

func getFieldTypeName(field *data.Field) string {
	if field == nil {
		return "unknown"
	}
	return field.Type().String()
}

func formatFieldValue(field *data.Field, idx int) string {
	if field == nil || idx >= field.Len() {
		return "<nil>"
	}

	val := field.At(idx)
	if val == nil {
		return "NULL"
	}

	switch v := val.(type) {
	case time.Time:
		return v.Format("2006-01-02 15:04:05")
	case *time.Time:
		if v == nil {
			return "NULL"
		}
		return v.Format("2006-01-02 15:04:05")
	case float64:
		return fmt.Sprintf("%.6f", v)
	case *float64:
		if v == nil {
			return "NULL"
		}
		return fmt.Sprintf("%.6f", *v)
	case string:
		return v
	default:
		return fmt.Sprintf("%v", val)
	}
}

func countNonNull(field *data.Field) int {
	if field == nil {
		return 0
	}
	count := 0
	for i := 0; i < field.Len(); i++ {
		val := field.At(i)
		if val != nil {
			if f64, ok := val.(*float64); ok && f64 == nil {
				continue
			}
			if t, ok := val.(*time.Time); ok && t == nil {
				continue
			}
			count++
		}
	}
	return count
}

// frameToArrowIPC converts a Grafana DataFrame to Arrow IPC bytes
func frameToArrowIPC(frame *data.Frame) ([]byte, error) {
	if frame == nil {
		return nil, fmt.Errorf("frame is nil")
	}

	// Convert Grafana frame to Arrow table
	table, err := data.FrameToArrowTable(frame)
	if err != nil {
		return nil, fmt.Errorf("failed to convert frame to arrow table: %w", err)
	}

	tr := array.NewTableReader(table, -1)
	defer tr.Release()

	// Create IPC writer
	var buf bytes.Buffer
	w := ipc.NewWriter(&buf, ipc.WithSchema(table.Schema()))
	for tr.Next() {
		if err := w.Write(tr.RecordBatch()); err != nil {
			return nil, err
		}
	}

	// Write table (simplified approach)
	// Note: This is a placeholder - proper implementation would iterate through table chunks
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	return buf.Bytes(), nil
}

// arrowIPCToFrame converts Arrow IPC bytes back to Grafana DataFrame
func arrowIPCToFrame(dataBytes []byte) (*data.Frame, error) {
	if len(dataBytes) == 0 {
		return nil, fmt.Errorf("empty data bytes")
	}

	// Create a reader for Arrow IPC Stream format
	reader, err := ipc.NewReader(bytes.NewReader(dataBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create arrow IPC reader: %w", err)
	}
	defer reader.Release()

	// Read the first record batch and convert it directly
	// Most of the time there's only one batch, so we handle that case first
	if !reader.Next() {
		return nil, fmt.Errorf("no record batches in response")
	}

	batch := reader.RecordBatch()

	frame, err := data.FromArrowRecord(batch)
	if err != nil {
		return nil, fmt.Errorf("failed to convert record to frame: %w", err)
	}
	return frame, nil
}

func getGrafanaPluginDir() string {
	// Use environment variable to get the Grafana plugin directory
	gfPluginDir := os.Getenv("GF_PATHS_PLUGINS")
	if gfPluginDir == "" {
		gfPluginDir = "/var/lib/grafana/plugins"
	}
	return filepath.Join(gfPluginDir, constant.PluginDatasourceType)
}
