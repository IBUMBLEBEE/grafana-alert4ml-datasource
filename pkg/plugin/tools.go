package plugin

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/data"
	str2duration "github.com/xhit/go-str2duration/v2"
)

func UniqueSlice[T comparable](s []T) []T {
	seen := make(map[T]struct{}) // 空结构体节省内存
	result := make([]T, 0, len(s))
	for _, v := range s {
		if _, exists := seen[v]; !exists {
			seen[v] = struct{}{}       // 标记元素已存在
			result = append(result, v) // 保留首次出现的元素
		}
	}
	return result
}

// ParseDuration 将字符串转换为时间，支持 天，小时，分钟， 将天转换为小时
func ParsePeriods(durations string, intervalMs int64) ([]uint, error) {
	periods := make([]uint, 0)
	for _, dStr := range strings.FieldsFunc(durations, func(r rune) bool { return r == ',' }) {
		d, err := str2duration.ParseDuration(dStr) // 支持 天，小时，分钟
		if err != nil {
			return nil, err
		}
		periods = append(periods, uint(d.Milliseconds()/intervalMs))
	}
	return periods, nil
}

// GetRecalculateTimeRange 根据历史时间范围和查询时间范围计算重新计算的时间范围
//
// from 是查询时间范围的开始时间
//
// to 是查询时间范围的结束时间
//
// htr 是历史时间范围
//
// 返回值是从查询时间范围的开始时间加上历史时间范围的开始时间到查询时间范围的结束时间加上历史时间范围的结束时间
func GetRecalculateTimeRange(from, to time.Time, htr HistoryTimeRange) (time.Time, time.Time) {
	duration := htr.To - htr.From
	return from.Add(time.Duration(duration) * time.Second), to
}

// GetHistoryTimeRange 计算历史查询窗口
//
// 约定：HistoryTimeRange 的 from/to 表示相对于当前查询窗口的秒级偏移量（非绝对时间）。
// 例如 {from: 604800, to: 0} 表示历史窗口为 [current.from-7d, current.from-0s]
func GetHistoryTimeRange(currentFrom time.Time, htr HistoryTimeRange) (time.Time, time.Time) {
	duration := htr.To - htr.From
	historyTo := currentFrom.Add(-time.Duration(duration) * time.Second)
	return currentFrom, historyTo
}

// GetHistoryFrame 在给定的绝对时间范围 [from, to] 内过滤并返回新 frame
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

	// copyFrameProperties(filteredFrame, frame)

	return filteredFrame, nil
}

// DebugFrameHead 打印 Frame 的前 n 行，类似 pandas head()
// 如果 n <= 0，则打印所有行
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

	// 打印表头
	for i, field := range frame.Fields {
		if i > 0 {
			sb.WriteString(" | ")
		}
		sb.WriteString(fmt.Sprintf("%s (%s)", field.Name, getFieldTypeName(field)))
	}
	sb.WriteString("\n")
	sb.WriteString(strings.Repeat("-", 80))
	sb.WriteString("\n")

	// 打印数据行
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

// DebugFrameInfo 打印 Frame 的详细信息，类似 pandas info()
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

// DebugFrame 打印 Frame 的完整调试信息（head + info）
func DebugFrame(frame *data.Frame, headRows int) string {
	var sb strings.Builder
	sb.WriteString(DebugFrameInfo(frame))
	sb.WriteString("\n")
	sb.WriteString(DebugFrameHead(frame, headRows))
	return sb.String()
}

// 辅助函数：获取字段类型名称
func getFieldTypeName(field *data.Field) string {
	if field == nil {
		return "unknown"
	}
	return field.Type().String()
}

// 辅助函数：格式化字段值
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

// 辅助函数：统计非空值数量
func countNonNull(field *data.Field) int {
	if field == nil {
		return 0
	}
	count := 0
	for i := 0; i < field.Len(); i++ {
		val := field.At(i)
		if val != nil {
			// 检查是否为 nullable float64
			if f64, ok := val.(*float64); ok && f64 == nil {
				continue
			}
			// 检查是否为 nullable time
			if t, ok := val.(*time.Time); ok && t == nil {
				continue
			}
			count++
		}
	}
	return count
}
