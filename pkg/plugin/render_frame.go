package plugin

import (
	"alert4ml/pkg/constant"
	"fmt"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/backend/log"
	"github.com/grafana/grafana-plugin-sdk-go/data"
)

func RenderFrame(df *data.Frame) *data.Frame {
	return df
}

func RenderFrameWithBaseline(df *data.Frame, refID string) *data.Frame {
	df.RefID = refID
	df.Name = constant.GF_FRAME_RESULT_NAME_BASELINE
	log.DefaultLogger.Info("df.Fields: ", df.Fields)
	for idx, field := range df.Fields {
		switch field.Name {
		// 小写的time字段是为了兼容Rust端 polars的输出
		case constant.GF_FRAME_RESULT_NAME_TIME, "time":
			// 创建一个新的 time.Time 类型的字段
			fieldLength := field.Len()
			timeField := data.NewField(constant.GF_FRAME_RESULT_NAME_TIME, field.Labels, make([]time.Time, fieldLength))
			timeField.Config = &data.FieldConfig{DisplayName: constant.GF_FRAME_RESULT_NAME_TIME}

			// 将时间戳转换为 time.Time
			for i := range fieldLength {
				var timestampMs int64
				val := field.At(i)

				// 处理不同的时间戳类型
				switch v := val.(type) {
				case int64:
					timestampMs = v
				case float64:
					timestampMs = int64(v)
				case time.Time:
					// 如果已经是 time.Time，直接使用
					timeField.Set(i, v)
					continue
				default:
					log.DefaultLogger.Warn("Unexpected time field type", "type", fmt.Sprintf("%T", v), "value", v)
					continue
				}

				// 时间戳是毫秒，转换为秒并创建 time.Time
				timeField.Set(i, time.Unix(timestampMs/1000, (timestampMs%1000)*1000000))
			}

			// 替换原字段
			df.Fields[idx] = timeField
		case constant.GF_FRAME_RESULT_NAME_BASELINE, "baseline":
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s", refID, constant.GF_FRAME_RESULT_NAME_BASELINE),
				// Color: map[string]any{
				// 	"fixedColor": "#ccccdc",
				// 	"mode":       "fixed",
				// },
				// Custom: map[string]any{
				// 	"lineStyle": "solid",
				// 	"drawStyle": "lines",
				// 	"pointSize": 1,
				// },
			}
		case constant.GF_FRAME_RESULT_NAME_LOWER_BOUND:
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_LOWER_BOUND),
				Color: map[string]any{
					"fixedColor": "#ccccdc",
					"mode":       "fixed",
				},
				Custom: map[string]any{
					"lineStyle": "solid",
					"drawStyle": "lines",
					"pointSize": 1,
				},
			}
		case constant.GF_FRAME_RESULT_NAME_UPPER_BOUND:
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_UPPER_BOUND),
				Color: map[string]any{
					"fixedColor": "#ccccdc",
					"mode":       "fixed",
				},
				Custom: map[string]any{
					"lineStyle": "solid",
					"drawStyle": "lines",
					"pointSize": 1,
				},
			}
		case constant.GF_FRAME_RESULT_NAME_ANOMALY, "anomaly":
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_ANOMALY),
				Color: map[string]any{
					"fixedColor": "red",
					"mode":       "fixed",
				},
				Custom: map[string]any{
					"lineStyle": "solid",
					"drawStyle": "points",
					"pointSize": 10,
				},
			}
		}
	}
	return df
}

func RenderFrameWithOutlier(df *data.Frame) *data.Frame {
	return df
}

func RenderFrameWithLLM(df *data.Frame) *data.Frame {
	return df
}

func RenderFrameWithForecast(df *data.Frame, refID string, seriesName string) *data.Frame {
	df.RefID = refID
	df.Name = seriesName
	df.Meta = &data.FrameMeta{
		Type: data.FrameTypeTimeSeriesWide,
	}
	for idx, field := range df.Fields {
		switch field.Name {
		case constant.GF_FRAME_RESULT_NAME_TIME, "time":
			// 创建一个新的 time.Time 类型的字段
			fieldLength := field.Len()
			timeField := data.NewField(constant.GF_FRAME_RESULT_NAME_TIME, field.Labels, make([]time.Time, fieldLength))
			timeField.Config = &data.FieldConfig{DisplayName: constant.GF_FRAME_RESULT_NAME_TIME}

			// 将时间戳转换为 time.Time
			for i := range fieldLength {
				var timestampMs int64
				val := field.At(i)

				// 处理不同的时间戳类型
				switch v := val.(type) {
				case int64:
					timestampMs = v
				case float64:
					timestampMs = int64(v)
				case time.Time:
					// 如果已经是 time.Time，直接使用
					timeField.Set(i, v)
					continue
				default:
					log.DefaultLogger.Warn("Unexpected time field type", "type", fmt.Sprintf("%T", v), "value", v)
					continue
				}

				// 时间戳是毫秒，转换为秒并创建 time.Time
				timeField.Set(i, time.Unix(timestampMs, (timestampMs%1000)*1000000))
			}

			// 替换原字段
			df.Fields[idx] = timeField
		case constant.GF_FRAME_RESULT_NAME_FORECAST, "pred":
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_FORECAST),
				Custom: map[string]any{
					"lineStyle": "dash",
					"drawStyle": "lines",
					"pointSize": 1,
				},
			}
		case constant.GF_FRAME_RESULT_NAME_UPPER_BOUND:
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_UPPER_BOUND),
				Color: map[string]any{
					"fixedColor": "#ccccdc",
					"mode":       "fixed",
				},
			}
		case constant.GF_FRAME_RESULT_NAME_LOWER_BOUND:
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_LOWER_BOUND),
				Color: map[string]any{
					"fixedColor": "#ccccdc",
					"mode":       "fixed",
				},
			}
		case constant.GF_FRAME_RESULT_NAME_ANOMALY, "anomaly":
			field.Config = &data.FieldConfig{
				DisplayName: fmt.Sprintf("%s-%s-%s", refID, df.Name, constant.GF_FRAME_RESULT_NAME_ANOMALY),
				Color: map[string]any{
					"fixedColor": "red",
					"mode":       "fixed",
				},
				Custom: map[string]any{
					"lineStyle": "solid",
					"drawStyle": "points",
					"pointSize": 10,
				},
			}
		}
	}
	return df
}
