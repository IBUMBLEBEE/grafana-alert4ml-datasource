package plugin

import (
	"fmt"
	"time"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/constant"

	"github.com/grafana/grafana-plugin-sdk-go/backend/log"
	"github.com/grafana/grafana-plugin-sdk-go/data"
)

// convertTimeField converts a field containing timestamps to a time.Time field
func convertTimeField(field *data.Field) *data.Field {
	fieldLength := field.Len()
	timeField := data.NewField(constant.GF_FRAME_RESULT_NAME_TIME, field.Labels, make([]time.Time, fieldLength))
	timeField.Config = &data.FieldConfig{DisplayName: constant.GF_FRAME_RESULT_NAME_TIME}

	for i := range fieldLength {
		var timestampSec float64
		val := field.At(i)

		switch v := val.(type) {
		case int64:
			timestampSec = float64(v)
		case *int64:
			if v == nil {
				continue
			}
			timestampSec = float64(*v)
		case float64:
			timestampSec = v
		case *float64:
			if v == nil {
				continue
			}
			timestampSec = *v
		case time.Time:
			timeField.Set(i, v)
			continue
		default:
			log.DefaultLogger.Warn("Unexpected time field type", "type", fmt.Sprintf("%T", v), "value", v)
			continue
		}

		if timestampSec > 1e12 {
			timestampSec = timestampSec / 1000.0
		}
		timeField.Set(i, time.Unix(int64(timestampSec), 0))
	}

	return timeField
}

func RenderFrameWithBaseline(df *data.Frame, refID string) *data.Frame {
	df.RefID = refID
	df.Name = constant.GF_FRAME_RESULT_NAME_BASELINE
	log.DefaultLogger.Info("df.Fields: ", df.Fields)
	for idx, field := range df.Fields {
		switch field.Name {
		case constant.GF_FRAME_RESULT_NAME_TIME, "time":
			df.Fields[idx] = convertTimeField(field)
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
			df.Fields[idx] = convertTimeField(field)
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
