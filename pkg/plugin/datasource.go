package plugin

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"time"

	"github.com/tidwall/sjson"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/constant"
	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/models"
	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/rsod"
	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/sdk"

	"github.com/grafana/grafana-plugin-sdk-go/backend"
	"github.com/grafana/grafana-plugin-sdk-go/backend/instancemgmt"
	"github.com/grafana/grafana-plugin-sdk-go/data"
	_ "github.com/lib/pq"
)

// Make sure Datasource implements required interfaces. This is important to do
// since otherwise we will only get a not implemented error response from plugin in
// runtime. In this example datasource instance implements backend.QueryDataHandler,
// backend.CheckHealthHandler interfaces. Plugin should not implement all these
// interfaces - only those which are required for a particular task.
var (
	_ backend.QueryDataHandler      = (*Datasource)(nil)
	_ backend.CheckHealthHandler    = (*Datasource)(nil)
	_ instancemgmt.InstanceDisposer = (*Datasource)(nil)
)

// NewDatasource creates a new datasource instance.
func NewDatasource(_ context.Context, _ backend.DataSourceInstanceSettings) (instancemgmt.Instance, error) {
	return &Datasource{}, nil
}

// Datasource is an example datasource which can respond to data queries, reports
// its health and has streaming skills.
type Datasource struct{}

// Dispose here tells plugin SDK that plugin wants to clean up resources when a new instance
// created. As soon as datasource settings change detected by SDK old datasource instance will
// be disposed and a new one will be created using NewSampleDatasource factory function.
func (d *Datasource) Dispose() {
	// Clean up datasource instance resources.
}

// QueryData handles multiple queries and returns multiple responses.
// req contains the queries []DataQuery (where each query contains RefID as a unique identifier).
// The QueryDataResponse contains a map of RefID to the response for each query, and each response
// contains Frames ([]*Frame).
func (d *Datasource) QueryData(ctx context.Context, req *backend.QueryDataRequest) (*backend.QueryDataResponse, error) {
	// 加载插件配置
	config, err := models.LoadPluginSettings(*req.PluginContext.DataSourceInstanceSettings)
	if err != nil {
		return nil, err
	}

	queryAlert4MLQueryBody, err := ParseAlert4MLQueryTargets(req.Queries)
	if err != nil {
		return nil, err
	}

	client := sdk.NewGrafanaClient(config.URL, config.Secrets.ApiToken)
	newResponses := backend.NewQueryDataResponse()
	for _, queryAlert4MLQueryBody := range queryAlert4MLQueryBody {
		rsp, err := client.DataSourceQuery(queryAlert4MLQueryBody)
		if err != nil {
			return nil, fmt.Errorf("datasource query: %v", err.Error())
		}
		for selfRefID, queryResponse := range rsp.Responses {
			if len(queryResponse.Frames) == 0 {
				continue
			}

			queryJson, hyperParams, err := getQueryJsonAndHyperParamsFromRefId(selfRefID, req.Queries)
			if err != nil {
				return nil, err
			}

			// 注意：不同 DetectType 对应不同的 HyperParams 类型，避免在此处做通用断言

			newframes := make([]*data.Frame, 0)
			for frameIdx, f := range queryResponse.Frames {
				if f == nil || len(f.Fields) == 0 {
					continue
				}

				uk := UniqueKeysUUID{
					DetectType:    queryJson.DetectType,
					SupportDetect: queryJson.SupportDetect,
					UniqueKeys:    queryJson.UniqueKeys,
					SeriesName:    f.Name,
				}

				ukUUID, err := uk.ToUUIDString()
				if err != nil {
					return nil, err
				}

				switch queryJson.DetectType {
				case constant.DetectTypeOutlier:
					// 仅在 Outlier 模式下解析 Rsod 参数
					rsodParams := hyperParams.(*RsodHyperParams)
					periods, err := ParsePeriods(rsodParams.Periods, queryAlert4MLQueryBody.IntervalMs)
					if err != nil {
						return nil, err
					}

					options := rsod.OutlierOptions{
						ModelName: rsodParams.ModelName,
						Periods:   periods,
						UUID:      ukUUID,
					}

					// Prepare request
					reqFrame := queryResponse.DeepCopy().Frames[frameIdx]
					err = TransformDataFrame(reqFrame)
					if err != nil {
						return nil, err
					}

					resultOutlier, err := rsod.OutlierFitPredict(reqFrame, options)
					if err != nil {
						return nil, err
					}

					newframe := newDataFrameFromeResult(f, constant.DetectTypeOutlier, constant.GF_FRAME_RESULT_NAME_ANOMALY, selfRefID, resultOutlier)
					newframes = append(newframes, newframe)

				case constant.BaselineDetectTypeStd, constant.BaselineDetectTypeZScore, constant.BaselineDetectTypeMovingAverage:
					options := rsod.BaselineOptions{
						TrendType:        hyperParams.(*BaselineHyperParams).TrendType,
						IntervalMins:     hyperParams.(*BaselineHyperParams).IntervalMins,
						StdDevMultiplier: hyperParams.(*BaselineHyperParams).StdDevMultiplier,
						UUID:             ukUUID,
					}
					// Prepare frames
					rawFrame := queryResponse.DeepCopy().Frames[frameIdx]
					currentFrame, historyFrame, err := splitFrames(rawFrame, queryAlert4MLQueryBody.From, queryAlert4MLQueryBody.To, queryJson.HistoryTimeRange)
					if err != nil {
						return nil, err
					}
					// 转换时间字段为 float64
					err = TransformDataFrame(currentFrame)
					if err != nil {
						return nil, err
					}
					err = TransformDataFrame(historyFrame)
					if err != nil {
						return nil, err
					}

					resultBaselineDF, err := rsod.BaselineFitPredict(currentFrame, historyFrame, options)
					if err != nil {
						return nil, err
					}

					newframe := RenderFrameWithBaseline(resultBaselineDF, selfRefID)
					if queryJson.ShowAnomalyPoints {
						removeNonAnomalyFields(newframe)
					}
					newframes = append(newframes, newframe)

				case constant.BaselineDetectTypeDynamics:
					options := rsod.DynamicsOptions{
						Seasonality:       hyperParams.(*DynamicsHyperParams).Seasonality,
						WindowSize:        hyperParams.(*DynamicsHyperParams).WindowSize,
						MinPoints:         hyperParams.(*DynamicsHyperParams).MinPoints,
						WarningThreshold:  hyperParams.(*DynamicsHyperParams).WarningThreshold,
						CriticalThreshold: hyperParams.(*DynamicsHyperParams).CriticalThreshold,
						RobustMode:        hyperParams.(*DynamicsHyperParams).RobustMode,
					}
					// Prepare frames
					rawFrame := queryResponse.DeepCopy().Frames[frameIdx]
					currentFrame, historyFrame, err := splitFrames(rawFrame, queryAlert4MLQueryBody.From, queryAlert4MLQueryBody.To, queryJson.HistoryTimeRange)
					if err != nil {
						return nil, err
					}
					// 转换时间字段为 float64
					err = TransformDataFrame(currentFrame)
					if err != nil {
						return nil, err
					}
					err = TransformDataFrame(historyFrame)
					if err != nil {
						return nil, err
					}

					resultDynamicsDF, err := rsod.DynamicsFitPredict(currentFrame, historyFrame, options)
					if err != nil {
						return nil, err
					}

					newframe := RenderFrameWithBaseline(resultDynamicsDF, selfRefID)
					if queryJson.ShowAnomalyPoints {
						removeNonAnomalyFields(newframe)
					}
					newframes = append(newframes, newframe)

				case constant.DetectTypeForecast:
					periods, err := ParsePeriods(hyperParams.(*ForecastHyperParams).Periods, queryAlert4MLQueryBody.IntervalMs)
					if err != nil {
						return nil, err
					}

					forecasterOptions := rsod.ForecasterOptions{
						ModelName:           hyperParams.(*ForecastHyperParams).ModelName,
						Periods:             periods,
						UUID:                ukUUID,
						Budget:              hyperParams.(*ForecastHyperParams).Budget,
						NumThreads:          hyperParams.(*ForecastHyperParams).NumThreads,
						Nlags:               hyperParams.(*ForecastHyperParams).Nlags,
						StdDevMultiplier:    hyperParams.(*ForecastHyperParams).StdDevMultiplier,
						AllowNegativeBounds: hyperParams.(*ForecastHyperParams).AllowNegativeBounds,
					}

					// Prepare frames
					rawFrame := queryResponse.DeepCopy().Frames[frameIdx]
					currentFrame, historyFrame, err := splitFrames(rawFrame, queryAlert4MLQueryBody.From, queryAlert4MLQueryBody.To, queryJson.HistoryTimeRange)
					if err != nil {
						return nil, err
					}
					// 转换时间字段为 float64
					err = TransformDataFrame(currentFrame)
					if err != nil {
						return nil, err
					}
					err = TransformDataFrame(historyFrame)
					if err != nil {
						return nil, err
					}

					resultForecastDF, err := rsod.RSODForecaster(currentFrame, historyFrame, forecasterOptions)
					if err != nil {
						return nil, err
					}
					newframe := RenderFrameWithForecast(resultForecastDF, selfRefID, f.Name)
					if queryJson.ShowAnomalyPoints {
						removeNonAnomalyFields(newframe)
					}
					newframes = append(newframes, newframe)
				}
			}
			existingResponse := rsp.Responses[selfRefID]
			if queryJson.ShowAnomalyPoints {
				// 仅显示异常点：丢弃原始数据帧，只返回检测结果帧
				existingResponse.Frames = newframes
			} else {
				existingResponse.Frames = append(existingResponse.Frames, newframes...)
			}
			newResponses.Responses[selfRefID] = existingResponse
		}
	}

	return newResponses, nil
}

func TransformDataFrame(df *data.Frame) error {
	// Convert Time field from time.Time to float64
	err := ConvertField(df, df.Fields[0].Name, time2Float64FieldConverter)
	if err != nil {
		return err
	}
	// err = ConvertField(df, df.Fields[1].Name, value2Float64FieldConverter)
	// if err != nil {
	// 	return err
	// }
	return nil
}

// CheckHealth handles health checks sent from Grafana to the plugin.
// The main use case for these health checks is the test button on the
// datasource configuration page which allows users to verify that
// a datasource is working as expected.
func (d *Datasource) CheckHealth(_ context.Context, req *backend.CheckHealthRequest) (*backend.CheckHealthResult, error) {
	res := &backend.CheckHealthResult{}
	config, err := models.LoadPluginSettings(*req.PluginContext.DataSourceInstanceSettings)

	if err != nil {
		res.Status = backend.HealthStatusError
		res.Message = "Unable to load settings"
		return res, nil
	}

	if config.Secrets.ApiToken == "" {
		res.Status = backend.HealthStatusError
		res.Message = "API Token is missing"
		return res, nil
	}

	err = d.CheckGrafanaHealth(config)
	if err != nil {
		return &backend.CheckHealthResult{
			Status:  backend.HealthStatusError,
			Message: err.Error(),
		}, nil
	}

	if config.TrialMode {
		return &backend.CheckHealthResult{
			Status:  backend.HealthStatusOk,
			Message: "Data source is working (trial mode, using SQLite in-memory storage)",
		}, nil
	}

	if config.PgHost == "" || config.PgDatabase == "" || config.PgUser == "" {
		return &backend.CheckHealthResult{
			Status:  backend.HealthStatusError,
			Message: "PostgreSQL configuration is incomplete: host, database and user are required",
		}, nil
	}

	err = d.CheckPgHealth(config)
	if err != nil {
		return &backend.CheckHealthResult{
			Status:  backend.HealthStatusError,
			Message: fmt.Sprintf("PostgreSQL: %s", err.Error()),
		}, nil
	}

	return &backend.CheckHealthResult{
		Status:  backend.HealthStatusOk,
		Message: "Data source is working",
	}, nil
}

func (d *Datasource) CheckGrafanaHealth(config *models.PluginSettings) error {
	// Check if Grafana is healthy
	_, err := url.Parse(config.URL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}
	client := sdk.NewGrafanaClient(config.URL, config.Secrets.ApiToken)
	return client.LoginPing()
}

func (d *Datasource) CheckPgHealth(config *models.PluginSettings) error {
	db, err := sql.Open("postgres", config.PgDSN())
	if err != nil {
		return fmt.Errorf("failed to open PostgreSQL connection: %w", err)
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping PostgreSQL: %w", err)
	}
	return nil
}

// ParseQuery 解析查询参数
func ParseQueryJson(jsonData json.RawMessage) (*Alert4MLQueryJson, error) {
	query := Alert4MLQueryJson{}
	err := json.Unmarshal(jsonData, &query)
	if err != nil {
		return nil, err
	}

	return &query, nil
}

// ParseAlert4MLQueryTargets 解析查询参数
func ParseAlert4MLQueryTargets(queries []backend.DataQuery) ([]*Alert4MLQueryBody, error) {
	queryBodies := make([]*Alert4MLQueryBody, 0)
	for _, query := range queries {
		queryJson, err := ParseQueryJson(query.JSON)
		if err != nil {
			return nil, err
		}
		queriesJson := make([]json.RawMessage, 0)
		for idx := range queryJson.Targets {
			queryStr, err := sjson.Set(string(queryJson.Targets[idx]), "intervalMs", query.Interval.Milliseconds())
			if err != nil {
				return nil, err
			}
			// 将refId设置为查询参数中的refId
			queryStr, err = sjson.Set(queryStr, "refId", query.RefID)
			if err != nil {
				return nil, err
			}
			queryJson.Targets[idx] = []byte(queryStr)
			queriesJson = append(queriesJson, queryJson.Targets[idx])
		}
		from, to := GetRecalculateTimeRange(query.TimeRange.From, query.TimeRange.To, queryJson.HistoryTimeRange)
		queryBodies = append(queryBodies, &Alert4MLQueryBody{
			Queries:    queriesJson,
			From:       from,
			To:         to,
			IntervalMs: query.Interval.Milliseconds(),
		})
	}
	return queryBodies, nil
}

func getQueryJsonAndHyperParamsFromRefId(refId string, queries []backend.DataQuery) (*Alert4MLQueryJson, HyperParams, error) {
	for _, query := range queries {
		if query.RefID == refId {
			queryJson, err := ParseQueryJson(query.JSON)
			if err != nil {
				return nil, nil, err
			}
			hyperParams, err := ParseHyperParams(queryJson.DetectType, queryJson.HyperParams)
			if err != nil {
				return nil, nil, err
			}
			return queryJson, hyperParams, nil
		}
	}
	return nil, nil, fmt.Errorf("refId not found")
}

// newDataFrameFromeResult 根据检测类型和结果生成新的 data.Frame
func newDataFrameFromeResult(df *data.Frame, detectType string, resultName string, refID string, result []float64) *data.Frame {
	fieldLength := df.Fields[0].Len()

	timeField := data.NewField(df.Fields[0].Name, df.Fields[0].Labels, make([]time.Time, fieldLength))
	timeField.Config = &data.FieldConfig{DisplayName: df.Fields[0].Name}

	valueField := data.NewField(df.Fields[1].Name, df.Fields[1].Labels, make([]float64, fieldLength))
	valueField.SetConfig(setDataFrameFieldConfigForOutlier(df, refID, resultName))
	if constant.IsBaselineDetectType(detectType) {
		valueField.SetConfig(setDataFrameFieldConfigForBaseline(df, refID, resultName))
	}
	for idx := range fieldLength {
		// 处理时间字段：可能是 time.Time 或 float64（Unix 时间戳）
		if t, ok := df.Fields[0].At(idx).(time.Time); ok {
			// 时间字段是 time.Time 类型（未转换的情况）
			timeField.Set(idx, t)
		} else if timestamp, ok := df.Fields[0].At(idx).(float64); ok {
			// 时间字段已经被转换为 float64（Unix 时间戳）
			// 将 Unix 时间戳转换回 time.Time
			timeField.Set(idx, time.Unix(int64(timestamp), 0))
		}
		// 处理结果值
		if idx < len(result) {
			value, err := df.Fields[1].NullableFloatAt(idx)
			if err != nil {
				return nil
			}
			switch {
			case detectType == constant.DetectTypeOutlier:
				if result[idx] == 1 {
					valueField.Set(idx, *value*result[idx])
				} else {
					valueField.Set(idx, math.NaN())
				}
			case constant.IsBaselineDetectType(detectType):
				valueField.Set(idx, result[idx])
			default:
				valueField.Set(idx, math.NaN())
			}
		} else {
			valueField.Set(idx, math.NaN())
		}
	}
	newFrame := data.NewFrame(resultName, timeField, valueField)
	newFrame.RefID = refID
	newFrame.Meta = df.Meta
	return newFrame
}

func setDataFrameFieldConfigForOutlier(df *data.Frame, refID string, resultName string) *data.FieldConfig {
	return &data.FieldConfig{
		DisplayName: fmt.Sprintf("%s %s-%s", df.Name, refID, resultName),
		Color: map[string]any{
			"fixedColor": "red",
			"mode":       "fixed",
		},
		Custom: map[string]any{
			"lineStyle": map[string]any{"fill": "solid"},
			"drawStyle": "points",
			"pointSize": 10,
		},
	}
}

func setDataFrameFieldConfigForBaseline(df *data.Frame, refID string, resultName string) *data.FieldConfig {
	return &data.FieldConfig{
		DisplayName: fmt.Sprintf("%s %s-%s", df.Name, refID, resultName),
		Color: map[string]any{
			"fixedColor": "#ccccdc",
			"mode":       "fixed",
		},
		Custom: map[string]any{
			"lineStyle": map[string]any{"fill": "solid"},
			"drawStyle": "lines",
			"pointSize": 1,
		},
	}
}
