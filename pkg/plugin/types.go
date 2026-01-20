package plugin

import (
	"alert4ml/pkg/constant"
	"encoding/json"
	"fmt"
	"runtime"
	"time"

	"github.com/google/uuid"
)

type Alert4MLQueryJson struct {
	DetectType        string            `json:"detectType"`
	SupportDetect     string            `json:"supportDetect"`
	SeriesRefId       string            `json:"seriesRefId"`
	HyperParams       json.RawMessage   `json:"hyperParams"`
	Targets           []json.RawMessage `json:"targets,omitempty"`
	ShowOriginalData  bool              `json:"showOriginalData"`
	ShowAnomalyPoints bool              `json:"showAnomalyPoints"`
	HistoryTimeRange  HistoryTimeRange  `json:"historyTimeRange"`
	UniqueKeys        UniqueKeys        `json:"uniqueKeys"`
}

type HistoryTimeRange struct {
	From uint `json:"from"`
	To   uint `json:"to"`
}

// HyperParams 是所有超参数的接口
// 参考 gRPC 源码模式，添加 SetDefaults 方法
type HyperParams interface {
	GetType() string
	SetDefaults() // 设置默认值的方法
}

// ParseHyperParams 根据检测类型解析超参数
func ParseHyperParams(detectType string, data json.RawMessage) (HyperParams, error) {
	switch detectType {
	case constant.DetectTypeOutlier:
		var params RsodHyperParams
		if err := json.Unmarshal(data, &params); err != nil {
			return nil, fmt.Errorf("failed to parse RsodHyperParams: %w", err)
		}
		// 参考 gRPC 源码模式，解析后自动设置默认值
		params.SetDefaults()
		return &params, nil
	case constant.SupportDetectTypeBaseline:
		var params BaselineHyperParams
		if err := json.Unmarshal(data, &params); err != nil {
			return nil, fmt.Errorf("failed to parse BaselineHyperParams: %w", err)
		}
		params.SetDefaults()
		return &params, nil
	case constant.SupportDetectTypeLLM:
		var params LLMHyperParams
		if err := json.Unmarshal(data, &params); err != nil {
			return nil, fmt.Errorf("failed to parse LLMHyperParams: %w", err)
		}
		params.SetDefaults()
		return &params, nil
	case constant.DetectTypeForecast:
		var params ForecastHyperParams
		if err := json.Unmarshal(data, &params); err != nil {
			return nil, fmt.Errorf("failed to parse ForecastHyperParams: %w", err)
		}
		params.SetDefaults()
		return &params, nil
	default:
		return nil, fmt.Errorf("unknown detect type: %s", detectType)
	}
}

type RsodHyperParams struct {
	Periods   string `json:"periods,omitempty"`
	ModelName string `json:"model_name,omitempty"`
}

func (p *RsodHyperParams) GetType() string {
	return constant.DetectTypeOutlier
}

// SetDefaults 设置 RSOD 参数的默认值
func (p *RsodHyperParams) SetDefaults() {
	if p.Periods == "" {
		p.Periods = constant.DefaultRsodPeriods
	}
	if p.ModelName == "" {
		p.ModelName = constant.DefaultRsodModelName
	}
}

type Alert4MLQueryBody struct {
	Queries    []json.RawMessage `json:"queries"`
	From       time.Time         `json:"from"`
	To         time.Time         `json:"to"`
	IntervalMs int64             `json:"-"` // 查询间隔，单位毫秒
}

type BaselineHyperParams struct {
	TrendType           string  `json:"trendType,omitempty"`
	IntervalMins        int     `json:"intervalMins,omitempty"`
	ConfidenceLevel     float64 `json:"confidenceLevel,omitempty"`
	AllowNegativeBounds bool    `json:"allowNegativeBounds,omitempty"`
	StdDevMultiplier    float64 `json:"stdDevMultiplier,omitempty"`
}

func (p *BaselineHyperParams) GetType() string {
	return constant.SupportDetectTypeBaseline
}

func (p *BaselineHyperParams) SetDefaults() {
	if p.TrendType == "" {
		p.TrendType = constant.DefaultBaselineTrendType
	}
	if p.IntervalMins == 0 {
		p.IntervalMins = constant.DefaultBaselineIntervalMins
	}
	if p.ConfidenceLevel == 0 {
		p.ConfidenceLevel = constant.DefaultBaselineConfidenceLevel
	}
	if !p.AllowNegativeBounds {
		p.AllowNegativeBounds = constant.DefaultBaselineAllowNegativeBounds
	}
	if p.StdDevMultiplier == 0 {
		p.StdDevMultiplier = constant.DefaultBaselineStdDevMultiplier
	}
}

type LLMHyperParams struct {
	ModelName   string `json:"modelName,omitempty"`
	Temperature int    `json:"temperature,omitempty"`
	MaxTokens   int    `json:"maxTokens,omitempty"`
}

func (p *LLMHyperParams) GetType() string {
	return constant.SupportDetectTypeLLM
}

func (p *LLMHyperParams) SetDefaults() {
	if p.ModelName == "" {
		p.ModelName = constant.LLMDetectTypeDeepseek
	}
	if p.Temperature == 0 {
		p.Temperature = 1
	}
	if p.MaxTokens == 0 {
		p.MaxTokens = 1000
	}
}

type UniqueKeys struct {
	DashboardUid string `json:"dashboardUid"`
	PanelId      int    `json:"panelId"`
	SeriesRefId  string `json:"seriesRefId"`
}

type ForecastHyperParams struct {
	ModelName           string  `json:"model_name"`
	Periods             string  `json:"periods"`
	UUID                string  `json:"uuid"`
	Budget              float32 `json:"budget,omitempty"`
	NumThreads          int     `json:"numThreads,omitempty"`
	Nlags               int     `json:"nlags,omitempty"`
	StdDevMultiplier    float64 `json:"stdDevMultiplier,omitempty"`
	AllowNegativeBounds bool    `json:"allowNegativeBounds,omitempty"`
}

func (p *ForecastHyperParams) GetType() string {
	return constant.DetectTypeForecast
}

func (p *ForecastHyperParams) SetDefaults() {
	if p.ModelName == "" {
		p.ModelName = constant.DetectTypeForecast
	}

	if p.Periods == "" {
		p.Periods = "24h,7d"
	}
	if p.UUID == "" {
		p.UUID = uuid.New().String()
	}
	if p.Budget == 0 {
		p.Budget = 1
	}
	if p.NumThreads == 0 {
		p.NumThreads = runtime.GOMAXPROCS(runtime.NumCPU())
	}
	if p.Nlags == 0 {
		p.Nlags = 5
	}
	if p.StdDevMultiplier == 0 {
		p.StdDevMultiplier = 2.0
	}
	if !p.AllowNegativeBounds {
		p.AllowNegativeBounds = false
	}
}
