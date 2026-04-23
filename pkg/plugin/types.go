package plugin

import (
	"encoding/json"
	"fmt"
	"runtime"
	"time"

	"github.com/IBUMBLEBEE/grafana-alert4ml-datasource/pkg/constant"

	"github.com/google/uuid"
)

type Alert4MLQueryJson struct {
	DetectType        string            `json:"detectType"`
	SupportDetect     string            `json:"supportDetect"`
	SeriesRefId       string            `json:"seriesRefId"`
	HyperParams       json.RawMessage   `json:"hyperParams"`
	Targets           []json.RawMessage `json:"targets,omitempty"`
	ShowAnomalyPoints bool              `json:"showAnomalyPoints"`
	HistoryTimeRange  HistoryTimeRange  `json:"historyTimeRange"`
	UniqueKeys        UniqueKeys        `json:"uniqueKeys"`
}

// HistoryTimeRange stores only the length of the history window.
// The history window's upper bound is always anchored at the current panel's
// timeRange.From (derived at query time, not stored here), so:
//
//	history.to   = panel.timeRange.From
//	history.from = panel.timeRange.From - DurationMs
//
// A custom UnmarshalJSON handles legacy dashboards that stored
// `{from, to}` relative-seconds pairs.
type HistoryTimeRange struct {
	DurationMs uint64 `json:"durationMs"`
}

// UnmarshalJSON accepts three wire formats:
//   - modern: {"durationMs": 300000}
//   - legacy-rel-seconds: {"from": 300, "to": 0}  -> duration = |to-from| * 1000 ms
//   - picker-style: {"mode": "relative", "relativeDuration": "5m"} -> ignored (duration stays 0)
func (h *HistoryTimeRange) UnmarshalJSON(data []byte) error {
	var modern struct {
		DurationMs *uint64 `json:"durationMs"`
	}
	if err := json.Unmarshal(data, &modern); err == nil && modern.DurationMs != nil {
		h.DurationMs = *modern.DurationMs
		return nil
	}
	var legacy struct {
		From *int64 `json:"from"`
		To   *int64 `json:"to"`
	}
	if err := json.Unmarshal(data, &legacy); err == nil && legacy.From != nil && legacy.To != nil {
		diff := *legacy.From - *legacy.To
		if diff < 0 {
			diff = -diff
		}
		h.DurationMs = uint64(diff) * 1000
		return nil
	}
	// Unknown shape (e.g. picker mode object) — leave DurationMs as zero.
	h.DurationMs = 0
	return nil
}

// HyperParams 是所有超参数的接口
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
		params.SetDefaults()
		return &params, nil
	case constant.BaselineDetectTypeDynamics:
		var params DynamicsHyperParams
		if err := json.Unmarshal(data, &params); err != nil {
			return nil, fmt.Errorf("failed to parse DynamicsHyperParams: %w", err)
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
	Periods        string `json:"periods,omitempty"`
	ModelName      string `json:"model_name,omitempty"`
	NTrees         *int   `json:"nTrees,omitempty"`
	SampleSize     *int   `json:"sampleSize,omitempty"`
	MaxTreeDepth   *int   `json:"maxTreeDepth,omitempty"`
	ExtensionLevel *int   `json:"extensionLevel,omitempty"`
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

// DynamicsHyperParams 动态基线检测参数
type DynamicsHyperParams struct {
	Trend            string  `json:"trend,omitempty"`
	PeriodDays       int     `json:"periodDays,omitempty"`
	StdDevMultiplier float64 `json:"stdDevMultiplier,omitempty"`
}

func (p *DynamicsHyperParams) GetType() string {
	return constant.BaselineDetectTypeDynamics
}

func (p *DynamicsHyperParams) SetDefaults() {
	if p.Trend == "" {
		p.Trend = "weekly"
	}
	if p.StdDevMultiplier == 0 {
		p.StdDevMultiplier = 2.0
	}
}

type UniqueKeys struct {
	DashboardUid string `json:"dashboardUid"`
	PanelId      int    `json:"panelId"`
	SeriesRefId  string `json:"seriesRefId"`
}

type ForecastHyperParams struct {
	ModelName           string   `json:"model_name"`
	Periods             string   `json:"periods"`
	UUID                string   `json:"uuid"`
	Budget              float32  `json:"budget,omitempty"`
	NumThreads          int      `json:"numThreads,omitempty"`
	Nlags               int      `json:"nlags,omitempty"`
	StdDevMultiplier    float64  `json:"stdDevMultiplier,omitempty"`
	AllowNegativeBounds bool     `json:"allowNegativeBounds,omitempty"`
	MaxBin              uint16   `json:"maxBin,omitempty"`
	IterationLimit      *int     `json:"iterationLimit,omitempty"`
	Timeout             *float32 `json:"timeout,omitempty"`
	StoppingRounds      *int     `json:"stoppingRounds,omitempty"`
	Seed                *uint64  `json:"seed,omitempty"`
	LogIterations       *int     `json:"logIterations,omitempty"`
}

func (p *ForecastHyperParams) GetType() string {
	return constant.DetectTypeForecast
}

// ForecastTrainingKey contains only parameters that affect model training.
// Used to derive a deterministic UUID so that parameter changes trigger retraining.
type ForecastTrainingKey struct {
	Periods        []uint   `json:"periods"`
	Budget         float32  `json:"budget"`
	NumThreads     int      `json:"num_threads"`
	MaxBin         uint16   `json:"max_bin"`
	IterationLimit *int     `json:"iteration_limit"`
	Timeout        *float32 `json:"timeout"`
	StoppingRounds *int     `json:"stopping_rounds"`
	Seed           *uint64  `json:"seed"`
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
	if p.MaxBin == 0 {
		p.MaxBin = 255
	}
}
