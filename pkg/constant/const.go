package constant

// 用例类型
const (
	UseCasePanel = "Panel"
	UseCaseAlert = "Alert"
)

// 检测类型
const (
	DetectTypeOutlier     = "outlier"
	DetectTypeForecast    = "forecast"
	DetectTypeChangePoint = "changepoint"
)

// LLM检测类型
const (
	LLMDetectTypeDeepseek = "deepseek"
	LLMDetectTypeQwen     = "qwen"
	LLMDetectTypeChatGPT  = "chatgpt"
)

// 支持的检测类型
const (
	SupportDetectTypeBaseline = "baseline"
	SupportDetectTypeML       = "machine_learning"
	SupportDetectTypeLLM      = "llm"
)

// 基线检测类型
const (
	BaselineDetectTypeStd           = "std"
	BaselineDetectTypeZScore        = "zscore"
	BaselineDetectTypeMovingAverage = "moving_average"
)

const (
	GF_FRAME_RESULT_NAME_ANOMALY     = "Anomaly"
	GF_FRAME_RESULT_NAME_BASELINE    = "Baseline"
	GF_FRAME_RESULT_NAME_TIME        = "Time"
	GF_FRAME_RESULT_NAME_LOWER_BOUND = "lower_bound"
	GF_FRAME_RESULT_NAME_UPPER_BOUND = "upper_bound"
	GF_FRAME_RESULT_NAME_FORECAST    = "Pred"
)

const (
	PluginDatasourceType = "ibumblebee-alert4ml-datasource"
)

// RSOD 算法默认值
const (
	DefaultRsodPeriods   = ""
	DefaultRsodModelName = "rsod_model"
	DefaultRsodDBPath    = "rsod_sqlite.db"
)

// 基线算法默认值
const (
	DefaultBaselineTrendType           = "daily"
	DefaultBaselineIntervalMins        = 15
	DefaultBaselineConfidenceLevel     = 95.0
	DefaultBaselineAllowNegativeBounds = false
	DefaultBaselineStdDevMultiplier    = 2.0
)

// 默认时间范围（秒）
const (
	DefaultTimeRangeFrom = 300
	DefaultTimeRangeTo   = 0
)

type SupportDetectOption struct {
	CommKV
	DetectTypes []CommKV
}

type CommKV struct {
	Label string
	Value string
}

var SUPPORT_DETECT_OPTIONS = []SupportDetectOption{
	{
		CommKV: CommKV{
			Label: SupportDetectTypeBaseline,
			Value: SupportDetectTypeBaseline,
		},
		DetectTypes: []CommKV{
			{Label: BaselineDetectTypeStd, Value: BaselineDetectTypeStd},
			{Label: BaselineDetectTypeZScore, Value: BaselineDetectTypeZScore},
			{Label: BaselineDetectTypeMovingAverage, Value: BaselineDetectTypeMovingAverage},
		},
	},
	{
		CommKV: CommKV{
			Label: SupportDetectTypeML,
			Value: SupportDetectTypeML,
		},
		DetectTypes: []CommKV{
			{Label: DetectTypeOutlier, Value: DetectTypeOutlier},
			{Label: DetectTypeChangePoint, Value: DetectTypeChangePoint},
			{Label: DetectTypeForecast, Value: DetectTypeForecast},
		},
	},
	{
		CommKV: CommKV{
			Label: SupportDetectTypeLLM,
			Value: SupportDetectTypeLLM,
		},
		DetectTypes: []CommKV{
			{Label: LLMDetectTypeDeepseek, Value: LLMDetectTypeDeepseek},
			{Label: LLMDetectTypeQwen, Value: LLMDetectTypeQwen},
			{Label: LLMDetectTypeChatGPT, Value: LLMDetectTypeChatGPT},
		},
	},
}
