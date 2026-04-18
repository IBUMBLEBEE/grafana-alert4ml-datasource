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

// 支持的检测类型
const (
	SupportDetectTypeBaseline = "baseline"
	SupportDetectTypeML       = "machine_learning"
)

// 基线检测类型
const (
	BaselineDetectTypeDynamics = "dynamics"
)

// IsBaselineDetectType 判断 detectType 是否为 Baseline 子类型
func IsBaselineDetectType(detectType string) bool {
	return detectType == BaselineDetectTypeDynamics
}

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
			{Label: BaselineDetectTypeDynamics, Value: BaselineDetectTypeDynamics},
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
}
