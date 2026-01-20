package threshold

import (
	"math"
	"math/rand"
	"sync"
)

// DynamicThresholdCalculator 动态阈值计算器
type DynamicThresholdCalculator struct {
	mu         sync.Mutex
	dataWindow []float64 // 滑动窗口存储数据
	windowSize int       // 窗口大小
}

// NewDynamicThresholdCalculator 初始化计算器
func NewDynamicThresholdCalculator(windowSize int) *DynamicThresholdCalculator {
	return &DynamicThresholdCalculator{
		dataWindow: make([]float64, 0, windowSize),
		windowSize: windowSize,
	}
}

// AddData 向滑动窗口添加数据
func (dtc *DynamicThresholdCalculator) AddData(value float64) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return // 忽略无效值
	}

	dtc.mu.Lock()
	defer dtc.mu.Unlock()

	// 维护窗口大小
	if len(dtc.dataWindow) >= dtc.windowSize {
		// 使用 copy 来移动数据，避免创建新切片
		copy(dtc.dataWindow, dtc.dataWindow[1:])
		dtc.dataWindow = dtc.dataWindow[:len(dtc.dataWindow)-1]
	}
	dtc.dataWindow = append(dtc.dataWindow, value)
}

// CalculateThresholds 计算动态阈值
func (dtc *DynamicThresholdCalculator) CalculateThresholds() (lower, upper float64) {
	dtc.mu.Lock()
	defer dtc.mu.Unlock()

	n := len(dtc.dataWindow)
	if n < 2 {
		return 0, 0
	}

	// 单次遍历计算均值和方差
	mean := 0.0
	m2 := 0.0 // 二阶矩
	for _, x := range dtc.dataWindow {
		mean += x
	}
	mean /= float64(n)

	for _, x := range dtc.dataWindow {
		diff := x - mean
		m2 += diff * diff
	}

	// 使用 n-1 作为除数（贝塞尔校正）
	stdDev := math.Sqrt(m2 / float64(n-1))

	lower = mean - 2*stdDev
	upper = mean + 2*stdDev
	return lower, upper
}

// VoteDetect 使用投票法进行异常检测
func (dtc *DynamicThresholdCalculator) VoteDetect(value float64, voteThreshold int) bool {
	dtc.mu.Lock()
	defer dtc.mu.Unlock()

	if len(dtc.dataWindow) < 2 {
		return false
	}

	// 进行5次检测
	votes := 0
	for i := 0; i < 5; i++ {
		// 每次使用数据的随机80%进行计算
		sampleSize := int(float64(len(dtc.dataWindow)) * 0.8)
		if sampleSize < 2 {
			sampleSize = len(dtc.dataWindow)
		}

		// 随机采样
		sampledData := make([]float64, sampleSize)
		for j := range sampledData {
			randIndex := rand.Intn(len(dtc.dataWindow))
			sampledData[j] = dtc.dataWindow[randIndex]
		}

		// 计算当前样本的均值和标准差
		mean := 0.0
		m2 := 0.0
		for _, x := range sampledData {
			mean += x
		}
		mean /= float64(sampleSize)

		for _, x := range sampledData {
			diff := x - mean
			m2 += diff * diff
		}

		stdDev := math.Sqrt(m2 / float64(sampleSize-1))
		// lower := mean - 2*stdDev
		upper := mean + 2*stdDev

		// 判断是否异常
		if value > upper {
			votes++
		}
	}

	// 返回投票结果，至少3票才判定为异常
	return votes >= voteThreshold
}
