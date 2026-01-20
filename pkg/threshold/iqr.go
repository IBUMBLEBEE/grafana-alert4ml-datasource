package threshold

import (
	"sort"
)

// 计算分位数（线性插值法，等价于numpy/scipy的method='linear'）
func quantile(data []float64, q float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
	sort.Float64s(data)
	pos := q * float64(n-1)
	lower := int(pos)
	upper := lower + 1
	if upper >= n {
		return data[lower]
	}
	weight := pos - float64(lower)
	return data[lower]*(1-weight) + data[upper]*weight
}

// 计算IQR（四分位距）
func IQR(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	// 拷贝一份，避免排序影响原数据
	cpy := make([]float64, len(data))
	copy(cpy, data)
	q75 := quantile(cpy, 0.75)
	q25 := quantile(cpy, 0.25)
	return q75 - q25
}

// IQRAnomalyDetect 返回每个点是否为异常（1为异常，0为正常）
func IQRAnomalyDetect(data []float64, k float64) []float64 {
	if len(data) == 0 {
		return nil
	}
	cpy := make([]float64, len(data))
	copy(cpy, data)
	q75 := quantile(cpy, 0.75)
	q25 := quantile(cpy, 0.25)
	iqr := q75 - q25
	upper := q75 + k*iqr
	lower := q25 - k*iqr
	results := make([]float64, len(data))
	for i, v := range data {
		if v > upper || v < lower {
			results[i] = 1
		} else {
			results[i] = 0
		}
	}
	return results
}

// IQRAnomalyDetects 针对二维数组每列做异常检测
func IQRAnomalyDetects(data [][]float64, k float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}
	nRows := len(data)
	nCols := len(data[0])
	results := make([][]float64, nRows)
	for i := range results {
		results[i] = make([]float64, nCols)
	}
	for col := 0; col < nCols; col++ {
		colData := make([]float64, nRows)
		for row := 0; row < nRows; row++ {
			colData[row] = data[row][col]
		}
		colResult := IQRAnomalyDetect(colData, k)
		for row := 0; row < nRows; row++ {
			results[row][col] = colResult[row]
		}
	}
	return results
}
