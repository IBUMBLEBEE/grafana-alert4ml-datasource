use std::f64;

/// 广义帕累托分布结构体
pub struct GeneralizedPareto {
    xm: f64,    // 阈值
    sigma: f64, // 尺度参数
    xi: f64,    // 形状参数
}

impl GeneralizedPareto {
    /// 计算累积分布函数值
    pub fn cdf(&self, x: f64) -> f64 {
        if self.sigma <= 0.0 {
            return 0.0;
        }
        if self.xi == 0.0 {
            return 1.0 - (-x / self.sigma).exp();
        }
        1.0 - (1.0 + self.xi * x / self.sigma).powf(-1.0 / self.xi)
    }
}

/// EVT异常检测器
pub struct EVTAnomalyDetector {
    threshold: f64,         // 阈值分位点
    top_k: usize,           // 用于拟合GPD的极值数量
    gpd: GeneralizedPareto, // 广义帕累托分布
    fitted: bool,           // 是否已拟合
}

impl EVTAnomalyDetector {
    /// 创建新的EVT检测器
    pub fn new(threshold: f64, top_k: usize) -> Self {
        Self {
            threshold,
            top_k,
            gpd: GeneralizedPareto {
                xm: 0.0,
                sigma: 0.0,
                xi: 0.0,
            },
            fitted: false,
        }
    }

    /// 用极值拟合GPD
    pub fn fit(&mut self, scores: &[f64]) {
        let n = scores.len();

        // 自动计算top_k
        let mut top_k = self.top_k;
        if top_k == 0 {
            let ratio = 0.05;
            let min_top_k = 20;
            top_k = (n as f64 * ratio) as usize;
            top_k = top_k.max(min_top_k).min(n);
        }

        // 档满足条件时，应该错误，终止执行
        if n < top_k {
            return;
        }

        // 取top_k极大值
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let extremes = &sorted[n - top_k..];

        // 以最小极值为阈值，计算超出部分
        let u = extremes[0];
        let excesses: Vec<f64> = extremes.iter().map(|&v| v - u).collect();

        // 用样本均值法近似估计shape/scale
        let mean = excesses.iter().sum::<f64>() / excesses.len() as f64;
        let shape = 0.5 * (1.0 - (mean * mean) / (mean * mean)); // 近似，实际可用MLE
        let scale = mean * (1.0 + shape);

        self.gpd = GeneralizedPareto {
            xm: u,
            sigma: scale,
            xi: shape,
        };
        self.fitted = true;
    }

    /// 计算每个分数的p-value，返回异常分数
    pub fn predict(&self, scores: &[f64]) -> Vec<f64> {
        if !self.fitted {
            panic!("EVTAnomalyDetector: must fit before predict");
        }

        let u = self.gpd.xm;
        scores
            .iter()
            .map(|&v| {
                if v <= u {
                    0.0
                } else {
                    // 计算右尾概率
                    let p = 1.0 - self.gpd.cdf(v - u);
                    if p < (1.0 - self.threshold) {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evt_detector() {
        let mut detector = EVTAnomalyDetector::new(0.98, 20);
        let scores = vec![
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
            3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0,
            3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0,
            3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0,
            4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0,
            4.0, 1.0, 2.0, 98.0, 99.0, 100.0,
        ];
        detector.fit(&scores);
        let results = detector.predict(&scores);
        let len = scores.len();
        assert_eq!(results[len - 1], 1.0);
        assert_eq!(results[len - 2], 1.0);
        assert_eq!(results[len - 3], 1.0);
    }
}
