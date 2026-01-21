use statrs::distribution::{Continuous, Normal};
use std::f64;

/// 计算一组数据的样本偏度
pub fn calculate_skewness(data: &[f64], bias: bool) -> Option<f64> {
    if data.len() < 3 {
        return None;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    // 计算二阶和三阶中心矩
    let m2 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    let m3 = data.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;

    // 如果二阶矩接近0，返回None
    if m2.abs() < f64::EPSILON * mean.abs() {
        return None;
    }

    // 计算偏度
    let mut skewness = m3 / m2.powf(1.5);

    // 如果不需要偏差校正，应用校正因子
    if !bias {
        skewness = skewness * ((n - 1.0) * n).sqrt() / (n - 2.0);
    }

    Some(skewness)
}

/// 用正态分布计算每个点的 PDF 值，并对这些 PDF 值计算偏度
pub fn norm_pdf_skew(x: &[[f64; 2]]) -> Option<f64> {
    let data: Vec<f64> = x.iter().map(|v: &[f64; 2]| v[1]).collect();
    if data.len() < 3 {
        return None;
    }

    // 拟合正态分布
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    let normal = Normal::new(mean, std_dev).expect("Invalid normal distribution");

    // 计算每个点的 PDF 值
    let pdf_values: Vec<f64> = data.iter().map(|&x| normal.pdf(x)).collect();

    // 计算 PDF 值的偏度
    calculate_skewness(&pdf_values, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_standard_normal() {
        let data = read_csv_to_vec("data/data1.csv");
        // 标准正态分布 (mean=0, std_dev=1) 的 PDF 应该是完全对称的
        let skewness = norm_pdf_skew(&data).unwrap();
        assert!(skewness != 0.0);
    }
}
