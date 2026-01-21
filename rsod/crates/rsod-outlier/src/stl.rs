// reference: https://github.com/ankane/stl-rust

use serde::{Deserialize, Serialize};
use stlrs::{Mstl, Stl};

#[derive(Debug, Serialize, Deserialize)]
pub struct DecompositionResult {
    pub seasonal: Vec<Vec<f32>>,
    pub trend: Vec<f32>,
    pub residual: Vec<f32>,
    pub periods: Vec<usize>,
}

const SEASONAL_STRENGTH_THRESHOLD: f64 = 0.8;

/// Perform time series decomposition using STL
///
/// # Parameters
///
/// * `data` - Input time series data
/// * `period` - Seasonal periods
///
/// # Returns
///
/// Returns decomposition results containing trend, seasonal and residual components
pub fn decompose(data: &[f32], periods: &mut Vec<usize>) -> DecompositionResult {
    // Validate input data
    if data.is_empty() {
        panic!("Input data cannot be empty");
    }
    // 检查periods的值是否大于data的长度, 如果大于，则删除该值
    if periods.iter().any(|&p| p * 2 > data.len()) {
        periods.retain(|&p| p * 2 <= data.len());
    }
    if data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        panic!("Input data contains NaN or infinite values");
    }
    if periods.is_empty() {
        return DecompositionResult {
            trend: Vec::new(),
            seasonal: Vec::new(),
            residual: Vec::new(),
            periods: Vec::new(),
        }
    }

    // Perform STL decomposition
    let sltparams = Stl::params()
        .seasonal_length(25) // For minimum period 12, set to 2x+1 for smoother results
        .trend_length(1201) // Much larger than longest period (720), recommended 1.5~2x
        .low_pass_length(25) // Same as seasonal_length
        .seasonal_degree(0) // No polynomial fitting (avoid overfitting)
        .trend_degree(1) // First order trend line
        .low_pass_degree(1) // Default value
        .seasonal_jump(1) // Regression for each point, high precision
        .trend_jump(2) // Improve performance
        .low_pass_jump(1)
        .inner_loops(2) // Sufficient for seasonal & trend convergence
        .outer_loops(15) // Enhance robustness, avoid trend interference from outliers
        .robust(true) // Enable robust processing (insensitive to outliers)
        .clone();
    let periodsclone = periods.clone();
    let seasonal_lengths = compute_seasonal_lengths(periodsclone, data.len());

    let mresult = Mstl::params()
        // .iterations(2)                   // number of iterations
        // .lambda(0.5)                     // lambda for Box-Cox transformation
        .seasonal_lengths(&seasonal_lengths) // 使用传入的period作为seasonal_lengths
        .stl_params(sltparams)
        .fit(data, periods) // STL params
        .expect("MSTL decomposition failed");

    for (i, seasonal_component) in mresult.seasonal().iter().enumerate() {
        let strength = seasonal_strength(seasonal_component, mresult.remainder());
        if strength < SEASONAL_STRENGTH_THRESHOLD {
            periods.remove(i);
        }
    }

    DecompositionResult {
        trend: mresult.trend().to_vec(),
        seasonal: mresult.seasonal().to_vec(),
        residual: mresult.remainder().to_vec(),
        periods: periods.to_vec(),
    }
}

fn compute_seasonal_lengths(periods: Vec<usize>, data_length: usize) -> Vec<usize> {
    // If data length is less than twice the minimum period, return default value [37]
    if periods.iter().any(|&p| data_length < p * 2) {
        return vec![37];
    }

    periods
        .iter()
        .map(|&p| {
            let mut length = ((p as f32) * 1.5).ceil() as usize;
            if length % 2 == 0 {
                length += 1; // Round up to nearest odd number
            }
            length
        })
        .collect()
}

fn seasonal_strength(seasonal: &[f32], remainder: &[f32]) -> f64 {
    let s_plus_r: Vec<f64> = seasonal
        .iter()
        .zip(remainder.iter())
        .map(|(s, r)| *s as f64 + *r as f64)
        .collect();

    let var_r = variance(remainder.iter().map(|r| *r as f64));
    let var_sr = variance(s_plus_r.into_iter());

    if var_sr.abs() < 1e-8 {
        0.0 // 防止除以零
    } else {
        1.0 - (var_r / var_sr)
    }
}

fn variance<I: Iterator<Item = f64>>(data: I) -> f64 {
    let data: Vec<f64> = data.collect();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;
    // use std::fs::File;
    // use std::io::Write;

    #[test]
    fn test_decompose_with_csv_data() {
        // 读取CSV数据
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/data.csv");

        // 提取时间序列数据
        let time_series: Vec<f32> = data
            .iter()
            .map(|x| x[1] as f32)
            .filter(|&x| !x.is_nan() && !x.is_infinite())
            .collect();

        // 设置周期（日周期）
        let periods = [24];

        // 执行分解
        let result = decompose(&time_series, &mut periods.to_vec());

        // 验证结果
        assert_eq!(result.trend.len(), time_series.len());
        assert_eq!(result.residual.len(), time_series.len());
        assert_ne!(result.periods.len(), 0);

        // // 绘制分解结果
        // let mut trend_data = Vec::new();
        // let mut seasonal_data = Vec::new();
        // let mut residual_data = Vec::new();

        // for i in 0..time_series.len() {
        //     trend_data.push([i as f64, result.trend[i] as f64]);
        //     for j in 0..result.seasonal.len() {
        //         seasonal_data.push([i as f64, result.seasonal[j][i] as f64]);
        //     }
        //     residual_data.push([i as f64, result.residual[i] as f64]);
        // }

        // plot_time_series(&trend_data, "data/stl_trend.png");
        // plot_time_series(&seasonal_data, "data/stl_seasonal.png");
        // plot_time_series(&residual_data, "data/stl_residual.png");

        // // ==========================二次分解===========================================

        // // 对result.residual进行季节分解
        // let period = [168, 144, 296];
        // let residual_data: Vec<f32> = result.residual.iter()
        //     .filter(|&&x| !x.is_nan() && !x.is_infinite() && x.abs() < 1e6)  // 过滤掉异常值
        //     .map(|&x| x)
        //     .collect();

        // let mut residual_data_plot = Vec::new();
        // for i in 0..residual_data.len() {
        //     residual_data_plot.push([i as f64, residual_data.clone()[i] as f64]);
        // }

        // plot_time_series(&residual_data_plot, "data/stl_residual_residual.png");

        // // 确保数据长度足够
        // if residual_data.len() < period[0] * 2 {
        //     return;
        // }

        // let result_residual2 = decompose(&residual_data, &period);

        // // 绘制result_residual分解结果
        // let mut trend_data = Vec::new();
        // let mut seasonal_data = Vec::new();
        // let mut residual_data = Vec::new();

        // for i in 0..residual_data.len() {
        //     trend_data.push([i as f64, result_residual2.trend[i] as f64]);
        //     seasonal_data.push([i as f64, result_residual2.seasonal[0][i] as f64]);
        //     residual_data.push([i as f64, result_residual2.residual[i] as f64]);
        // }
        // plot_time_series(&trend_data, "data/stl_trend_residual.png");
        // plot_time_series(&seasonal_data, "data/stl_seasonal_residual.png");
        // plot_time_series(&residual_data, "data/stl_residual_residual.png");

        // ==========================从csv中读取数据，进行二次分解===========================================
        // let period = [168, 144, 296];
        // let residual_data_csv: Vec<f32> = read_csv_to_vec("data/stl_residual_seasonal2.csv").iter()
        //     .map(|x| x[1] as f32)
        //     .collect();

        // // 对result.residual进行季节分解
        // let period = [168, 144, 296];
        // let residual_data_csv: Vec<f32> = result
        //     .residual
        //     .iter()
        //     .filter(|&&x| !x.is_nan() && !x.is_infinite() && x.abs() < 1e6) // 过滤掉异常值
        //     .map(|&x| x)
        //     .collect();

        // let result_residual = decompose(&residual_data_csv, &mut period.to_vec());
        // let mut residual_data_residual = Vec::new();
        // let mut residual_data_seasonal = Vec::new();
        // let mut residual_data_trend = Vec::new();
        // for i in 0..result_residual.residual.len() {
        //     residual_data_residual.push([i as f64, result_residual.residual[i] as f64]);
        //     residual_data_seasonal.push([i as f64, result_residual.seasonal[0][i] as f64]);
        //     residual_data_trend.push([i as f64, result_residual.trend[i] as f64]);
        // }

        // // residual_data_residual 写入csv, 索引不写
        // let mut file = File::create("data/stl_residual_residual3.csv").unwrap();
        // for i in 0..result.residual.len() {
        //     writeln!(file, "{},{}", i, result_residual.residual[i]).unwrap();
        // }

        // plot_time_series(&residual_data_residual, "data/stl_residual_residual3.png");
        // plot_time_series(&residual_data_seasonal, "data/stl_residual_seasonal3.png");
        // plot_time_series(&residual_data_trend, "data/stl_residual_trend3.png");

        // 对result_residual.residual进行季节分解
    }
}
