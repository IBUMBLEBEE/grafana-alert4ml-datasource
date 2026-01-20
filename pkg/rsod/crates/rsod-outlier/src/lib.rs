extern crate rsod_utils;

mod auto_mstl;
mod evt;
mod ext_iforest;
mod iqr;
mod seasons;
mod skew;
mod stl;
mod preprocessing; 

use augurs::changepoint::{DefaultArgpcpDetector, Detector};
use auto_mstl::auto_mstl;
pub use ext_iforest::{
    iforest, load_iforest_model, predict_with_saved_model, save_iforest_model, EIFOptions,
    SavedIForestModel,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierOptions {
    pub model_name: String,
    pub periods: Vec<usize>,
    pub uuid: String,
}

/// 检测数组中的异常值
///
/// 支持周期性分组检测，periods 为多个周期长度
///
/// # 参数
/// * `data` - 输入数据
/// * `periods` - 周期长度数组
/// * `uuid` - 模型的唯一标识符，用于保存和加载模型
///
/// 返回异常0,1，1表示异常
pub fn outlier(data: &[[f64; 2]], periods: &[usize], uuid: &str) -> Result<Vec<f64>, std::io::Error> {
    println!("iforestuuid");
    if data.is_empty() {
        return Ok(vec![]);
    }

    // let data_filled = fill_nan(data);
    let data_filled_f32: Vec<[f32; 2]> = data.iter().map(|x| [x[0] as f32, x[1] as f32]).collect();

    // let pvalue = adf(data_filled);
    // if pvalue < STATIONARY_P_VALUE {
    //     // 时序平稳，使用EIF检测
    //     return ensemble_detect(data);
    // }
    // let data_f32: Vec<[f32; 2]> = data.iter().map(|x| [x[0] as f32, x[1] as f32]).collect();
    let mres = auto_mstl(&data_filled_f32, periods);
    if mres.periods.len() > 0 {
        // 有周期，使用残差进行检测
        // residual 使用eif检测，residual + trend 使用changepoint检测
        let residual_2d: Vec<[f64; 2]> = mres
            .residual
            .iter()
            .enumerate()
            .map(|(i, &v)| [i as f64, v as f64])
            .collect();

        // 并发执行 EIF 和 changepoint 检测
        let uuid_clone = uuid.to_string();
        let residual_clone = residual_2d.clone();
        let (eif_scores, changepoints) = rayon::join(
            || {
                let options = EIFOptions {
                    n_trees: 100,
                    sample_size: Some(256),
                    max_tree_depth: None,
                    extension_level: Some(0),
                };
                iforest(uuid_clone.clone(), options, &residual_clone)
            },
            || {
                let deseasonalized_2d: Vec<[f64; 2]> = mres
                    .trend
                    .iter()
                    .zip(mres.residual.iter())
                    .enumerate()
                    .map(|(i, (&t, &r))| [i as f64, t as f64 + r as f64])
                    .collect();
                DefaultArgpcpDetector::default().detect_changepoints(
                    &deseasonalized_2d.iter().map(|x| x[1]).collect::<Vec<f64>>(),
                )
            },
        );

        // eif_scores 进行阈值处理
        let eif_scores = eif_scores?;
        let eif_scores_threshold =
            outlier_threshold(&residual_2d.clone(), &eif_scores).unwrap();
        // 合并结果
        let mut outlier_result = eif_scores_threshold;
        // 将changepoints里小于5的值移除
        let mut changepoints = changepoints;
        changepoints.retain(|&cp| cp >= 5);
        for cp in changepoints {
            outlier_result[cp as usize] = 1.0;
        }
        return Ok(outlier_result);
    } else {
        // 没有周期，数据平稳性未知
        return ensemble_detect(data, uuid);
    }
}

fn ensemble_detect(data: &[[f64; 2]], uuid: &str) -> Result<Vec<f64>, std::io::Error> {
    let options = EIFOptions {
        n_trees: 100,
        sample_size: Some(256),
        max_tree_depth: None,
        extension_level: Some(0),
    };
    // 使用并发计算 eif 和 changepoint 的异常检测结果
    let uuid_clone = uuid.to_string();
    let data_clone = data.to_vec();
    let (eif_scores, changepoints) = rayon::join(
        || iforest(uuid_clone, options, &data_clone),
        || {
            DefaultArgpcpDetector::default()
                .detect_changepoints(&data.iter().map(|x| x[1]).collect::<Vec<f64>>())
        },
    );

    // 将changepoints的坐标与eif_outlier的坐标合并，返回新的异常检测结果
    let eif_scores = eif_scores?;
    let eif_scores_threshold = outlier_threshold(&data, &eif_scores).unwrap();
    let mut outlier_result = eif_scores_threshold;
    // 将changepoints里小于5的值移除，因为在changepoint里，前面几个没有上下文参考，无法判断是否异常
    let mut changepoints = changepoints;
    changepoints.retain(|&cp| cp >= 5);
    for cp in changepoints {
        outlier_result[cp as usize] = 1.0;
    }
    return Ok(outlier_result);
}

/// 异常值检测的阈值常量
const HIGH_SKEW_THRESHOLD: f64 = 1.0;
const MEDIUM_SKEW_THRESHOLD: f64 = 0.8;
const EVT_THRESHOLD: f64 = 0.9;
const IQR_LOWER_PERCENTILE: f64 = 1.0;
const IQR_UPPER_PERCENTILE: f64 = 99.0;

/// 根据偏度值对异常分数进行阈值处理
///
/// # 参数
///
/// * `x` - 输入数据
/// * `scores` - 异常分数
///
/// # 返回
///
/// 返回处理后的异常分数（0或1）
///
/// # 错误
///
/// 如果偏度计算失败，返回错误信息
pub fn outlier_threshold(x: &[[f64; 2]], scores: &[f64]) -> Result<Vec<f64>, String> {
    // 计算偏度
    let skew_val = skew::norm_pdf_skew(x)
        .ok_or_else(|| "偏度计算失败".to_string())?
        .abs();

    // 根据偏度值选择不同的处理方法
    let result = if skew_val >= HIGH_SKEW_THRESHOLD {
        // 高偏态使用 EVT 方法
        let mut evt_detector = evt::EVTAnomalyDetector::new(EVT_THRESHOLD, 10);
        evt_detector.fit(scores);
        evt_detector.predict(scores)
    } else if skew_val < HIGH_SKEW_THRESHOLD && skew_val >= MEDIUM_SKEW_THRESHOLD {
        // 中度偏态使用 IQR 方法
        iqr::iqr_anomaly_detection(
            scores,
            5,
            Some(IQR_UPPER_PERCENTILE),
            Some(IQR_LOWER_PERCENTILE),
            Some(2.0),
        )
    } else {
        // 低偏态，默认无异常值
        vec![0.0; scores.len()]
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_outlier_score() {
        let data: Vec<[f64; 2]> = read_csv_to_vec("data/data.csv");

        let result = outlier(&data, &[], "test-outlier-uuid").unwrap();
        assert_eq!(result.len(), data.len());
        // 拥有异常点
        assert!(result.clone().iter().copied().fold(0.0, f64::max) > 0.5);
        let mut outlier_data = Vec::new();
        for i in 0..result.len() {
            outlier_data.push([i as f64, result[i] as f64]);
        }
        // assert!(result[13] == 1.0); // 100.0的异常分数应该大于1.0的异常分数
    }
}
