// 实现 extended-isolation-forest 基于时间序列的孤立森林算法
// 参考 https://github.com/nmandery/extended-isolation-forest

use extended_isolation_forest::{Forest, ForestOptions};
use serde::{Serialize, Deserialize};
use rsod_storage::model::Model;
use std::io::{Error, ErrorKind};

/// 保存 iForest 模型的数据结构
/// 启用 serde feature 后，可以直接序列化 Forest 对象
#[derive(Serialize, Deserialize)]
pub struct SavedIForestModel {
    /// 训练好的 Forest 模型（启用 serde feature 后支持直接序列化）
    pub forest: Forest<f64, 4>,
    /// 标准化参数：均值
    pub mean: Vec<f64>,
    /// 标准化参数：标准差
    pub std_dev: Vec<f64>,
}

// 手动实现 Debug（Forest 不支持 Debug，只打印基本信息）
impl std::fmt::Debug for SavedIForestModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SavedIForestModel")
            .field("mean", &self.mean)
            .field("std_dev", &self.std_dev)
            .field("forest", &"<Forest<f64, 4>>")
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct EIFOptions {
    pub n_trees: usize,
    pub sample_size: Option<usize>,
    pub max_tree_depth: Option<usize>,
    pub extension_level: Option<usize>,
}

/// 训练 iForest 模型并返回异常分数
/// 
/// # 参数
/// * `uuid` - 模型的唯一标识符，用于检查 SQLite 中是否已有保存的模型
/// * `options` - Forest 训练选项（仅在需要训练新模型时使用）
/// * `data` - 训练/预测数据
/// 
/// # 行为
/// 1. 先检查 SQLite 中是否存在 uuid 对应的模型
/// 2. 如果存在，直接加载模型并使用其进行异常检测
/// 3. 如果不存在，训练新模型，保存到 SQLite，然后进行异常检测
/// 
/// # 返回
/// 异常分数向量
pub fn iforest(uuid: String, options: EIFOptions, data: &[[f64; 2]]) -> Result<Vec<f64>, Error> {
    // 先尝试加载已保存的模型
    match load_iforest_model(uuid.clone()) {
        Ok(saved_model) => {
            // 模型存在，使用保存的模型进行预测
            let normalized_features =
                normalize_features_with_params(data, &saved_model.mean, &saved_model.std_dev);
            
            let scores: Vec<f64> = normalized_features
                .iter()
                .map(|x| saved_model.forest.score(x))
                .collect();
            
            Ok(scores)
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            // 模型不存在，训练新模型
            let (forest, normalized_features, mean, std_dev) = train_iforest(options.clone(), data);
            
            // 计算异常分数
            let scores: Vec<f64> = normalized_features
                .iter()
                .map(|x| forest.score(x))
                .collect();
            
            // 保存模型到 SQLite
            let saved_model = SavedIForestModel {
                forest,
                mean,
                std_dev,
            };
            
            let serialized = bincode::serialize(&saved_model)
                .map_err(|e| Error::new(ErrorKind::Other, format!("序列化失败: {}", e)))?;
            
            let model = Model::new(uuid, serialized);
            model.write()?;
            
            Ok(scores)
        }
        Err(e) => {
            // 其他错误（如反序列化失败等），返回错误
            Err(e)
        }
    }
}

/// 训练 iForest 模型并返回 Forest、标准化后的特征和标准化参数
/// 
/// # 返回
/// (Forest, normalized_features, mean, std_dev)
pub fn train_iforest(
    mut options: EIFOptions,
    data: &[[f64; 2]],
) -> (Forest<f64, 4>, Vec<[f64; 4]>, Vec<f64>, Vec<f64>) {
    let (normalized_features, mean, std_dev) = extract_and_normalize_features(data);

    // 自动计算 sample_size：数据长度的 1/10，向下取整，最小值为 1
    options.sample_size = options
        .sample_size
        .or_else(|| Some(((data.len() as f64 * 0.1).floor() as usize).max(1)))
        .map(|size| size.min(normalized_features.len()));

    // 自动计算 extension_level
    options.extension_level = options.extension_level.or_else(|| {
        if normalized_features.is_empty() {
            Some(0)
        } else {
            Some((normalized_features[0].len() - 1) as usize)
        }
    });

    let forest_options = ForestOptions {
        n_trees: options.n_trees,
        sample_size: options.sample_size.unwrap(),
        max_tree_depth: options.max_tree_depth,
        extension_level: options.extension_level.unwrap(),
    };

    let forest = Forest::from_slice(normalized_features.as_slice(), &forest_options).unwrap();
    (forest, normalized_features, mean, std_dev)
}

/// 提取并标准化特征
/// 
/// # 返回
/// (normalized_features, mean, std_dev)
fn extract_and_normalize_features(data: &[[f64; 2]]) -> (Vec<[f64; 4]>, Vec<f64>, Vec<f64>) {
    // 数据预处理：提取时间特征
    let values: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let week_over_week = calc_week_over_week(&values, 6);
    let day_over_day = calc_day_over_day(&values, 6);
    let ma = moving_average(&values, 5);
    let std = moving_std(&values, 5);

    let features: Vec<Vec<f64>> = vec![
        data.iter().map(|x| x[0]).collect(), // time
        values.clone(),                      // value
        (0..data.len())
            .map(|i| if i > 0 { values[i - 1] } else { values[0] })
            .collect(), // lag_value
        (0..data.len())
            .map(|i| {
                if i >= 5 {
                    values[i - 5..i].iter().sum::<f64>() / 5.0
                } else {
                    values[0]
                }
            })
            .collect(), // moving_avg
        // 新特征
        (0..data.len())
            .map(|i| {
                if i >= 6 {
                    week_over_week[i - 6]
                } else {
                    f64::NAN
                }
            })
            .collect(),
        (0..data.len())
            .map(|i| {
                if i >= 6 {
                    day_over_day[i - 6]
                } else {
                    f64::NAN
                }
            })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { ma[i - 5] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { std[i - 5] } else { f64::NAN })
            .collect(),
    ];

    // 转置为 Vec<[f64; N]>
    let features: Vec<Vec<f64>> = (0..data.len())
        .map(|i| features.iter().map(|col| col[i]).collect())
        .collect();

    // 数据标准化处理
    let mean: Vec<f64> = (0..4)
        .map(|j| features.iter().map(|x| x[j]).sum::<f64>() / features.len() as f64)
        .collect();
    let std_dev: Vec<f64> = (0..4)
        .map(|j| {
            (features
                .iter()
                .map(|x| (x[j] - mean[j]).powi(2))
                .sum::<f64>()
                / features.len() as f64)
                .sqrt()
        })
        .collect();
    let normalized_features: Vec<[f64; 4]> = features
        .iter()
        .map(|x| {
            let mut normalized = [0.0; 4];
            for j in 0..4 {
                normalized[j] = (x[j] - mean[j]) / std_dev[j];
            }
            normalized
        })
        .collect();

    (normalized_features, mean, std_dev)
}

/// 保存 iForest 模型到 SQLite 数据库
/// 
/// # 参数
/// * `uuid` - 模型的唯一标识符
/// * `options` - Forest 训练选项
/// * `data` - 训练数据
/// 
/// # 返回
/// 成功返回 `Ok(())`，失败返回 `Error`
/// 
/// # 示例
/// ```no_run
/// use rsod_outlier::ext_iforest::{save_iforest_model, EIFOptions};
/// 
/// let options = EIFOptions {
///     n_trees: 100,
///     sample_size: Some(256),
///     max_tree_depth: None,
///     extension_level: Some(0),
/// };
/// let data = vec![[1.0, 2.0], [2.0, 3.0]];
/// save_iforest_model("model-uuid".to_string(), options, &data).unwrap();
/// ```
pub fn save_iforest_model(uuid: String, options: EIFOptions, data: &[[f64; 2]]) -> Result<(), Error> {
    let (forest, _, mean, std_dev) = train_iforest(options, data);

    let saved_model = SavedIForestModel {
        forest,
        mean,
        std_dev,
    };

    // 使用 bincode 序列化（二进制格式，比 JSON 更高效）
    let serialized = bincode::serialize(&saved_model)
        .map_err(|e| Error::new(ErrorKind::Other, format!("序列化失败: {}", e)))?;

    // 保存到 SQLite 数据库
    let model = Model::new(uuid, serialized);
    model.write()?;

    Ok(())
}

/// 从 SQLite 数据库加载 iForest 模型
/// 
/// # 参数
/// * `uuid` - 模型的唯一标识符
/// 
/// # 返回
/// 成功返回 `Ok(SavedIForestModel)`，失败返回 `Error`
/// 
/// # 示例
/// ```no_run
/// use rsod_outlier::ext_iforest::load_iforest_model;
/// 
/// let saved_model = load_iforest_model("model-uuid".to_string()).unwrap();
/// let forest = &saved_model.forest;
/// ```
pub fn load_iforest_model(uuid: String) -> Result<SavedIForestModel, Error> {
    let mut model = Model::new(uuid.clone(), vec![]);
    model.read()?;

    if model.artifacts.is_empty() {
        return Err(Error::new(
            ErrorKind::NotFound,
            format!("模型 {} 不存在", uuid),
        ));
    }

    // 使用 bincode 反序列化
    let saved_model: SavedIForestModel = bincode::deserialize(&model.artifacts)
        .map_err(|e| Error::new(ErrorKind::InvalidData, format!("反序列化失败: {}", e)))?;

    Ok(saved_model)
}

/// 使用保存的标准化参数对新数据进行标准化
fn normalize_features_with_params(
    data: &[[f64; 2]],
    mean: &[f64],
    std_dev: &[f64],
) -> Vec<[f64; 4]> {
    let (_features, _, _) = extract_and_normalize_features(data);
    // 重新标准化，使用保存的参数
    let values: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let week_over_week = calc_week_over_week(&values, 6);
    let day_over_day = calc_day_over_day(&values, 6);
    let ma = moving_average(&values, 5);
    let std = moving_std(&values, 5);

    let feature_cols: Vec<Vec<f64>> = vec![
        data.iter().map(|x| x[0]).collect(),
        values.clone(),
        (0..data.len())
            .map(|i| if i > 0 { values[i - 1] } else { values[0] })
            .collect(),
        (0..data.len())
            .map(|i| {
                if i >= 5 {
                    values[i - 5..i].iter().sum::<f64>() / 5.0
                } else {
                    values[0]
                }
            })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 6 { week_over_week[i - 6] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 6 { day_over_day[i - 6] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { ma[i - 5] } else { f64::NAN })
            .collect(),
        (0..data.len())
            .map(|i| if i >= 5 { std[i - 5] } else { f64::NAN })
            .collect(),
    ];

    let feature_rows: Vec<Vec<f64>> = (0..data.len())
        .map(|i| feature_cols.iter().map(|col| col[i]).collect())
        .collect();

    feature_rows
        .iter()
        .map(|x| {
            let mut normalized = [0.0; 4];
            for j in 0..4 {
                normalized[j] = (x[j] - mean[j]) / std_dev[j];
            }
            normalized
        })
        .collect()
}

/// 使用已保存的模型进行预测
/// 
/// # 参数
/// * `uuid` - 模型的唯一标识符
/// * `data` - 待预测的数据
/// 
/// # 返回
/// 成功返回异常分数向量，失败返回 `Error`
/// 
/// # 示例
/// ```no_run
/// use rsod_outlier::ext_iforest::predict_with_saved_model;
/// 
/// let data = vec![[1.0, 2.0], [2.0, 3.0]];
/// let scores = predict_with_saved_model("model-uuid".to_string(), &data).unwrap();
/// ```
pub fn predict_with_saved_model(uuid: String, data: &[[f64; 2]]) -> Result<Vec<f64>, Error> {
    let saved_model = load_iforest_model(uuid)?;

    // 使用保存的标准化参数对新数据进行标准化
    let normalized_features =
        normalize_features_with_params(data, &saved_model.mean, &saved_model.std_dev);

    // 使用保存的 Forest 进行预测
    let scores: Vec<f64> = normalized_features
        .iter()
        .map(|x| saved_model.forest.score(x))
        .collect();

    Ok(scores)
}

// 周同比
fn calc_week_over_week(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..data.len() {
        result.push(data[i] / data[i - win]);
    }
    result
}

// 日环比
fn calc_day_over_day(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..data.len() {
        result.push(data[i] / data[i - win]);
    }
    result
}

// 移动平均
fn moving_average(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..=data.len() {
        let avg = data[i - win..i].iter().sum::<f64>() / win as f64;
        result.push(avg);
    }
    result
}

// 标准差
fn moving_std(data: &[f64], win: usize) -> Vec<f64> {
    let mut result = Vec::new();
    for i in win..=data.len() {
        let slice = &data[i - win..i];
        let mean = slice.iter().sum::<f64>() / win as f64;
        let std = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / win as f64).sqrt();
        result.push(std);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_make_f64_forest() {
        let now = 1609459200.0; // 2021-01-01 00:00:00 UTC
        let mut rng = rand::thread_rng();
        let mut data: Vec<[f64; 2]> = (0..500)
            .map(|i| {
                let time = now - (i as f64 * 3600.0);
                let value = (i as f64 * 0.1).sin() + rng.gen_range(-0.1..0.1); // 正弦波 + 随机噪声
                [time, value]
            })
            .collect();

        // 插入异常值
        data[0][1] = 0.0; // 第一个点异常
        data[10][1] = 2.0; // 第10个点异常
        data[11][1] = 1.5; // 第11个点异常
        data[12][1] = 2.2; // 第12个点异常
        data[13][1] = 0.9; // 第13个点异常

        data[110][1] = 1.3; // 第110个点异常
        data[111][1] = 9.0; // 第111个点异常
        data[112][1] = 5.0; // 第112个点异常
        data[113][1] = 4.0; // 第113个点异常

        // // 在中间时间点插入异常值
        // data[500][1] = 10.0; // 第500个点异常
        // data[501][1] = 10.0; // 第501个点异常
        // data[502][1] = 10.0; // 第502个点异常

        // // 在末尾时间点插入异常值
        // data[990][1] = 11.0; // 第990个点异常
        // data[991][1] = 11.0; // 第991个点异常
        // data[992][1] = 11.0; // 第992个点异常

        let options = EIFOptions {
            n_trees: 500,
            sample_size: None,
            max_tree_depth: None,
            extension_level: None,
        };
        let scores = iforest("test-model-uuid".to_string(), options, &data).unwrap();

        // 验证返回的分数数量与输入数据长度一致
        assert_eq!(scores.len(), data.len());

        // 验证分数在合理范围内（0到1之间）
        println!("index 0: {:?}", scores[0]);
        println!("index 11: {:?}", scores[11]);
        println!("index 12: {:?}", scores[12]);
        println!("index 13: {:?}", scores[13]);

        println!("index 110: {:?}", scores[110]);
        println!("index 111: {:?}", scores[111]);
        println!("index 112: {:?}", scores[112]);
        println!("index 113: {:?}", scores[113]);

        assert!(scores[11] > 0.5);
        assert!(scores[12] > 0.5);
        assert!(scores[13] > 0.5);
        // 验证正常值的分数较低
        assert!(scores[0] < scores[13]);
    }
}
