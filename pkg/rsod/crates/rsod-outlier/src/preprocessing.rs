// 使用 https://crates.io/crates/augurs-forecaster transforms::LinearInterpolator 进行数据Nan填充

use augurs_forecaster::transforms::interpolate::*;

#[allow(dead_code)]
/// 使用线性插值填充时间序列中的NaN值
/// 
/// # 参数
/// * `data` - 包含NaN值的时间序列数据
/// 
/// # 返回值
/// * 填充了NaN值的新时间序列
/// 
/// # 示例
/// ```
/// use rsod_outlier::preprocessing::fill_nan;
/// 
/// let data = vec![1.0, f64::NAN, f64::NAN, 2.0];
/// let filled = fill_nan(data);
/// assert_eq!(filled, vec![1.0, 1.3333333333333333, 1.6666666666666667, 2.0]);
/// ```
pub fn fill_nan(data: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let data_f64: Vec<f64> = data.iter().map(|x| x[1]).collect();
    let filled = data_f64.into_iter()
        .interpolate(LinearInterpolator::default())
        .collect::<Vec<_>>();
    let mut result = Vec::new();
    for (i, x) in data.iter().enumerate() {
        result.push([x[0], filled[i]]);
    }
    result
}

#[allow(dead_code)]
/// 检查数据中是否包含NaN值
/// 
/// # 参数
/// * `data` - 要检查的数据
/// 
/// # 返回值
/// * 如果包含NaN值返回true，否则返回false
pub fn has_nan(data: &[f64]) -> bool {
    data.iter().any(|x| x.is_nan())
}

#[allow(dead_code)]
/// 统计数据中NaN值的数量
/// 
/// # 参数
/// * `data` - 要统计的数据
/// 
/// # 返回值
/// * NaN值的数量
pub fn count_nan(data: &[f64]) -> usize {
    data.iter().filter(|x| x.is_nan()).count()
}

#[allow(dead_code)]
/// 移除数据中的NaN值
/// 
/// # 参数
/// * `data` - 包含NaN值的数据
/// 
/// # 返回值
/// * 移除NaN值后的数据
pub fn remove_nan(data: Vec<f64>) -> Vec<f64> {
    data.into_iter().filter(|x| !x.is_nan()).collect()
}

#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_nan_basic() {
        let data = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 2.0]];
        let filled = fill_nan(&data);
        
        // 检查第一个和最后一个值保持不变
        assert_eq!(filled[0][1], 1.0);
        assert_eq!(filled[2][1], 2.0);
        
        // 检查中间的值被插值填充
        assert!(!filled[1][1].is_nan());
        assert!(!filled[2][1].is_nan());
        assert!(filled[1][1] > 1.0 && filled[1][1] < 2.0);
        assert!(filled[2][1] > 1.0 && filled[2][1] < 2.0);
    }

    #[test]
    fn test_fill_nan_no_nan() {
        let data = vec![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let filled = fill_nan(&data);
        assert_eq!(filled, data);
    }

    #[test]
    fn test_fill_nan_all_nan() {
        let data = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, f64::NAN]];
        let filled = fill_nan(&data);
        // 如果所有值都是NaN，插值后应该仍然是NaN
        assert_eq!(filled, data);
    }

    #[test]
    fn test_fill_nan_start_nan() {
        let data = vec![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, 1.0], [4.0, 2.0]];
        let filled = fill_nan(&data);
        // 开头的NaN值应该保持不变
        assert!(filled[0][1].is_nan());
        assert!(filled[1][1].is_nan());
        assert_eq!(filled[2][1], 1.0);
        assert_eq!(filled[3][1], 2.0);
    }

    #[test]
    fn test_fill_nan_end_nan() {
        let data = vec![[1.0, 1.0], [2.0, 2.0], [3.0, f64::NAN], [4.0, f64::NAN]];
        let filled = fill_nan(&data);
        // 结尾的NaN值应该保持不变
        assert_eq!(filled[0][1], 1.0);
        assert_eq!(filled[1][1], 2.0);
        assert!(filled[2][1].is_nan());
        assert!(filled[3][1].is_nan());
    }

    #[test]
    fn test_has_nan() {
        assert!(has_nan(&[1.0, f64::NAN, 2.0]));
        assert!(!has_nan(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_count_nan() {
        assert_eq!(count_nan(&[1.0, f64::NAN, 2.0, f64::NAN]), 2);
        assert_eq!(count_nan(&[1.0, 2.0, 3.0]), 0);
    }

    #[test]
    fn test_remove_nan() {
        let data = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
        let cleaned = remove_nan(data);
        assert_eq!(cleaned, vec![1.0, 2.0, 3.0]);
    }
}