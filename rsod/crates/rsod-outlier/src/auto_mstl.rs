use crate::seasons::PeriodDetector;
use crate::stl::decompose;
use itertools::Itertools;

/// Represents the result of an automatic MSTL decomposition.
///
/// This structure contains all components of the decomposed time series:
/// - trend: The long-term trend component
/// - seasonal: Multiple seasonal components, one for each detected period
/// - residual: The remaining component after removing trend and seasonal components
/// - periods: The detected seasonal periods
#[allow(dead_code)]
pub struct AutoMSTLDecompositionResult {
    pub trend: Vec<f32>,
    pub seasonal: Vec<Vec<f32>>,
    pub residual: Vec<f32>,
    pub periods: Vec<usize>,
}

/// Performs automatic Multiple Seasonal-Trend decomposition using Loess (MSTL) on time series data.
///
/// This function automatically detects seasonal periods and decomposes the time series into trend,
/// seasonal, and residual components. If periods are provided, it directly uses MSTL decomposition.
/// Otherwise, it iteratively detects periods and performs decomposition up to a maximum number of iterations.
///
/// # Arguments
///
/// * `data` - A slice of [timestamp, value] pairs representing the time series data
/// * `periods` - Optional slice of known seasonal periods. If empty, periods will be automatically detected
///
/// # Returns
///
/// Returns an `AutoMSTLDecompositionResult` containing:
/// * trend: The trend component of the time series
/// * seasonal: Vector of seasonal components for each detected period
/// * residual: The residual component after removing trend and seasonal components
/// * periods: Vector of detected seasonal periods
///
/// # Algorithm
///
/// 1. If periods are provided, performs direct MSTL decomposition
/// 2. Otherwise, iteratively:
///    - Detects potential seasonal periods using period detection
///    - Performs MSTL decomposition
///    - Updates residual for next iteration
///    - Continues until no new periods are found or max iterations reached
/// 3. Removes duplicate periods and periods that are too long
#[allow(dead_code)]
pub fn auto_mstl(data: &[[f32; 2]], periods: &[usize]) -> AutoMSTLDecompositionResult {
    let mut y = data.iter().map(|x| x[1]).collect::<Vec<f32>>();
    // If periods is not empty, directly use mstl for period decomposition
    if periods.len() > 0 {
        let mut periods_clone = periods.to_vec();
        let mres = decompose(&y.clone(), &mut periods_clone);
        let auto_mstl_decom = AutoMSTLDecompositionResult {
            trend: mres.trend,
            seasonal: mres.seasonal,
            residual: mres.residual,
            periods: periods_clone,
        };
        return auto_mstl_decom;
    }

    let mut new_residual: Vec<f32> = Vec::new();
    let mut new_periods: Vec<usize> = Vec::new();
    let mut new_seasonal: Vec<Vec<f32>> = Vec::new();
    let mut new_trend: Vec<f32> = Vec::new();

    let detector = PeriodDetector {
        window_size: 2,
        min_peak_threshold: 0.8,
    };

    let loop_max = 3;
    let mut loop_count = 0;

    loop {
        if loop_count >= loop_max {
            break;
        }
        let mut pd_periods = detector.detect_periods(
            &y.iter().map(|x| *x as f64).collect::<Vec<f64>>().as_slice(),
            1,
            1,
            24 * 7,
        );
        if pd_periods.is_empty() {
            break;
        }

        // let mut pd_periods = pd_periods;
        let mres = decompose(&y, &mut pd_periods);
        if mres.periods.is_empty() {
            break;
        }
        y = mres.residual.clone();
        new_residual = mres.residual.clone();
        for seasonal in mres.seasonal {
            new_seasonal.push(seasonal);
        }
        new_trend = mres.trend.clone();
        new_periods.extend(mres.periods.iter().map(|x| *x));
        loop_count += 1;
    }

    // Remove duplicates from new_periods
    new_periods = new_periods.into_iter().unique().collect();
    // If a value in new_periods is greater than the data length, remove it
    new_periods.retain(|&p| p * 2 <= data.len());

    let auto_mstl_decom = AutoMSTLDecompositionResult {
        trend: new_trend,
        seasonal: new_seasonal,
        residual: new_residual,
        periods: new_periods,
    };
    auto_mstl_decom
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    /// Tests the automatic MSTL decomposition functionality.
    ///
    /// This test:
    /// 1. Reads test data from a CSV file
    /// 2. Converts the data to the required format
    /// 3. Performs automatic MSTL decomposition
    /// 4. Verifies that at least one period was detected
    #[test]
    fn test_auto_mstl() {
        let data = read_csv_to_vec("data/data.csv");
        let data_f32: Vec<[f32; 2]> = data.iter().map(|x| [x[0] as f32, x[1] as f32]).collect();
        let test_periods: Vec<usize> = Vec::new();
        let result = auto_mstl(&data_f32, &test_periods);
        assert!(result.periods.len() > 0);
    }
}
