use num_complex::Complex64;
use rustfft::FftPlanner;
use std::collections::HashMap;
use std::f64;

pub struct PeriodDetector {
    pub window_size: usize,
    pub min_peak_threshold: f64,
}

impl PeriodDetector {
    #[allow(dead_code)]
    pub fn detect_period(&self, data: &[f64]) -> usize {
        let ma = self.moving_average(data);
        if ma.is_empty() {
            return 0;
        }
        let start_idx = self.window_size - 1;
        let residuals: Vec<f64> = ma
            .iter()
            .enumerate()
            .map(|(i, &m)| data[start_idx + i] - m)
            .collect();

        let ac = self.circular_autocorrelation(&residuals);
        let peaks = self.find_significant_peaks(&ac);
        let period = self.determine_period(&peaks);
        if data.len() < period * 2 {
            return 0;
        }
        period
    }

    fn moving_average(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.window_size || self.window_size == 0 {
            return vec![];
        }
        let mut ma = Vec::with_capacity(n - self.window_size + 1);
        let mut sum: f64 = data[..self.window_size].iter().sum();
        ma.push(sum / self.window_size as f64);
        for i in 1..=n - self.window_size {
            sum += data[i + self.window_size - 1] - data[i - 1];
            ma.push(sum / self.window_size as f64);
        }
        ma
    }

    fn circular_autocorrelation(&self, residuals: &[f64]) -> Vec<f64> {
        let n = residuals.len();
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex
        let mut buffer: Vec<Complex64> =
            residuals.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        fft.process(&mut buffer);

        // X[i] *= conj(X[i])
        for x in &mut buffer {
            *x = *x * x.conj();
        }

        ifft.process(&mut buffer);

        // Normalize
        buffer.iter().map(|c| c.re / n as f64).collect()
    }

    fn find_significant_peaks(&self, ac: &[f64]) -> Vec<usize> {
        let mut peaks = Vec::new();
        let mut max_ac = f64::NEG_INFINITY;
        for &v in &ac[1..] {
            if v > max_ac {
                max_ac = v;
            }
        }
        let threshold = max_ac * self.min_peak_threshold;
        for t in 1..ac.len() - 1 {
            if ac[t] > ac[t - 1] && ac[t] > ac[t + 1] && ac[t] >= threshold {
                peaks.push(t);
            }
        }
        peaks
    }

    #[allow(dead_code)]
    fn determine_period(&self, peaks: &[usize]) -> usize {
        if peaks.len() < 2 {
            return 0;
        }
        let mut interval_counts = HashMap::new();
        for i in 1..peaks.len() {
            let interval = peaks[i] - peaks[i - 1];
            *interval_counts.entry(interval).or_insert(0) += 1;
        }
        let mut best_interval = 0;
        let mut max_count = 0;
        for (&interval, &count) in &interval_counts {
            if count > max_count || (count == max_count && interval < best_interval) {
                max_count = count;
                best_interval = interval;
            }
        }
        best_interval
    }

    #[allow(dead_code)]
    pub fn detect_periods(
        &self,
        data: &[f64],
        top_n: usize,
        min_period: usize,
        max_period: usize,
    ) -> Vec<usize> {
        let ma = self.moving_average(data);
        if ma.is_empty() {
            return vec![];
        }
        let start_idx = self.window_size - 1;
        let residuals: Vec<f64> = ma
            .iter()
            .enumerate()
            .map(|(i, &m)| data[start_idx + i] - m)
            .collect();
        let ac = self.circular_autocorrelation(&residuals);
        let peaks = self.find_significant_peaks(&ac);
        self.determine_periods(&peaks, top_n, min_period, max_period)
    }

    #[allow(dead_code)]
    pub fn determine_periods(
        &self,
        peaks: &[usize],
        top_n: usize,
        min_period: usize,
        max_period: usize,
    ) -> Vec<usize> {
        if peaks.len() < 2 {
            return vec![];
        }
        let mut interval_counts = HashMap::new();
        for i in 1..peaks.len() {
            let interval = peaks[i] - peaks[i - 1];
            if interval >= min_period && interval <= max_period {
                *interval_counts.entry(interval).or_insert(0) += 1;
            }
        }
        let mut sorted: Vec<(usize, i32)> = interval_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted
            .into_iter()
            .take(top_n)
            .map(|(interval, _)| interval)
            .collect()
    }

    #[allow(dead_code)]
    pub fn detect_periods_all_pairs(
        &self,
        data: &[f64],
        top_n: usize,
        min_period: usize,
        max_period: usize,
    ) -> Vec<usize> {
        let ma = self.moving_average(data);
        if ma.is_empty() {
            return vec![];
        }
        let start_idx = self.window_size - 1;
        let residuals: Vec<f64> = ma
            .iter()
            .enumerate()
            .map(|(i, &m)| data[start_idx + i] - m)
            .collect();
        let ac = self.circular_autocorrelation(&residuals);
        let peaks = self.find_significant_peaks(&ac);
        self.determine_periods_all_pairs(&peaks, top_n, min_period, max_period)
    }

    #[allow(dead_code)]
    pub fn determine_periods_all_pairs(
        &self,
        peaks: &[usize],
        top_n: usize,
        min_period: usize,
        max_period: usize,
    ) -> Vec<usize> {
        if peaks.len() < 2 {
            return vec![];
        }
        let mut interval_counts = HashMap::new();
        for i in 0..peaks.len() {
            for j in i + 1..peaks.len() {
                let interval = peaks[j] - peaks[i];
                if interval >= min_period && interval <= max_period {
                    *interval_counts.entry(interval).or_insert(0) += 1;
                }
            }
        }
        let mut sorted: Vec<(usize, i32)> = interval_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        sorted
            .into_iter()
            .take(top_n)
            .map(|(interval, _)| interval)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsod_utils::read_csv_to_vec;

    #[test]
    fn test_detect_periods() {
        let detector = PeriodDetector {
            window_size: 6,
            min_peak_threshold: 0.5,
        };

        // read data from csv
        let data = read_csv_to_vec("data/data.csv");
        // plot_time_series(&data, "data/stl_residual_residual5.png");
        let time_series: Vec<f64> = data.iter().map(|x| x[1]).collect();
        let periods = detector.detect_periods(&time_series, 3, 12, 24 * 30);
        assert!(periods.contains(&24));
        // assert!(periods.contains(&168)); // 7 * 24 = 168
    }

    #[test]
    fn test_detect_period() {
        let detector = PeriodDetector {
            window_size: 6,
            min_peak_threshold: 0.5,
        };

        // read data from csv
        let data = read_csv_to_vec("data/data.csv");
        // plot_time_series(&data, "data/stl_residual_residual5.png");
        let time_series: Vec<f64> = data.iter().map(|x| x[1]).collect();
        let periods = detector.detect_period(&time_series);
        assert!(periods == 24);
        // assert!(periods.contains(&168)); // 7 * 24 = 168
    }
}
