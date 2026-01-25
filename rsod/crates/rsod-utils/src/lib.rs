use chrono::{TimeZone, Utc};
use csv::ReaderBuilder;
use std::fs::File;
use std::path::Path;

#[allow(dead_code)]
pub fn read_csv_to_vec<P: AsRef<Path>>(path: P) -> Vec<[f64; 2]> {
    let file = File::open(path).expect("Failed to open CSV file");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut result = Vec::new();

    for record in rdr.records() {
        let record = record.expect("Failed to read CSV record");
        let timestamp: f64 = record[0]
            .trim_matches('"')
            .parse()
            .expect("Failed to parse timestamp");
        let value: f64 = record[1].parse().expect("Failed to parse value");
        result.push([timestamp, value]);
    }

    result
}


/// Convert Unix timestamp to RFC3339 format time string
#[allow(dead_code)]
fn format_timestamp(timestamp: f64) -> String {
    // Convert millisecond timestamp to seconds
    let seconds = (timestamp / 1000.0) as i64;
    let dt = Utc.timestamp_opt(seconds, 0).unwrap();
    dt.to_rfc3339()
}

#[allow(dead_code)]
pub fn flatten_data(input: Vec<[f64; 2]>) -> Vec<f64> {
    let mut col0 = Vec::with_capacity(input.len());
    let mut col1 = Vec::with_capacity(input.len());
    for [a, b] in input {
        col0.push(a);
        col1.push(b);
    }
    col0.extend(col1);
    col0
}
/// Get prediction interval multiplier c based on confidence level (percentage)
/// 
/// Based on normal distribution assumption, returns z-score multiplier corresponding to the confidence level
/// 
/// Arguments:
/// - confidence_level: Confidence level (percentage), range 50.0 to 99.0
/// 
/// Returns:
/// - Corresponding multiplier c, used to calculate prediction interval: ŷ ± c * σ̂
#[allow(dead_code)]
pub fn get_confidence_multiplier(confidence_level: f64) -> f64 {
    // Return corresponding multiplier based on table in documentation
    // Use linear interpolation for intermediate values
    match confidence_level {
        x if x < 50.0 => 0.67, // Minimum value
        x if x >= 50.0 && x < 55.0 => 0.67 + (x - 50.0) / 5.0 * (0.76 - 0.67),
        x if x >= 55.0 && x < 60.0 => 0.76 + (x - 55.0) / 5.0 * (0.84 - 0.76),
        x if x >= 60.0 && x < 65.0 => 0.84 + (x - 60.0) / 5.0 * (0.93 - 0.84),
        x if x >= 65.0 && x < 70.0 => 0.93 + (x - 65.0) / 5.0 * (1.04 - 0.93),
        x if x >= 70.0 && x < 75.0 => 1.04 + (x - 70.0) / 5.0 * (1.15 - 1.04),
        x if x >= 75.0 && x < 80.0 => 1.15 + (x - 75.0) / 5.0 * (1.28 - 1.15),
        x if x >= 80.0 && x < 85.0 => 1.28 + (x - 80.0) / 5.0 * (1.44 - 1.28),
        x if x >= 85.0 && x < 90.0 => 1.44 + (x - 85.0) / 5.0 * (1.64 - 1.44),
        x if x >= 90.0 && x < 95.0 => 1.64 + (x - 90.0) / 5.0 * (1.96 - 1.64),
        x if x >= 95.0 && x < 96.0 => 1.96 + (x - 95.0) / 1.0 * (2.05 - 1.96),
        x if x >= 96.0 && x < 97.0 => 2.05 + (x - 96.0) / 1.0 * (2.17 - 2.05),
        x if x >= 97.0 && x < 98.0 => 2.17 + (x - 97.0) / 1.0 * (2.33 - 2.17),
        x if x >= 98.0 && x < 99.0 => 2.33 + (x - 98.0) / 1.0 * (2.58 - 2.33),
        x if x >= 99.0 => 2.58, // Maximum value
        _ => 1.96, // Default 95%
    }
}