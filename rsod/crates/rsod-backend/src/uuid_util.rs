//! Deterministic model-key derivation (UUID v5).
//!
//! The model key must stay byte-compatible with the former Go backend: the
//! SHA-1 hash runs over `json.Marshal` output, so this module reproduces Go's
//! JSON encoding byte-for-byte — field order, HTML escaping (`<`, `>`, `&`),
//! ` `/` `, lowercase `\u00xx` control escapes, and Go's float32
//! shortest-round-trip formatting with its `'g'`-style exponent thresholds.

use uuid::Uuid;

/// Namespace UUID used by the Go backend (derived from
/// `uuid.NewSHA1(uuid.NameSpaceOID, []byte("alert4ml-plugin"))`).
pub const ALERT4ML_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

/// Mirror of the Go `UniqueKeysUUID` struct — field order is the JSON byte
/// contract.
pub struct UniqueKeysUuid<'a> {
    pub detect_type: &'a str,
    pub support_detect: &'a str,
    pub dashboard_uid: &'a str,
    pub panel_id: i64,
    pub series_ref_id: &'a str,
    pub series_name: &'a str,
}

impl UniqueKeysUuid<'_> {
    /// Deterministic model key, byte-identical to the Go `ToUUID()`.
    pub fn to_uuid(&self) -> String {
        let json = format!(
            "{{\"detectType\":{},\"supportDetect\":{},\"uniqueKeys\":{{\"dashboardUid\":{},\"panelId\":{},\"seriesRefId\":{}}},\"seriesName\":{}}}",
            go_json_string(self.detect_type),
            go_json_string(self.support_detect),
            go_json_string(self.dashboard_uid),
            self.panel_id,
            go_json_string(self.series_ref_id),
            go_json_string(self.series_name),
        );
        let uuid = Uuid::new_v5(&ALERT4ML_NAMESPACE, json.as_bytes());
        uuid.to_string()
    }
}

/// Mirror of the Go `ForecastTrainingKey` struct — field order and null
/// serialization of the optional fields are part of the byte contract.
pub struct ForecastTrainingKey<'a> {
    pub periods: &'a [u64],
    pub budget: f32,
    pub num_threads: usize,
    pub max_bin: u16,
    pub iteration_limit: Option<i64>,
    pub timeout: Option<f32>,
    pub stopping_rounds: Option<i64>,
    pub seed: Option<u64>,
}

impl ForecastTrainingKey<'_> {
    /// Go `json.Marshal`-equivalent JSON string (used as the `DeriveUUID`
    /// extra payload).
    pub fn to_go_json(&self) -> String {
        let periods: Vec<String> = self.periods.iter().map(|p| p.to_string()).collect();
        format!(
            "{{\"periods\":[{}],\"budget\":{},\"num_threads\":{},\"max_bin\":{},\"iteration_limit\":{},\"timeout\":{},\"stopping_rounds\":{},\"seed\":{}}}",
            periods.join(","),
            go_f32(self.budget),
            self.num_threads,
            self.max_bin,
            go_opt_i64(self.iteration_limit),
            go_opt_f32(self.timeout),
            go_opt_i64(self.stopping_rounds),
            match self.seed {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            },
        )
    }
}

/// `DeriveUUID(baseUUID, extra)`: Go's `uuid.NewSHA1(base, extraJSON)` — the
/// *parsed base UUID is the SHA-1 namespace* and only the Go-JSON of `extra`
/// is hashed as data (not a concatenation under the plugin namespace).
pub fn derive_uuid(base_uuid: &str, extra_json: &str) -> Result<String, String> {
    let base =
        Uuid::parse_str(base_uuid).map_err(|e| format!("failed to parse base UUID: {}", e))?;
    Ok(Uuid::new_v5(&base, extra_json.as_bytes()).to_string())
}

fn go_opt_i64(v: Option<i64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".to_string(),
    }
}

fn go_opt_f32(v: Option<f32>) -> String {
    match v {
        Some(x) => go_f32(x),
        None => "null".to_string(),
    }
}

/// Go `encoding/json` string escaping: HTML-escape `<`, `>`, `&`, escape
/// U+2028/U+2029 and control characters as lowercase `\u00xx`.
fn go_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Go `encoding/json` float32 formatting: shortest round-trip digits,
/// scientific notation only when `float32(abs) < 1e-6 || float32(abs) >= 1e21`
/// (the comparison runs on the widened f32 value, so the true threshold is
/// the nearest f32 to those decimal cutoffs), with Go's exponent cleanup:
/// positive exponents stay `+`-signed and zero-padded (`1e+21`), negative
/// exponents lose their leading zero (`1e-07` → `1e-7`).
///
/// Non-finite values are unreachable from the wire (serde_json rejects NaN /
/// Inf literals) and default to `"null"` — Go would error instead.
pub fn go_f32(x: f32) -> String {
    if !x.is_finite() {
        return "null".to_string(); // unreachable from wire JSON; Go errors here
    }
    let abs = x.abs();
    if abs != 0.0 && !(1e-6..1e21).contains(&abs) {
        go_e_format(x)
    } else {
        go_f_format(x)
    }
}

/// Shortest digits for a finite f32 (Rust's `{:e}` is ryu-based shortest).
fn shortest_digits(x: f32) -> (String, i32) {
    let s = format!("{:e}", x);
    match s.split_once('e') {
        Some((mantissa, exp)) => (mantissa.to_string(), exp.parse::<i32>().unwrap_or(0)),
        None => (s, 0),
    }
}

/// Go `'e'` presentation: `mantissa` + `e` + `+`-padded or `-`-unpadded exp.
fn go_e_format(x: f32) -> String {
    let (mantissa, exp) = shortest_digits(x);
    if exp < 0 {
        format!("{}e-{}", mantissa, -exp)
    } else {
        format!("{}e+{:02}", mantissa, exp)
    }
}

/// Go `'f'` presentation: plain decimal, no exponent.
fn go_f_format(x: f32) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    let (mantissa, exp) = shortest_digits(x);
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa
        .trim_start_matches('-')
        .chars()
        .filter(|c| *c != '.')
        .collect();
    let (int_len, _frac_len) = if mantissa.contains('.') {
        let (i, f) = mantissa.trim_start_matches('-').split_once('.').unwrap();
        (i.len(), f.len())
    } else {
        (mantissa.trim_start_matches('-').len(), 0)
    };
    let point_pos = int_len as i32 + exp; // decimal point position relative to digit start
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if point_pos <= 0 {
        out.push_str("0.");
        for _ in 0..(-point_pos) {
            out.push('0');
        }
        out.push_str(&digits);
    } else if (point_pos as usize) >= digits.len() {
        out.push_str(&digits);
        for _ in 0..(point_pos as usize - digits.len()) {
            out.push('0');
        }
    } else {
        out.push_str(&digits[..point_pos as usize]);
        out.push('.');
        out.push_str(&digits[point_pos as usize..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_json_string_escapes() {
        assert_eq!(go_json_string("a<b>&c"), "\"a\\u003cb\\u003e\\u0026c\"");
        assert_eq!(go_json_string("quote\"slash\\"), "\"quote\\\"slash\\\\\"");
        assert_eq!(
            go_json_string("\u{0001}\u{2028}\u{2029}"),
            "\"\\u0001\\u2028\\u2029\""
        );
        // Non-ASCII text (accented Latin + supplementary-plane emoji) passes through unescaped.
        assert_eq!(go_json_string("café ☕"), "\"café ☕\"");
    }

    #[test]
    fn go_f32_plain() {
        // Expected values verified against `encoding/json` on Go 1.25.
        assert_eq!(go_f32(1.0), "1");
        assert_eq!(go_f32(0.5), "0.5");
        assert_eq!(go_f32(0.1), "0.1");
        assert_eq!(go_f32(123.456), "123.456");
        assert_eq!(go_f32(-1.5), "-1.5");
        assert_eq!(go_f32(0.0), "0");
        // The f32 nearest to 1e-6 is ≥ 1e-6 → plain (Go prints 0.000001).
        assert_eq!(go_f32(1e-6), "0.000001");
        assert_eq!(go_f32(100000.0), "100000");
    }

    #[test]
    fn go_f32_scientific() {
        // Negative exponents unpadded, positive zero-padded with '+'.
        assert_eq!(go_f32(1e-7), "1e-7");
        assert_eq!(go_f32(1e21), "1e+21");
        assert_eq!(go_f32(1e22), "1e+22");
        assert_eq!(go_f32(1.5e22), "1.5e+22");
        assert_eq!(go_f32(-2.5e-8), "-2.5e-8");
    }

    #[test]
    fn training_key_json_matches_go() {
        let key = ForecastTrainingKey {
            periods: &[24, 168],
            budget: 1.0,
            num_threads: 8,
            max_bin: 255,
            iteration_limit: None,
            timeout: None,
            stopping_rounds: None,
            seed: None,
        };
        assert_eq!(
            key.to_go_json(),
            "{\"periods\":[24,168],\"budget\":1,\"num_threads\":8,\"max_bin\":255,\"iteration_limit\":null,\"timeout\":null,\"stopping_rounds\":null,\"seed\":null}"
        );
    }

    #[test]
    fn unique_keys_uuid_matches_go() {
        // Expected value verified against the Go backend's
        // `uuid.NewSHA1(Alert4MLNamespace, json.Marshal(uk))` on Go 1.25.
        let uk = UniqueKeysUuid {
            detect_type: "forecast",
            support_detect: "",
            dashboard_uid: "abc123",
            panel_id: 4,
            series_ref_id: "A",
            series_name: "cpu",
        };
        assert_eq!(uk.to_uuid(), "c138e166-a5c0-5c75-9528-da1950d6a57e");
    }

    #[test]
    fn derive_uuid_matches_go() {
        // Same inputs as `unique_keys_uuid_matches_go`, then DeriveUUID with
        // the training-key JSON from the earlier test — verified against Go.
        let uk = UniqueKeysUuid {
            detect_type: "forecast",
            support_detect: "",
            dashboard_uid: "abc123",
            panel_id: 4,
            series_ref_id: "A",
            series_name: "cpu",
        };
        let key = ForecastTrainingKey {
            periods: &[24, 168],
            budget: 1.0,
            num_threads: 8,
            max_bin: 255,
            iteration_limit: None,
            timeout: None,
            stopping_rounds: None,
            seed: None,
        };
        let derived = derive_uuid(&uk.to_uuid(), &key.to_go_json()).unwrap();
        assert_eq!(derived, "ec56c101-b493-5759-9b02-6defc9f63fed");
    }
}
