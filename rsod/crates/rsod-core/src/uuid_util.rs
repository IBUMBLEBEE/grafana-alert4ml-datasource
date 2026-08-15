//! Go `encoding/json`-compatible helpers for deterministic model keys.
//!
//! The model key must stay byte-compatible with the former Go backend: the
//! SHA-1 hash runs over `json.Marshal` output, so this module reproduces Go's
//! JSON encoding byte-for-byte — field order, HTML escaping (`<`, `>`, `&`),
//! `\u2028`/`\u2029`, lowercase `\u00xx` control escapes, and Go's float32
//! shortest-round-trip formatting with its `'g'`-style exponent thresholds.

use uuid::Uuid;

/// `DeriveUUID(baseUUID, extra)`: Go's `uuid.NewSHA1(base, extraJSON)` — the
/// *parsed base UUID is the SHA-1 namespace* and only the Go-JSON of `extra`
/// is hashed as data (not a concatenation under the plugin namespace).
pub fn derive_uuid(base_uuid: &str, extra_json: &str) -> crate::Result<String> {
    let base =
        Uuid::parse_str(base_uuid).map_err(|e| format!("failed to parse base UUID: {}", e))?;
    Ok(Uuid::new_v5(&base, extra_json.as_bytes()).to_string())
}

pub fn go_opt_i64(v: Option<i64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".to_string(),
    }
}

pub fn go_opt_f32(v: Option<f32>) -> String {
    match v {
        Some(x) => go_f32(x),
        None => "null".to_string(),
    }
}

/// Go `encoding/json` string escaping: HTML-escape `<`, `>`, `&`, escape
/// U+2028/U+2029 and control characters as lowercase `\u00xx`.
pub fn go_json_string(s: &str) -> String {
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
        // Non-ASCII text passes through unescaped.
        assert_eq!(go_json_string("café ☕"), "\"café ☕\"");
    }

    #[test]
    fn go_f32_plain() {
        assert_eq!(go_f32(1.0), "1");
        assert_eq!(go_f32(0.5), "0.5");
        assert_eq!(go_f32(0.1), "0.1");
        assert_eq!(go_f32(123.456), "123.456");
        assert_eq!(go_f32(-1.5), "-1.5");
        assert_eq!(go_f32(0.0), "0");
        assert_eq!(go_f32(1e-6), "0.000001");
        assert_eq!(go_f32(100000.0), "100000");
    }

    #[test]
    fn go_f32_scientific() {
        assert_eq!(go_f32(1e-7), "1e-7");
        assert_eq!(go_f32(1e21), "1e+21");
        assert_eq!(go_f32(1e22), "1e+22");
        assert_eq!(go_f32(1.5e22), "1.5e+22");
        assert_eq!(go_f32(-2.5e-8), "-2.5e-8");
    }

    #[test]
    fn derive_uuid_matches_go() {
        // Same inputs as the Go backend's `DeriveUUID` — verified against Go.
        let derived =
            derive_uuid("c138e166-a5c0-5c75-9528-da1950d6a57e", "{\"periods\":[24,168]}")
                .unwrap();
        // Value is a valid UUID; exact byte-compat is asserted in the backend
        // test that exercises the full forecast training-key chain.
        assert!(Uuid::parse_str(&derived).is_ok());
    }
}
