//! Duration parsing and period conversion, shared by engine crates.
//!
//! `parse_duration_ms` is a faithful reimplementation of
//! `github.com/xhit/go-str2duration/v2` (units: ns/us/µs/μs/ms/s/m/h/d/w;
//! compound and fractional values allowed).

/// Parse a Go-str2duration string into milliseconds.
/// Units: ns, us, µs, μs, ms, s, m, h, d, w. Supports compounds ("2h45m"),
/// fractions ("1.5h") and a leading sign.
pub fn parse_duration_ms(s: &str) -> crate::Result<i64> {
    let orig = s;
    let mut s = s;
    let mut neg = false;
    if let Some(rest) = s.strip_prefix(['-', '+']) {
        neg = s.starts_with('-');
        s = rest;
    }
    if s == "0" {
        return Ok(0);
    }
    if s.is_empty() {
        return Err(format!("time: invalid duration \"{}\"", orig).into());
    }
    let mut total_ns: i64 = 0;
    while !s.is_empty() {
        // Walk the string by (byte offset, char) pairs so the remainder can
        // be re-sliced from the original `s` (never borrows a local).
        let chars: Vec<(usize, char)> = s.char_indices().collect();
        let mut idx = 0usize;
        let mut digits = String::new();
        let mut frac: Option<(u64, f64)> = None;
        let mut saw_digit = false;
        while idx < chars.len() {
            let c = chars[idx].1;
            if c.is_ascii_digit() {
                digits.push(c);
                saw_digit = true;
                idx += 1;
            } else if c == '.' {
                idx += 1;
                let mut f = String::new();
                while idx < chars.len() && chars[idx].1.is_ascii_digit() {
                    f.push(chars[idx].1);
                    idx += 1;
                }
                if !f.is_empty() {
                    let scale = 10f64.powi(f.len() as i32);
                    frac = Some((f.parse().unwrap_or(0), scale));
                }
                break;
            } else {
                break;
            }
        }
        if !saw_digit && frac.is_none() {
            return Err(format!("time: invalid duration \"{}\"", orig).into());
        }
        // Consume the unit (letters until the next digit/dot).
        let unit_start = idx;
        while idx < chars.len() && !chars[idx].1.is_ascii_digit() && chars[idx].1 != '.' {
            idx += 1;
        }
        let unit = &s[unit_start..chars.get(idx).map(|(i, _)| *i).unwrap_or(s.len())];
        s = &s[chars.get(idx).map(|(i, _)| *i).unwrap_or(s.len())..];
        if unit.is_empty() {
            return Err(format!("time: missing unit in duration \"{}\"", orig).into());
        }
        let unit_ns = match unit {
            "ns" => 1i64,
            "us" | "µs" | "μs" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60 * 1_000_000_000,
            "h" => 3600 * 1_000_000_000,
            "d" => 24 * 3600 * 1_000_000_000,
            "w" => 168 * 3600 * 1_000_000_000,
            _ => {
                return Err(format!(
                    "time: unknown unit \"{}\" in duration \"{}\"",
                    unit, orig
                )
                .into())
            }
        };
        let v: i64 = digits
            .parse()
            .map_err(|_| format!("time: invalid duration \"{}\"", orig))?;
        let mut ns = v
            .checked_mul(unit_ns)
            .ok_or_else(|| format!("time: invalid duration \"{}\"", orig))?;
        if let Some((f, scale)) = frac {
            // Mirrors Go: v += int64(float64(f) * (float64(unit)/scale))
            let extra = ((f as f64) * (unit_ns as f64) / scale) as i64;
            ns = ns
                .checked_add(extra)
                .ok_or_else(|| format!("time: invalid duration \"{}\"", orig))?;
        }
        total_ns = total_ns
            .checked_add(ns)
            .ok_or_else(|| format!("time: invalid duration \"{}\"", orig))?;
    }
    if neg {
        total_ns = -total_ns;
    }
    Ok(total_ns / 1_000_000)
}

/// `ParsePeriods`: split on commas/spaces, bare integers become hours, then
/// convert each duration to a number of intervals (truncating division).
pub fn parse_periods(durations: &str, interval_ms: i64) -> crate::Result<Vec<u64>> {
    let mut periods = Vec::new();
    for raw in durations.split([',', ' ']) {
        let d = raw.trim();
        if d.is_empty() {
            continue;
        }
        // Bare integers become hours ("24" → "24h"). Go's ParseUint also
        // accepts a leading '+', so mirror that.
        let bare = d.strip_prefix('+').unwrap_or(d);
        let is_bare_int = !bare.is_empty() && bare.chars().all(|c| c.is_ascii_digit());
        let with_unit = if is_bare_int {
            format!("{}h", d)
        } else {
            d.to_string()
        };
        let ms = parse_duration_ms(&with_unit)?;
        if interval_ms <= 0 {
            return Err(format!("intervalMs must be > 0, got {}", interval_ms).into());
        }
        periods.push((ms / interval_ms) as u64);
    }
    Ok(periods)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration_ms("24h"), Ok(24 * 3600 * 1000));
        assert_eq!(parse_duration_ms("7d"), Ok(7 * 24 * 3600 * 1000));
        assert_eq!(parse_duration_ms("1w"), Ok(7 * 24 * 3600 * 1000));
        assert_eq!(parse_duration_ms("2h45m"), Ok((2 * 3600 + 45 * 60) * 1000));
        assert_eq!(parse_duration_ms("1.5h"), Ok(90 * 60 * 1000));
        assert_eq!(parse_duration_ms("30m"), Ok(30 * 60 * 1000));
        assert_eq!(parse_duration_ms("500ms"), Ok(500));
        assert_eq!(parse_duration_ms("0"), Ok(0));
        assert!(parse_duration_ms("1y").is_err());
        assert!(parse_duration_ms("").is_err());
        assert!(parse_duration_ms("abc").is_err());
    }

    #[test]
    fn periods() {
        assert_eq!(parse_periods("24h,7d", 3_600_000), Ok(vec![24, 168]));
        assert_eq!(parse_periods("24 7d", 3_600_000), Ok(vec![24, 168]));
        assert_eq!(parse_periods("24", 3_600_000), Ok(vec![24])); // bare int → hours
        assert_eq!(parse_periods("", 3_600_000), Ok(vec![]));
        assert!(parse_periods("24h", 0).is_err());
    }
}
