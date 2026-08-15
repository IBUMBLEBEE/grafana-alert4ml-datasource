//! Deterministic model-key derivation (UUID v5).
//!
//! The model key must stay byte-compatible with the former Go backend: the
//! SHA-1 hash runs over `json.Marshal` output. The Go-JSON encoding helpers
//! (`go_json_string`, `go_f32`, `derive_uuid`, …) have moved to `rsod-core`
//! so every engine crate can reuse them; only the plugin-specific
//! `UniqueKeysUuid` (a frontend↔backend wire contract) remains here.

use uuid::Uuid;

use rsod_core::go_json_string;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
