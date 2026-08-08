//! Plugin instance configuration.
//!
//! These types deserialize the datasource `jsonData` / `secureJsonData`
//! configured in the Grafana UI. Field order and `json` tags are part of the
//! frontend↔backend contract and must not be reordered.

use serde::Deserialize;

/// Non-secret settings from the datasource `jsonData`.
///
/// Field names must match the frontend's camelCase keys (`trialMode`,
/// `pgSSLMode`, ...) — Grafana stores and the SDK forwards them verbatim,
/// so serde renames are part of the contract, not a style choice.
#[derive(Clone, Debug, Deserialize)]
pub struct PluginSettings {
    #[serde(default)]
    pub url: String,
    #[serde(default, rename = "trialMode")]
    pub trial_mode: bool,
    #[serde(default, rename = "pgHost")]
    pub pg_host: String,
    #[serde(default, rename = "pgPort")]
    pub pg_port: i32,
    #[serde(default, rename = "pgDatabase")]
    pub pg_database: String,
    #[serde(default, rename = "pgUser")]
    pub pg_user: String,
    #[serde(default, rename = "pgSSLMode")]
    pub pg_ssl_mode: String,
}

/// Secret settings from the datasource `secureJsonData` (decrypted by Grafana).
#[derive(Clone, Debug, Deserialize)]
pub struct SecretPluginSettings {
    #[serde(default, rename = "apiToken")]
    pub api_token: String,
    #[serde(default, rename = "pgPassword")]
    pub pg_password: String,
}

impl PluginSettings {
    /// Build a PostgreSQL connection DSN, mirroring the Go `PgDSN()`.
    pub fn pg_dsn(&self, secrets: &SecretPluginSettings) -> String {
        let port = if self.pg_port == 0 {
            5432
        } else {
            self.pg_port
        };
        let ssl_mode = if self.pg_ssl_mode.is_empty() {
            "disable"
        } else {
            &self.pg_ssl_mode
        };
        format!(
            "host={} port={} user={} password={} dbname={} sslmode={}",
            self.pg_host, port, self.pg_user, secrets.pg_password, self.pg_database, ssl_mode
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Contract fixture: keys exactly as the frontend sends them
    /// (`src/types.ts` `Alert4MLDataSourceOptions` / `Alert4MLSecureJsonData`
    /// / `Alert4MLPgSecureJsonData`). Renames drift here would silently reset
    /// settings to defaults — the same bug the health check surfaced.
    #[test]
    fn deserializes_frontend_json_data_keys() {
        let json = r#"{
            "url": "http://grafana:3000",
            "trialMode": true,
            "pgHost": "pg.example.com",
            "pgPort": 5433,
            "pgDatabase": "alert4ml",
            "pgUser": "alice",
            "pgSSLMode": "require"
        }"#;
        let settings: PluginSettings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.url, "http://grafana:3000");
        assert!(settings.trial_mode);
        assert_eq!(settings.pg_host, "pg.example.com");
        assert_eq!(settings.pg_port, 5433);
        assert_eq!(settings.pg_database, "alert4ml");
        assert_eq!(settings.pg_user, "alice");
        assert_eq!(settings.pg_ssl_mode, "require");
    }

    #[test]
    fn deserializes_frontend_secure_json_data_keys() {
        let json = r#"{"apiToken": "glsa-token-123", "pgPassword": "s3cret"}"#;
        let secrets: SecretPluginSettings = serde_json::from_str(json).unwrap();
        assert_eq!(secrets.api_token, "glsa-token-123");
        assert_eq!(secrets.pg_password, "s3cret");
    }

    #[test]
    fn missing_keys_default_to_empty() {
        let settings: PluginSettings = serde_json::from_str(r#"{"url": "x"}"#).unwrap();
        assert!(!settings.trial_mode);
        assert_eq!(settings.pg_port, 0);
        assert!(settings.pg_host.is_empty());

        let secrets: SecretPluginSettings = serde_json::from_str("{}").unwrap();
        assert!(secrets.api_token.is_empty());
        assert!(secrets.pg_password.is_empty());
    }
}
