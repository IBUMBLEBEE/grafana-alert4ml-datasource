//! Grafana HTTP client for proxied queries and health checks.
//!
//! Port of the former Go backend (`pkg/sdk/client.go`): the plugin forwards
//! the query bodies to the datasource configured in the plugin settings via
//! Grafana's `/api/ds/query` (it proxies another Grafana datasource), and
//! pings `/api/login/ping` for health.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, SecondsFormat, Utc};
use grafana_plugin_sdk::data::Frame;
use serde::Serialize;
use serde_json::Value;

pub struct GrafanaClient {
    base_url: String,
    api_token: String,
    http: reqwest::Client,
}

/// The proxied query body: `{ queries, from, to }`. `intervalMs` is carried
/// per-target (injected by `build_targets_with_interval`), not at body level;
/// `interval_ms` mirrors the Go body field (never serialized, `json:"-"`).
#[derive(Serialize)]
pub struct ProxyQueryBody {
    pub queries: Vec<Value>,
    #[serde(serialize_with = "ser_time_rfc3339")]
    pub from: DateTime<Utc>,
    #[serde(serialize_with = "ser_time_rfc3339")]
    pub to: DateTime<Utc>,
    #[serde(skip)]
    pub interval_ms: i64,
}

fn ser_time_rfc3339<S: serde::Serializer>(t: &DateTime<Utc>, s: S) -> Result<S::Ok, S::Error> {
    // Go time.Time JSON marshal: RFC3339Nano, "Z" suffix for UTC.
    s.serialize_str(&t.to_rfc3339_opts(SecondsFormat::AutoSi, true))
}

/// `/api/ds/query` response: `{ results: { <refId>: { frames: [...] } } }`.
/// Upstream per-query errors arrive as `status`/`error` — the Go backend
/// ignored those, so this port only reads `frames`.
///
/// `Serialize` is required for history-cache persistence (Phase 3).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GrafanaQueryDataResponse {
    pub results: HashMap<String, GrafanaDataResponse>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GrafanaDataResponse {
    pub frames: Vec<Frame>,
}

impl GrafanaClient {
    pub fn new(base_url: String, api_token: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");
        Self {
            base_url,
            api_token,
            http,
        }
    }

    /// Fingerprint of datasource URL + token for cache isolation (not for logging).
    pub fn cache_scope(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.base_url.hash(&mut hasher);
        self.api_token.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Build and parse the endpoint URL up front so misconfigured base URLs
    /// surface as a clear `invalid URL '...': <reason>` (Go reported the same
    /// via `url.Parse` / `http.NewRequest`) instead of reqwest's opaque
    /// "builder error".
    fn url(&self, path: &str) -> rsod_core::Result<reqwest::Url> {
        let full = format!("{}/{}", self.base_url.trim_end_matches('/'), path);
        reqwest::Url::parse(&full).map_err(|e| format!("invalid URL '{}': {}", full, e).into())
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        if !self.api_token.is_empty() {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.api_token))
                    .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static("Bearer")),
            );
        }
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        headers
    }

    /// POST /api/ds/query — proxy the query bodies to the configured datasource.
    pub async fn data_source_query(
        &self,
        body: &ProxyQueryBody,
    ) -> rsod_core::Result<GrafanaQueryDataResponse> {
        let response = self
            .http
            .post(self.url("api/ds/query")?)
            .headers(self.headers())
            .json(body)
            .send()
            .await
            .map_err(|e| format!("request failed: {}", e))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(format!("received error response: {}", text).into());
        }
        response
            .json::<GrafanaQueryDataResponse>()
            .await
            .map_err(|e| format!("failed to decode /api/ds/query response: {}", e).into())
    }

    /// GET /api/login/ping — health check against the configured Grafana.
    pub async fn login_ping(&self) -> rsod_core::Result<()> {
        let response = self
            .http
            .get(self.url("api/login/ping")?)
            .headers(self.headers())
            .send()
            .await
            .map_err(|e| format!("login ping failed: {}", e))?;
        if !response.status().is_success() {
            return Err(format!(
                "login ping failed: received error response: {}",
                response.text().await.unwrap_or_default()
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_joins_base_and_path() {
        let c = GrafanaClient::new("http://grafana:3000/".to_string(), String::new());
        assert_eq!(
            c.url("api/ds/query").unwrap().as_str(),
            "http://grafana:3000/api/ds/query"
        );
    }

    #[test]
    fn url_requires_valid_base() {
        let c = GrafanaClient::new("192.168.59.132:3000".to_string(), String::new());
        let err = c.url("api/ds/query").unwrap_err();
        assert!(
            err.to_string().contains("invalid URL")
                && err.to_string().contains("192.168.59.132:3000"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn url_empty_base_fails_clearly() {
        let c = GrafanaClient::new(String::new(), String::new());
        let err = c.url("api/ds/query").unwrap_err();
        assert!(
            err.to_string().contains("invalid URL"),
            "unexpected error: {}",
            err
        );
    }

    /// Regression test for the vendored SDK fix (rsod/vendor): Grafana's
    /// FrameJSON omits `schema.name` entirely when the frame name is empty
    /// (prometheus/testdata upstreams) — the SDK 0.5.0 deserializer used to
    /// reject that with `missing field 'name'`, breaking every proxied query.
    #[test]
    fn decodes_frame_json_without_schema_name() {
        let json = r#"{
            "results": {
                "A": {
                    "frames": [{
                        "schema": {
                            "refId": "A",
                            "fields": [
                                {"name": "Time", "type": "time", "typeInfo": {"frame": "time.Time"}},
                                {"name": "up", "type": "number", "typeInfo": {"frame": "float64"}}
                            ]
                        },
                        "data": {"values": [[1786147200000], [1.0]]}
                    }]
                }
            }
        }"#;
        let rsp: GrafanaQueryDataResponse = serde_json::from_str(json).unwrap();
        let frames = &rsp.results["A"].frames;
        assert_eq!(frames.len(), 1);
        assert!(frames[0].name.is_empty());
        assert_eq!(frames[0].fields().len(), 2);
    }
}
