//! The plugin service: query data + health check handlers.
//!
//! The service is stateless — all per-datasource configuration arrives with
//! each request via the plugin context (mirroring the former Go backend).

use std::convert::Infallible;

use futures_util::stream::FuturesOrdered;
use grafana_plugin_sdk::{backend, prelude::*};
use thiserror::Error;

use crate::client::GrafanaClient;
use crate::config::{PluginSettings, SecretPluginSettings};

#[derive(Clone, Debug, GrafanaPlugin)]
#[grafana_plugin(
    plugin_type = "datasource",
    json_data = "PluginSettings",
    secure_json_data = "SecretPluginSettings"
)]
pub struct PluginService;

impl PluginService {
    pub fn new() -> Self {
        Self
    }
}

/// An error that may occur during a query.
///
/// Stores the `ref_id` so Grafana can line it up with the failing query.
#[derive(Debug, Error)]
pub enum QueryError {
    #[error("missing datasource instance settings for query {ref_id}")]
    MissingInstanceSettings { ref_id: String },
    #[error("query {ref_id} failed: {message}")]
    Internal { ref_id: String, message: String },
}

impl backend::DataQueryError for QueryError {
    fn ref_id(self) -> String {
        match self {
            Self::MissingInstanceSettings { ref_id } | Self::Internal { ref_id, .. } => ref_id,
        }
    }

    fn status(&self) -> backend::DataQueryStatus {
        backend::DataQueryStatus::Internal
    }
}

#[backend::async_trait]
impl backend::DataService for PluginService {
    type Query = serde_json::Value;
    type QueryError = QueryError;
    type Stream = backend::BoxDataResponseStream<Self::QueryError>;

    async fn query_data(
        &self,
        request: backend::QueryDataRequest<Self::Query, Self>,
    ) -> Self::Stream {
        let instance_settings = request.plugin_context.instance_settings;
        Box::pin(
            request
                .queries
                .into_iter()
                .map(move |query: backend::DataQuery<Self::Query>| {
                    let instance_settings = instance_settings.clone();
                    let ref_id = query.ref_id.clone();
                    async move {
                        let settings = instance_settings.as_ref().ok_or_else(|| {
                            QueryError::MissingInstanceSettings {
                                ref_id: ref_id.clone(),
                            }
                        })?;
                        tracing::debug!(
                            ref_id = %ref_id,
                            url = %settings.json_data.url,
                            trial_mode = settings.json_data.trial_mode,
                            "query received datasource settings"
                        );
                        // `json_data.url` (Go: `json.Unmarshal(JSONData, ...)`) —
                        // NOT the SDK struct's top-level `url` (the datasource's
                        // legacy url field, which the ConfigEditor never sets).
                        let client = GrafanaClient::new(
                            settings.json_data.url.clone(),
                            settings.decrypted_secure_json_data.api_token.clone(),
                        );
                        crate::pipeline::process_query(
                            &client,
                            query,
                            &settings.json_data,
                            &settings.decrypted_secure_json_data,
                        )
                        .await
                        .map_err(|message| QueryError::Internal {
                            ref_id,
                            message: message.to_string(),
                        })
                    }
                })
                .collect::<FuturesOrdered<_>>(),
        )
    }
}

#[backend::async_trait]
impl backend::DiagnosticsService for PluginService {
    type CheckHealthError = Infallible;

    async fn check_health(
        &self,
        request: backend::CheckHealthRequest<Self>,
    ) -> Result<backend::CheckHealthResponse, Self::CheckHealthError> {
        let response = match request.plugin_context.instance_settings {
            Some(settings) => {
                crate::health::run(&settings.json_data, &settings.decrypted_secure_json_data).await
            }
            None => backend::CheckHealthResponse::error("Unable to load settings".to_string()),
        };
        Ok(response)
    }

    type CollectMetricsError = Infallible;

    async fn collect_metrics(
        &self,
        _request: backend::CollectMetricsRequest<Self>,
    ) -> Result<backend::CollectMetricsResponse, Self::CollectMetricsError> {
        Ok(backend::CollectMetricsResponse::new(None))
    }
}
