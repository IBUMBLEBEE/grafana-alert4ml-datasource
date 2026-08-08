//! Health check — a faithful port of the Go backend's `CheckHealth`.
//!
//! Order matters: API token → Grafana login ping → trial mode → PostgreSQL
//! fields → PostgreSQL ping.

use std::time::Duration;

use grafana_plugin_sdk::backend;
use postgres::NoTls;
use tokio::task::spawn_blocking;
use tracing::debug;

use crate::config::{PluginSettings, SecretPluginSettings};

pub async fn run(
    settings: &PluginSettings,
    secrets: &SecretPluginSettings,
) -> backend::CheckHealthResponse {
    if secrets.api_token.is_empty() {
        return backend::CheckHealthResponse::error("API Token is missing".to_string());
    }

    // Go parsed the URL first (`invalid URL: ...`); reqwest surfaces bad URLs
    // at request time with a clear message, so this check is folded into the ping.
    let client = crate::client::GrafanaClient::new(settings.url.clone(), secrets.api_token.clone());
    if let Err(e) = client.login_ping().await {
        return backend::CheckHealthResponse::error(e);
    }

    if settings.trial_mode {
        // The Go backend never touched storage; funnel persistence lazily
        // initializes SQLite in-memory here (matching the message Go showed).
        best_effort_storage_init(true, "");
        return backend::CheckHealthResponse::ok(
            "Data source is working (trial mode, using SQLite in-memory storage)".to_string(),
        );
    }

    if settings.pg_host.is_empty() || settings.pg_database.is_empty() || settings.pg_user.is_empty()
    {
        return backend::CheckHealthResponse::error(
            "PostgreSQL configuration is incomplete: host, database and user are required"
                .to_string(),
        );
    }

    let dsn = settings.pg_dsn(secrets);
    if let Err(e) = pg_ping(&dsn).await {
        return backend::CheckHealthResponse::error(format!("PostgreSQL: {}", e));
    }
    best_effort_storage_init(false, &dsn);

    backend::CheckHealthResponse::ok("Data source is working".to_string())
}

/// `CheckPgHealth`: open + ping with a 5s deadline (mirrors Go's
/// `db.PingContext` timeout). Uses the sync `postgres` crate (same version
/// as `rsod-storage`) on a blocking thread.
async fn pg_ping(dsn: &str) -> Result<(), String> {
    let dsn = dsn.to_string();
    let handle = spawn_blocking(move || {
        postgres::Client::connect(&dsn, NoTls)
            .map(|_| ())
            .map_err(|e| format!("failed to ping PostgreSQL: {}", e))
    });
    // tokio >= 1.53: awaiting a `JoinHandle` yields `Result<T, JoinError>`
    // (it no longer panics on task panic), so there are three levels.
    match tokio::time::timeout(Duration::from_secs(5), handle).await {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(join_err)) => Err(format!("blocking task failed: {}", join_err)),
        Err(_) => Err("failed to ping PostgreSQL: context deadline exceeded".to_string()),
    }
}

/// rsod-storage is a one-shot (OnceLock) global; initialize it best-effort so
/// funnel persistence uses the configured backend. Errors are logged, never
/// fatal — the Go backend never initialized storage at all.
fn best_effort_storage_init(trial_mode: bool, dsn: &str) {
    if let Err(e) = rsod_storage::init_db_with_config(trial_mode, dsn) {
        debug!(error = %e, "failed to initialize storage");
    }
}
