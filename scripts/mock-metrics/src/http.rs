//! HTTP routes for the mock metrics API.

use crate::scenario::{SCENARIOS, SCENARIO_META};
use crate::series::{build_series, now_ms, parse_time_ms};
use axum::extract::Query;
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::{Any, CorsLayer};

#[derive(Debug, Deserialize)]
pub struct SeriesQuery {
    #[serde(default = "default_scenario")]
    pub scenario: String,
    pub from: Option<String>,
    pub to: Option<String>,
    #[serde(default = "default_step")]
    pub step: String,
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_scenario() -> String {
    "weekly".to_string()
}

fn default_step() -> String {
    "60000".to_string()
}

fn default_format() -> String {
    "object".to_string()
}

pub fn app() -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/api/scenarios", get(scenarios))
        .route("/api/series", get(series))
        .fallback(not_found)
        .layer(cors)
}

async fn health() -> Json<Value> {
    Json(json!({
        "ok": true,
        "service": "alert4ml-mock-metrics",
        "scenarios": SCENARIOS,
    }))
}

async fn scenarios() -> Json<Value> {
    let list: Vec<Value> = SCENARIO_META
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "for": m.for_panel,
                "notes": m.notes,
            })
        })
        .collect();
    Json(json!({ "scenarios": list }))
}

async fn series(Query(q): Query<SeriesQuery>) -> Response {
    match series_payload(q) {
        Ok((status, body)) => (status, Json(body)).into_response(),
        Err(msg) => (StatusCode::BAD_REQUEST, Json(json!({ "error": msg }))).into_response(),
    }
}

fn series_payload(q: SeriesQuery) -> Result<(StatusCode, Value), String> {
    let now = now_ms();
    let to_ms = parse_time_ms(q.to.as_deref(), now)?;
    let from_ms = parse_time_ms(q.from.as_deref(), to_ms - 3_600_000)?;
    let step_ms: i64 = q
        .step
        .parse()
        .map_err(|_| format!("invalid step: {}", q.step))?;

    let data = build_series(&q.scenario, from_ms, to_ms, step_ms)?;

    let fmt = q.format.to_ascii_lowercase();
    if matches!(fmt.as_str(), "array" | "rows" | "raw") {
        let body = serde_json::to_value(&data).map_err(|e| e.to_string())?;
        return Ok((StatusCode::OK, body));
    }

    Ok((
        StatusCode::OK,
        json!({
            "scenario": q.scenario,
            "from": from_ms,
            "to": to_ms,
            "stepMs": step_ms,
            "count": data.len(),
            "data": data,
        }),
    ))
}

async fn not_found(uri: axum::http::Uri) -> (StatusCode, Json<Value>) {
    let path = uri.path().trim_end_matches('/');
    let path = if path.is_empty() { "/" } else { path };
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": format!("not found: {path}") })),
    )
}
