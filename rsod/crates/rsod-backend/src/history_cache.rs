//! In-process LRU+TTL cache for upstream datasource query responses.
//!
//! Phase 1: exact / subset hits.
//! Phase 2: sliding-window reuse — keep overlapping history and fetch only the
//! missing left and/or right gap, then merge frames.
//! Phase 3: optional durability via `rsod-storage` **Postgres only**.
//! Trial (in-memory SQLite) skips persist — it cannot survive restart and would
//! only add sync serialize/DB cost on every store.
//! When durable: memory miss may hydrate from storage; writes are deferred on a
//! blocking pool so the query path is not stalled.
//!
//! Caches **raw upstream frames**, not ML models.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use rsod_core::ModelStorage;
use rsod_storage::GlobalStorage;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::client::{GrafanaClient, GrafanaDataResponse, GrafanaQueryDataResponse, ProxyQueryBody};
use crate::tools::{body_for_time_range, clone_frame, concat_frames, split_frame_by_time};

const DEFAULT_CAPACITY: usize = 64;
const DEFAULT_TTL: Duration = Duration::from_secs(120);
const PERSIST_PREFIX: &str = "history_cache:";
const PERSIST_VERSION: u32 = 1;

struct CacheEntry {
    stored_at: Instant,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    response: GrafanaQueryDataResponse,
}

/// Exact-match + sliding-window upstream response cache (LRU + TTL).
pub struct HistoryCache {
    capacity: usize,
    ttl: Duration,
    map: HashMap<u64, CacheEntry>,
    order: VecDeque<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlidePlan {
    /// Cached window fully covers the request — trim only.
    Cover,
    /// Request extends past cached `to`; reuse overlap and fetch the right gap.
    ExtendRight,
    /// Request extends before cached `from`; reuse overlap and fetch the left gap.
    ExtendLeft,
    /// Request extends past both edges; reuse the cached middle and fetch both gaps.
    ExtendBoth,
    /// No useful overlap — full fetch.
    Miss,
}

impl HistoryCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity: capacity.max(1),
            ttl,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn touch(&mut self, key: u64) {
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }

    fn get_entry(&mut self, key: u64, now: Instant) -> Option<&CacheEntry> {
        let expired = match self.map.get(&key) {
            Some(entry) => now.duration_since(entry.stored_at) > self.ttl,
            None => return None,
        };
        if expired {
            self.remove(key);
            return None;
        }
        self.touch(key);
        self.map.get(&key)
    }

    fn put(
        &mut self,
        key: u64,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        response: GrafanaQueryDataResponse,
        now: Instant,
    ) {
        if self.map.contains_key(&key) {
            if let Some(pos) = self.order.iter().position(|k| *k == key) {
                self.order.remove(pos);
            }
        } else {
            while self.map.len() >= self.capacity {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                } else {
                    break;
                }
            }
        }
        self.order.push_back(key);
        self.map.insert(
            key,
            CacheEntry {
                stored_at: now,
                from,
                to,
                response,
            },
        );
    }

    fn remove(&mut self, key: u64) {
        self.map.remove(&key);
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

type LookupHit = (
    SlidePlan,
    DateTime<Utc>,
    DateTime<Utc>,
    rsod_core::Result<GrafanaQueryDataResponse>,
);

fn global() -> &'static Mutex<HistoryCache> {
    static CACHE: OnceLock<Mutex<HistoryCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HistoryCache::new(DEFAULT_CAPACITY, DEFAULT_TTL)))
}

/// Durable blob written through `rsod-storage` (models table).
#[derive(Serialize, Deserialize)]
struct PersistBlob {
    version: u32,
    from_ms: i64,
    to_ms: i64,
    expires_at_ms: i64,
    response: GrafanaQueryDataResponse,
}

fn persist_storage_key(key: u64) -> String {
    format!("{PERSIST_PREFIX}{key:016x}")
}

fn wall_now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Durable history persist is only useful with Postgres (cross-restart).
fn durable_persist_enabled() -> bool {
    !rsod_storage::is_trial_mode()
}

fn persist_save(
    key: u64,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    response: &GrafanaQueryDataResponse,
    ttl: Duration,
) {
    let blob = PersistBlob {
        version: PERSIST_VERSION,
        from_ms: from.timestamp_millis(),
        to_ms: to.timestamp_millis(),
        expires_at_ms: wall_now_ms() + ttl.as_millis() as i64,
        response: match clone_query_response(response) {
            Ok(r) => r,
            Err(e) => {
                debug!(error = %e, "history cache persist: clone failed; skip");
                return;
            }
        },
    };
    let bytes = match serde_json::to_vec(&blob) {
        Ok(b) => b,
        Err(e) => {
            debug!(error = %e, "history cache persist: serialize failed; skip");
            return;
        }
    };
    if let Err(e) = GlobalStorage::new().save(&persist_storage_key(key), &bytes) {
        // Soft-fail: trial DB may be unavailable; memory cache still works.
        warn!(error = %e, key, "history cache persist write failed; continuing memory-only");
    }
}

/// Clone + serialize + DB write off the async query path (fire-and-forget).
fn persist_save_bg(
    key: u64,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    response: &GrafanaQueryDataResponse,
    ttl: Duration,
) {
    if !durable_persist_enabled() {
        return;
    }
    let Ok(response) = clone_query_response(response) else {
        return;
    };
    let storage_key = persist_storage_key(key);
    let from_ms = from.timestamp_millis();
    let to_ms = to.timestamp_millis();
    let expires_at_ms = wall_now_ms() + ttl.as_millis() as i64;

    let write = move || {
        let blob = PersistBlob {
            version: PERSIST_VERSION,
            from_ms,
            to_ms,
            expires_at_ms,
            response,
        };
        let bytes = match serde_json::to_vec(&blob) {
            Ok(b) => b,
            Err(e) => {
                debug!(error = %e, "history cache persist: serialize failed; skip");
                return;
            }
        };
        if let Err(e) = GlobalStorage::new().save(&storage_key, &bytes) {
            warn!(
                error = %e,
                key,
                "history cache persist write failed; continuing memory-only"
            );
        }
    };

    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn_blocking(write);
        }
        Err(_) => {
            // Unit tests / no runtime: keep sync path for direct persist helpers.
            write();
        }
    }
}

/// Load a non-expired persisted entry. Returns `None` on miss / expire / error.
fn persist_load(key: u64) -> Option<(DateTime<Utc>, DateTime<Utc>, GrafanaQueryDataResponse, i64)> {
    let bytes = match GlobalStorage::new().load(&persist_storage_key(key)) {
        Ok(Some(b)) if !b.is_empty() => b,
        Ok(_) => return None,
        Err(e) => {
            debug!(error = %e, key, "history cache persist read failed");
            return None;
        }
    };
    let blob: PersistBlob = match serde_json::from_slice(&bytes) {
        Ok(b) => b,
        Err(e) => {
            debug!(error = %e, key, "history cache persist decode failed");
            return None;
        }
    };
    if blob.version != PERSIST_VERSION {
        debug!(key, version = blob.version, "history cache persist version mismatch");
        return None;
    }
    let now_ms = wall_now_ms();
    if blob.expires_at_ms <= now_ms {
        let _ = GlobalStorage::new().delete(&persist_storage_key(key));
        return None;
    }
    let from = DateTime::from_timestamp_millis(blob.from_ms)?;
    let to = DateTime::from_timestamp_millis(blob.to_ms)?;
    Some((from, to, blob.response, blob.expires_at_ms))
}

/// Memory miss → try durable store; on hit, hydrate LRU with remaining TTL.
async fn try_hydrate_from_persist(
    key: u64,
    req_from: DateTime<Utc>,
    req_to: DateTime<Utc>,
) -> Option<LookupHit> {
    if !durable_persist_enabled() {
        return None;
    }
    let loaded = match tokio::task::spawn_blocking(move || persist_load(key)).await {
        Ok(v) => v?,
        Err(e) => {
            debug!(error = %e, key, "history cache persist hydrate join failed");
            return None;
        }
    };
    hydrate_loaded_entry(key, loaded, req_from, req_to)
}

fn hydrate_loaded_entry(
    key: u64,
    loaded: (DateTime<Utc>, DateTime<Utc>, GrafanaQueryDataResponse, i64),
    req_from: DateTime<Utc>,
    req_to: DateTime<Utc>,
) -> Option<LookupHit> {
    let (from, to, response, expires_at_ms) = loaded;
    let remaining_ms = (expires_at_ms - wall_now_ms()).max(0) as u64;
    let remaining = Duration::from_millis(remaining_ms);
    let ttl = {
        match global().lock() {
            Ok(cache) => cache.ttl,
            Err(_) => DEFAULT_TTL,
        }
    };
    // Back-date `stored_at` so in-memory TTL matches remaining wall-clock life.
    let stored_at = Instant::now()
        .checked_sub(ttl.saturating_sub(remaining))
        .unwrap_or_else(Instant::now);

    info!(key, ?from, ?to, "upstream history cache persist hit; hydrating memory");
    let response_for_plan = clone_query_response(&response).ok()?;
    if let Ok(mut cache) = global().lock() {
        cache.put(key, from, to, response, stored_at);
    }

    let plan = slide_plan(from, to, req_from, req_to);
    Some((plan, from, to, Ok(response_for_plan)))
}

/// Series fingerprint: datasource scope + interval + queries with time params
/// normalized so a sliding window still hits the same entry.
pub fn series_key(scope: &str, body: &ProxyQueryBody) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    scope.hash(&mut hasher);
    body.interval_ms.hash(&mut hasher);
    let normalized = normalize_queries_for_key(&body.queries);
    match serde_json::to_string(&normalized) {
        Ok(s) => s.hash(&mut hasher),
        Err(_) => "queries-serialize-error".hash(&mut hasher),
    }
    hasher.finish()
}

fn normalize_queries_for_key(queries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    use crate::tools::rewrite_embedded_time_range;
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let Some(mut obj) = q.as_object().cloned() else {
            out.push(q.clone());
            continue;
        };
        // Pin embedded times so slides don't change the series fingerprint.
        rewrite_embedded_time_range(&mut obj, 0, 0);
        obj.remove("maxDataPoints");
        out.push(serde_json::Value::Object(obj));
    }
    out
}

fn slide_plan(
    cached_from: DateTime<Utc>,
    cached_to: DateTime<Utc>,
    req_from: DateTime<Utc>,
    req_to: DateTime<Utc>,
) -> SlidePlan {
    if req_to <= req_from || cached_to <= cached_from {
        return SlidePlan::Miss;
    }
    // No temporal overlap.
    if req_to <= cached_from || req_from >= cached_to {
        return SlidePlan::Miss;
    }
    // Full cover (exact or subset).
    if req_from >= cached_from && req_to <= cached_to {
        return SlidePlan::Cover;
    }
    let extends_left = req_from < cached_from;
    let extends_right = req_to > cached_to;
    match (extends_left, extends_right) {
        (false, true) => SlidePlan::ExtendRight,
        (true, false) => SlidePlan::ExtendLeft,
        (true, true) => SlidePlan::ExtendBoth,
        (false, false) => SlidePlan::Cover,
    }
}

fn clone_query_response(
    rsp: &GrafanaQueryDataResponse,
) -> rsod_core::Result<GrafanaQueryDataResponse> {
    let mut results = HashMap::with_capacity(rsp.results.len());
    for (ref_id, data) in &rsp.results {
        let frames = data
            .frames
            .iter()
            .map(clone_frame)
            .collect::<Result<Vec<_>, _>>()?;
        results.insert(ref_id.clone(), GrafanaDataResponse { frames });
    }
    Ok(GrafanaQueryDataResponse { results })
}

fn trim_response(
    rsp: &GrafanaQueryDataResponse,
    from: DateTime<Utc>,
    to: DateTime<Utc>,
) -> rsod_core::Result<GrafanaQueryDataResponse> {
    let mut results = HashMap::with_capacity(rsp.results.len());
    for (ref_id, data) in &rsp.results {
        let mut frames = Vec::with_capacity(data.frames.len());
        for frame in &data.frames {
            if frame.fields().is_empty() {
                frames.push(clone_frame(frame)?);
            } else {
                frames.push(split_frame_by_time(frame, from, to)?);
            }
        }
        results.insert(ref_id.clone(), GrafanaDataResponse { frames });
    }
    Ok(GrafanaQueryDataResponse { results })
}

fn merge_responses(
    left: &GrafanaQueryDataResponse,
    right: &GrafanaQueryDataResponse,
) -> rsod_core::Result<GrafanaQueryDataResponse> {
    let mut results = HashMap::with_capacity(left.results.len());
    for (ref_id, left_data) in &left.results {
        let right_frames = right
            .results
            .get(ref_id)
            .map(|d| d.frames.as_slice())
            .unwrap_or(&[]);
        if right_frames.is_empty() {
            results.insert(
                ref_id.clone(),
                GrafanaDataResponse {
                    frames: left_data
                        .frames
                        .iter()
                        .map(clone_frame)
                        .collect::<Result<Vec<_>, _>>()?,
                },
            );
            continue;
        }
        if left_data.frames.len() != right_frames.len() {
            return Err(format!(
                "history cache merge: frame count mismatch for refId {ref_id}: {} vs {}",
                left_data.frames.len(),
                right_frames.len()
            )
            .into());
        }
        let mut frames = Vec::with_capacity(left_data.frames.len());
        for (l, r) in left_data.frames.iter().zip(right_frames.iter()) {
            frames.push(concat_frames(l, r)?);
        }
        results.insert(ref_id.clone(), GrafanaDataResponse { frames });
    }
    // Keep any refIds that only appear on the right.
    for (ref_id, right_data) in &right.results {
        if results.contains_key(ref_id) {
            continue;
        }
        results.insert(
            ref_id.clone(),
            GrafanaDataResponse {
                frames: right_data
                    .frames
                    .iter()
                    .map(clone_frame)
                    .collect::<Result<Vec<_>, _>>()?,
            },
        );
    }
    Ok(GrafanaQueryDataResponse { results })
}

/// Look up / sliding-merge a cached upstream response, or fetch the full window.
///
/// Lock is not held across network calls. Phase 3: memory miss consults
/// `rsod-storage` before hitting the upstream.
pub async fn get_or_fetch(
    client: &GrafanaClient,
    body: &ProxyQueryBody,
) -> rsod_core::Result<GrafanaQueryDataResponse> {
    let scope = client.cache_scope();
    let key = series_key(&scope, body);
    let req_from = body.from;
    let req_to = body.to;

    let plan_and_cached = {
        match global().lock() {
            Ok(mut cache) => cache.get_entry(key, Instant::now()).map(|entry| {
                let plan = slide_plan(entry.from, entry.to, req_from, req_to);
                (
                    plan,
                    entry.from,
                    entry.to,
                    clone_query_response(&entry.response),
                )
            }),
            Err(e) => {
                debug!(error = %e, "upstream history cache lock poisoned; bypassing");
                None
            }
        }
    };

    let plan_and_cached = match plan_and_cached {
        Some(hit) => Some(hit),
        None => try_hydrate_from_persist(key, req_from, req_to).await,
    };

    if let Some((plan, cached_from, cached_to, cached_rsp)) = plan_and_cached {
        let cached_rsp = cached_rsp?;
        match plan {
            SlidePlan::Cover => {
                info!(
                    key,
                    ?cached_from,
                    ?cached_to,
                    "upstream history cache cover hit"
                );
                let trimmed = if req_from == cached_from && req_to == cached_to {
                    cached_rsp
                } else {
                    trim_response(&cached_rsp, req_from, req_to)?
                };
                return Ok(trimmed);
            }
            SlidePlan::ExtendRight => {
                info!(
                    key,
                    ?cached_to,
                    ?req_to,
                    "upstream history cache sliding extend right"
                );
                let reused = trim_response(&cached_rsp, req_from, cached_to)?;
                let gap_body = body_for_time_range(body, cached_to, req_to)?;
                let gap_rsp = client.data_source_query(&gap_body).await?;
                let merged = merge_responses(&reused, &gap_rsp)?;
                store(key, req_from, req_to, &merged);
                return Ok(merged);
            }
            SlidePlan::ExtendLeft => {
                info!(
                    key,
                    ?req_from,
                    ?cached_from,
                    "upstream history cache sliding extend left"
                );
                let reused = trim_response(&cached_rsp, cached_from, req_to)?;
                let gap_body = body_for_time_range(body, req_from, cached_from)?;
                let gap_rsp = client.data_source_query(&gap_body).await?;
                let merged = merge_responses(&gap_rsp, &reused)?;
                store(key, req_from, req_to, &merged);
                return Ok(merged);
            }
            SlidePlan::ExtendBoth => {
                info!(
                    key,
                    ?req_from,
                    ?cached_from,
                    ?cached_to,
                    ?req_to,
                    "upstream history cache sliding extend both"
                );
                let reused = trim_response(&cached_rsp, cached_from, cached_to)?;
                let left_body = body_for_time_range(body, req_from, cached_from)?;
                let right_body = body_for_time_range(body, cached_to, req_to)?;
                let (left_rsp, right_rsp) = tokio::try_join!(
                    client.data_source_query(&left_body),
                    client.data_source_query(&right_body),
                )?;
                let merged_left = merge_responses(&left_rsp, &reused)?;
                let merged = merge_responses(&merged_left, &right_rsp)?;
                store(key, req_from, req_to, &merged);
                return Ok(merged);
            }
            SlidePlan::Miss => {
                info!(key, "upstream history cache slide miss");
            }
        }
    } else {
        info!(key, "upstream history cache cold miss");
    }

    let rsp = client.data_source_query(body).await?;
    store(key, req_from, req_to, &rsp);
    Ok(rsp)
}

fn store(key: u64, from: DateTime<Utc>, to: DateTime<Utc>, rsp: &GrafanaQueryDataResponse) {
    let ttl = match global().lock() {
        Ok(cache) => cache.ttl,
        Err(_) => DEFAULT_TTL,
    };
    if let Ok(stored) = clone_query_response(rsp) {
        match global().lock() {
            Ok(mut cache) => cache.put(key, from, to, stored, Instant::now()),
            Err(e) => {
                debug!(error = %e, "upstream history cache lock poisoned; skip store");
            }
        }
    }
    persist_save_bg(key, from, to, rsp, ttl);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_body(from_ms: i64, to_ms: i64) -> ProxyQueryBody {
        ProxyQueryBody {
            queries: vec![json!({"refId": "A", "expr": "up", "maxDataPoints": 1500})],
            from: DateTime::from_timestamp_millis(from_ms).expect("from"),
            to: DateTime::from_timestamp_millis(to_ms).expect("to"),
            interval_ms: 60_000,
        }
    }

    fn empty_rsp() -> GrafanaQueryDataResponse {
        GrafanaQueryDataResponse {
            results: HashMap::new(),
        }
    }

    fn ts(ms: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(ms).expect("ts")
    }

    #[test]
    fn series_key_stable_across_rewritten_windows() {
        use crate::tools::body_for_time_range;
        let base = sample_body(1_000, 2_000);
        let slid = body_for_time_range(&base, ts(5_000), ts(6_000)).expect("rewrite");
        assert_eq!(series_key("ds", &base), series_key("ds", &slid));
    }

    #[test]
    fn series_key_differs_by_scope_and_interval() {
        let body = sample_body(1_000, 2_000);
        assert_ne!(series_key("ds-a", &body), series_key("ds-b", &body));
        let mut other = sample_body(1_000, 2_000);
        other.interval_ms = 30_000;
        assert_ne!(series_key("ds", &body), series_key("ds", &other));
    }

    #[test]
    fn slide_plan_cover_extend_left_right_both() {
        let c0 = ts(0);
        let c1 = ts(10_000);
        assert_eq!(slide_plan(c0, c1, ts(0), ts(10_000)), SlidePlan::Cover);
        assert_eq!(slide_plan(c0, c1, ts(2_000), ts(8_000)), SlidePlan::Cover);
        assert_eq!(
            slide_plan(c0, c1, ts(2_000), ts(12_000)),
            SlidePlan::ExtendRight
        );
        assert_eq!(
            slide_plan(c0, c1, ts(-5_000), ts(5_000)),
            SlidePlan::ExtendLeft
        );
        assert_eq!(
            slide_plan(c0, c1, ts(-5_000), ts(15_000)),
            SlidePlan::ExtendBoth
        );
        // No overlap.
        assert_eq!(slide_plan(c0, c1, ts(11_000), ts(15_000)), SlidePlan::Miss);
        assert_eq!(slide_plan(c0, c1, ts(-10_000), ts(-1)), SlidePlan::Miss);
        // Touches but does not overlap.
        assert_eq!(slide_plan(c0, c1, ts(-5_000), ts(0)), SlidePlan::Miss);
        assert_eq!(slide_plan(c0, c1, ts(10_000), ts(15_000)), SlidePlan::Miss);
    }

    #[test]
    fn lru_evicts_oldest() {
        let mut cache = HistoryCache::new(2, Duration::from_secs(60));
        let now = Instant::now();
        cache.put(1, ts(0), ts(1), empty_rsp(), now);
        cache.put(2, ts(0), ts(1), empty_rsp(), now);
        cache.put(3, ts(0), ts(1), empty_rsp(), now);
        assert_eq!(cache.len(), 2);
        assert!(cache.get_entry(1, now).is_none());
        assert!(cache.get_entry(2, now).is_some());
        assert!(cache.get_entry(3, now).is_some());
    }

    #[test]
    fn ttl_expires_entries() {
        let mut cache = HistoryCache::new(4, Duration::from_millis(10));
        let now = Instant::now();
        cache.put(7, ts(0), ts(1), empty_rsp(), now);
        assert!(cache.get_entry(7, now).is_some());
        let later = now + Duration::from_millis(20);
        assert!(cache.get_entry(7, later).is_none());
        assert_eq!(cache.len(), 0);
    }

    fn init_persist_storage() {
        let _ = rsod_storage::init_db_with_config(true, "");
    }

    fn clear_memory() {
        if let Ok(mut cache) = global().lock() {
            cache.map.clear();
            cache.order.clear();
        }
    }

    fn sample_frame_rsp() -> GrafanaQueryDataResponse {
        use grafana_plugin_sdk::data::IntoField;
        let times = vec![1_700_000_000_000_i64, 1_700_000_060_000];
        let values = vec![1.0_f64, 2.0];
        let frame = grafana_plugin_sdk::data::Frame::new("series")
            .with_field(times.into_field("Time"))
            .with_field(values.into_field("Value"));
        let mut results = HashMap::new();
        results.insert(
            "A".to_string(),
            GrafanaDataResponse {
                frames: vec![frame],
            },
        );
        GrafanaQueryDataResponse { results }
    }

    #[test]
    fn persist_round_trip_and_hydrate() {
        init_persist_storage();
        clear_memory();
        let key = 0xabc123u64;
        let from = ts(1_000);
        let to = ts(10_000);
        let rsp = sample_frame_rsp();
        persist_save(key, from, to, &rsp, Duration::from_secs(60));

        let loaded = persist_load(key).expect("persist load");
        assert_eq!(loaded.0, from);
        assert_eq!(loaded.1, to);
        assert_eq!(loaded.2.results.len(), 1);
        assert_eq!(loaded.2.results["A"].frames[0].fields().len(), 2);

        clear_memory();
        // Trial mode skips durable hydrate in get_or_fetch; exercise the
        // hydrate helper that production uses after a successful load.
        let hydrated = hydrate_loaded_entry(key, loaded, from, to).expect("hydrate");
        assert_eq!(hydrated.0, SlidePlan::Cover);
        assert!(global().lock().unwrap().get_entry(key, Instant::now()).is_some());
    }

    #[test]
    fn durable_persist_disabled_in_trial_mode() {
        init_persist_storage();
        assert!(
            rsod_storage::is_trial_mode(),
            "tests use trial SQLite"
        );
        assert!(!durable_persist_enabled());
    }

    #[test]
    fn persist_expired_is_dropped() {
        init_persist_storage();
        let key = 0xdeadbeefu64;
        let blob = PersistBlob {
            version: PERSIST_VERSION,
            from_ms: 0,
            to_ms: 1_000,
            expires_at_ms: wall_now_ms() - 1,
            response: empty_rsp(),
        };
        GlobalStorage::new()
            .save(&persist_storage_key(key), &serde_json::to_vec(&blob).unwrap())
            .unwrap();
        assert!(persist_load(key).is_none());
    }
}
