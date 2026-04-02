pub mod model;
pub mod db;

use std::sync::{Mutex, OnceLock};

use crate::db::{DbBackend, init_sqlite, init_postgres};

static INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

/// Storage configuration set before initialization.
struct StorageConfig {
    trial_mode: bool,
    pg_dsn: String,
}

static STORAGE_CONFIG: OnceLock<StorageConfig> = OnceLock::new();

/// Initialize storage with explicit configuration.
/// Call this before any model read/write operations.
/// - `trial_mode = true`: uses SQLite in-memory (volatile, data lost on restart)
/// - `trial_mode = false`: uses PostgreSQL (persistent, requires valid `pg_dsn`)
pub fn init_db_with_config(trial_mode: bool, pg_dsn: &str) -> Result<(), String> {
    let _ = STORAGE_CONFIG.set(StorageConfig {
        trial_mode,
        pg_dsn: pg_dsn.to_string(),
    });
    init_db()
}

/// Initialize the database backend.
/// Uses previously stored config from `init_db_with_config`, or defaults to trial mode (SQLite in-memory).
pub fn init_db() -> Result<(), String> {
    if let Ok(mut error_guard) = INIT_ERROR.lock() {
        *error_guard = None;
    }

    let config = STORAGE_CONFIG.get_or_init(|| StorageConfig {
        trial_mode: true,
        pg_dsn: String::new(),
    });

    let result = if config.trial_mode {
        init_sqlite_backend()
    } else {
        init_postgres_backend(&config.pg_dsn)
    };

    if let Err(e) = &result {
        if let Ok(mut error_guard) = INIT_ERROR.lock() {
            *error_guard = Some(e.clone());
        }
    }

    result
}

fn init_sqlite_backend() -> Result<(), String> {
    let backend = init_sqlite()?;

    match backend {
        DbBackend::Sqlite(mutex) => {
            let db = mutex.lock().map_err(|_| "Failed to lock SQLite database".to_string())?;

            db.execute(
                "CREATE TABLE IF NOT EXISTS models (
                    uuid TEXT PRIMARY KEY,
                    artifacts BLOB NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                )",
                [],
            )
            .map_err(|e| format!("Failed to create models table: {}", e))?;

            db.execute(
                "CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL
                )",
                [],
            )
            .map_err(|e| format!("Failed to create data table: {}", e))?;

            db.query_row("PRAGMA journal_mode=WAL", [], |row| {
                let mode: String = row.get(0)?;
                Ok(mode)
            })
            .map_err(|e| format!("Failed to set journal mode: {}", e))?;
        }
        DbBackend::Postgres(_) => {
            return Err("Expected SQLite backend but got PostgreSQL".to_string());
        }
    }

    Ok(())
}

fn init_postgres_backend(dsn: &str) -> Result<(), String> {
    let backend = init_postgres(dsn)?;

    match backend {
        DbBackend::Postgres(mutex) => {
            let mut client = mutex.lock().map_err(|_| "Failed to lock PostgreSQL client".to_string())?;

            client
                .batch_execute(
                    "CREATE TABLE IF NOT EXISTS models (
                        uuid TEXT PRIMARY KEY,
                        artifacts BYTEA NOT NULL,
                        created_at BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT),
                        updated_at BIGINT NOT NULL DEFAULT (EXTRACT(EPOCH FROM NOW())::BIGINT)
                    );
                    CREATE TABLE IF NOT EXISTS data (
                        id BIGSERIAL PRIMARY KEY,
                        timestamp DOUBLE PRECISION NOT NULL,
                        value DOUBLE PRECISION NOT NULL
                    );",
                )
                .map_err(|e| format!("Failed to create tables in PostgreSQL: {}", e))?;
        }
        DbBackend::Sqlite(_) => {
            return Err("Expected PostgreSQL backend but got SQLite".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::get_backend;

    #[test]
    fn test_init_db_sqlite_trial_mode() {
        init_db_with_config(true, "").unwrap();

        let backend = get_backend();
        match backend {
            DbBackend::Sqlite(mutex) => {
                let db = mutex.lock().unwrap();
                let mut stmt = db
                    .prepare(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('models', 'data')",
                    )
                    .unwrap();

                let tables: Vec<String> = stmt
                    .query_map([], |row| Ok(row.get::<_, String>(0)?))
                    .unwrap()
                    .map(|r| r.unwrap())
                    .collect();

                assert!(tables.contains(&"models".to_string()));
                assert!(tables.contains(&"data".to_string()));
            }
            DbBackend::Postgres(_) => {
                panic!("Expected SQLite backend in trial mode");
            }
        }
    }
}