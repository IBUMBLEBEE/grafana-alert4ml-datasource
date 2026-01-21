pub mod model;
mod db;

use std::sync::Mutex;
use rusqlite::Connection;

use crate::db::{get_db_mutex, init_db_mutex};

static INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

pub fn get_db_connection() -> std::sync::MutexGuard<'static, Connection> {
    get_db_mutex()
        .lock()
        .expect("Database mutex poisoned")
}


pub fn init_db() -> rusqlite::Result<()> {
    if let Ok(mut error_guard) = INIT_ERROR.lock() {
        *error_guard = None;
    }

    let result = (|| -> Result<(), String> {
        let db_mutex = init_db_mutex()?;
        let db = db_mutex
            .lock()
            .map_err(|_| "Failed to lock database".to_string())?;

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

        Ok(())
    })();

    if let Err(e) = result {
        if let Ok(mut error_guard) = INIT_ERROR.lock() {
            *error_guard = Some(e.clone());
        }
        return Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
            Some(e),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_db() {
        init_db().unwrap();

        let db = get_db_connection();

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
}