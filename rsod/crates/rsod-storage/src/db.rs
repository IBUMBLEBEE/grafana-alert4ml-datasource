use std::sync::{Mutex, OnceLock};
use rusqlite::Connection;

static DB_CONN: OnceLock<Mutex<Connection>> = OnceLock::new();
pub fn get_db_mutex() -> &'static Mutex<Connection> {
    DB_CONN.get().expect("Database not initialized. Call init_db() first.")
}

pub fn init_db_mutex() -> Result<&'static Mutex<Connection>, String> {
    if let Some(mutex) = DB_CONN.get() {
        return Ok(mutex);
    }

    let conn = Connection::open("file::memory:?cache=shared")
        .map_err(|e| format!("Failed to open shared memory database: {}", e))?;
    let _ = DB_CONN.set(Mutex::new(conn));

    Ok(DB_CONN.get().expect("DB_CONN must be initialized"))
}

