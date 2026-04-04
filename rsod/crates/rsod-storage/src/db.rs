use std::sync::{Mutex, OnceLock};
use rusqlite::Connection as SqliteConnection;
use postgres::{Client as PgClient, NoTls};

#[allow(missing_debug_implementations)]
pub enum DbBackend {
    Sqlite(Mutex<SqliteConnection>),
    Postgres(Mutex<PgClient>),
}

static DB_BACKEND: OnceLock<DbBackend> = OnceLock::new();

pub fn get_backend() -> &'static DbBackend {
    DB_BACKEND
        .get()
        .expect("Database not initialized. Call init_db() first.")
}

pub fn init_sqlite() -> Result<&'static DbBackend, String> {
    if let Some(backend) = DB_BACKEND.get() {
        return Ok(backend);
    }

    let conn = SqliteConnection::open("file::memory:?cache=shared")
        .map_err(|e| format!("Failed to open shared memory database: {}", e))?;
    let _ = DB_BACKEND.set(DbBackend::Sqlite(Mutex::new(conn)));

    Ok(DB_BACKEND.get().expect("DB_BACKEND must be initialized"))
}

pub fn init_postgres(dsn: &str) -> Result<&'static DbBackend, String> {
    if let Some(backend) = DB_BACKEND.get() {
        return Ok(backend);
    }

    let client = PgClient::connect(dsn, NoTls)
        .map_err(|e| format!("Failed to connect to PostgreSQL: {}", e))?;
    let _ = DB_BACKEND.set(DbBackend::Postgres(Mutex::new(client)));

    Ok(DB_BACKEND.get().expect("DB_BACKEND must be initialized"))
}

