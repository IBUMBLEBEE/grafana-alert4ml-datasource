use serde::{Serialize, Deserialize};
use rsod_core::{ModelStorage, RsodError};
use crate::db::{DbBackend, get_backend};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub uuid: String,
    pub artifacts: Vec<u8>,
}

impl Model {
    pub fn new(uuid: String, artifacts: Vec<u8>) -> Self {
        Self { uuid, artifacts }
    }

    pub fn write(&self) -> Result<(), std::io::Error> {
        println!("Model::write called for uuid: {}", self.uuid);
        crate::init_db().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let backend = get_backend();
        match backend {
            DbBackend::Sqlite(mutex) => self.write_sqlite(mutex),
            DbBackend::Postgres(mutex) => self.write_postgres(mutex),
        }
    }

    pub fn read(&mut self) -> Result<(), std::io::Error> {
        println!("Model::read called for uuid: {}", self.uuid);
        crate::init_db().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let backend = get_backend();
        match backend {
            DbBackend::Sqlite(mutex) => self.read_sqlite(mutex),
            DbBackend::Postgres(mutex) => self.read_postgres(mutex),
        }
    }

    fn write_sqlite(&self, mutex: &std::sync::Mutex<rusqlite::Connection>) -> Result<(), std::io::Error> {
        let db = mutex.lock().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::Other, "Failed to lock SQLite database")
        })?;

        let mut stmt = db
            .prepare(
                "INSERT OR REPLACE INTO models (uuid, artifacts, updated_at) \
                 VALUES (?1, ?2, strftime('%s', 'now'))",
            )
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Prepare insert statement failed: {}", e),
                )
            })?;

        stmt.execute((&self.uuid, &self.artifacts))
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Execute insert failed: {}", e),
                )
            })?;
        Ok(())
    }

    fn write_postgres(&self, mutex: &std::sync::Mutex<postgres::Client>) -> Result<(), std::io::Error> {
        let mut client = mutex.lock().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::Other, "Failed to lock PostgreSQL client")
        })?;

        client
            .execute(
                "INSERT INTO models (uuid, artifacts, updated_at) \
                 VALUES ($1, $2, EXTRACT(EPOCH FROM NOW())::BIGINT) \
                 ON CONFLICT (uuid) DO UPDATE SET artifacts = $2, updated_at = EXTRACT(EPOCH FROM NOW())::BIGINT",
                &[&self.uuid, &self.artifacts],
            )
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("PostgreSQL insert failed: {}", e),
                )
            })?;
        Ok(())
    }

    fn read_sqlite(&mut self, mutex: &std::sync::Mutex<rusqlite::Connection>) -> Result<(), std::io::Error> {
        let db = mutex.lock().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::Other, "Failed to lock SQLite database")
        })?;

        let mut stmt = db
            .prepare("SELECT uuid, artifacts FROM models WHERE uuid = ?")
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Prepare select statement failed: {}", e),
                )
            })?;

        let rows = stmt
            .query_map([&self.uuid], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Query map failed: {}", e),
                )
            })?;

        for row_result in rows {
            let (uuid, artifacts) = row_result.map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Row processing failed: {}", e),
                )
            })?;
            self.uuid = uuid;
            self.artifacts = artifacts;
        }

        Ok(())
    }

    fn read_postgres(&mut self, mutex: &std::sync::Mutex<postgres::Client>) -> Result<(), std::io::Error> {
        let mut client = mutex.lock().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::Other, "Failed to lock PostgreSQL client")
        })?;

        let rows = client
            .query(
                "SELECT uuid, artifacts FROM models WHERE uuid = $1",
                &[&self.uuid],
            )
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("PostgreSQL query failed: {}", e),
                )
            })?;

        for row in rows {
            self.uuid = row.get(0);
            self.artifacts = row.get(1);
        }

        Ok(())
    }
}

/// Global storage backend that delegates to the initialized database.
///
/// Uses the same global `DbBackend` as `Model::read` / `Model::write`.
/// Implements `rsod_core::ModelStorage` so that other crates can
/// depend only on the trait, not on concrete storage details.
#[derive(Debug, Clone, Copy)]
pub struct GlobalStorage;

impl GlobalStorage {
    pub fn new() -> Self {
        GlobalStorage
    }
}

impl ModelStorage for GlobalStorage {
    fn save(&self, uuid: &str, artifacts: &[u8]) -> rsod_core::Result<()> {
        let model = Model::new(uuid.to_string(), artifacts.to_vec());
        model.write().map_err(|e| RsodError::Storage(e.to_string()))
    }

    fn load(&self, uuid: &str) -> rsod_core::Result<Option<Vec<u8>>> {
        let mut model = Model::new(uuid.to_string(), vec![]);
        model.read().map_err(|e| RsodError::Storage(e.to_string()))?;
        if model.artifacts.is_empty() {
            Ok(None)
        } else {
            Ok(Some(model.artifacts))
        }
    }

    fn delete(&self, uuid: &str) -> rsod_core::Result<()> {
        crate::init_db().map_err(|e| RsodError::Storage(e))?;
        let backend = get_backend();
        match backend {
            DbBackend::Sqlite(mutex) => {
                let db = mutex.lock().map_err(|_| {
                    RsodError::Storage("Failed to lock SQLite database".into())
                })?;
                db.execute("DELETE FROM models WHERE uuid = ?1", [uuid])
                    .map_err(|e| RsodError::Storage(format!("SQLite delete failed: {}", e)))?;
            }
            DbBackend::Postgres(mutex) => {
                let mut client = mutex.lock().map_err(|_| {
                    RsodError::Storage("Failed to lock PostgreSQL client".into())
                })?;
                client
                    .execute("DELETE FROM models WHERE uuid = $1", &[&uuid])
                    .map_err(|e| RsodError::Storage(format!("PostgreSQL delete failed: {}", e)))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        crate::init_db_with_config(true, "").unwrap();

        let _model = Model::new("test".to_string(), vec![1, 2, 3]);
        _model.write().unwrap();
        let mut model = Model::new("test".to_string(), vec![]);
        model.read().unwrap();
        assert_eq!(model.uuid, "test");
        assert_eq!(model.artifacts, vec![1, 2, 3]);
    }

    #[test]
    fn test_global_storage_trait() {
        crate::init_db_with_config(true, "").unwrap();
        let storage = GlobalStorage::new();

        // save
        storage.save("trait-test", &[10, 20, 30]).unwrap();

        // exists
        assert!(storage.exists("trait-test").unwrap());
        assert!(!storage.exists("non-existent").unwrap());

        // load
        let artifacts = storage.load("trait-test").unwrap();
        assert_eq!(artifacts, Some(vec![10, 20, 30]));

        // load non-existent
        let none = storage.load("non-existent").unwrap();
        assert!(none.is_none());

        // delete
        storage.delete("trait-test").unwrap();
        assert!(!storage.exists("trait-test").unwrap());
    }
}