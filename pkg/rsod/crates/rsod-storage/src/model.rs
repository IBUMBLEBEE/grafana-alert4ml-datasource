use rusqlite::Connection;
use serde::{Serialize, Deserialize};
use crate::get_db_path;

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
        // 确保数据库已初始化
        crate::init_db(None).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        let db = Connection::open(get_db_path()).unwrap();
        let mut stmt = db.prepare("INSERT OR REPLACE INTO models (uuid, artifacts, updated_at) VALUES (?, ?, strftime('%s', 'now'))").unwrap();
        stmt.execute(rusqlite::params![self.uuid, self.artifacts]).unwrap();
        Ok(())
    }

    pub fn read(&mut self) -> Result<(), std::io::Error> {
        // 确保数据库已初始化
        crate::init_db(None).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        let db = Connection::open(get_db_path()).unwrap();
        let mut stmt = db.prepare("SELECT uuid, artifacts FROM models WHERE uuid = ?").unwrap();
        let rows = stmt.query_map([&self.uuid], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        }).unwrap();
        for row_result in rows {
            let (uuid, artifacts) = row_result.unwrap();
            self.uuid = uuid;
            self.artifacts = artifacts;
        }
        Ok(())  
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let _model = Model::new("test".to_string(), vec![1, 2, 3]);
        _model.write().unwrap();
        let mut model = Model::new("test".to_string(), vec![]);
        model.read().unwrap();
        assert_eq!(model.uuid, "test");
        assert_eq!(model.artifacts, vec![1, 2, 3]);
    }
}