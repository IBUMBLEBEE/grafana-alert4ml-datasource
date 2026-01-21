use serde::{Serialize, Deserialize};

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

        let db = crate::get_db_connection();

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

    pub fn read(&mut self) -> Result<(), std::io::Error> {
        println!("Model::read called for uuid: {}", self.uuid);
        crate::init_db().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let db = crate::get_db_connection();

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        crate::init_db().unwrap();

        let _model = Model::new("test".to_string(), vec![1, 2, 3]);
        _model.write().unwrap();
        let mut model = Model::new("test".to_string(), vec![]);
        model.read().unwrap();
        assert_eq!(model.uuid, "test");
        assert_eq!(model.artifacts, vec![1, 2, 3]);
    }
}