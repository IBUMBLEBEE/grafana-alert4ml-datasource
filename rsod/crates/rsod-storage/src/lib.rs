pub mod model;

use std::path::PathBuf;
use std::sync::{Once, OnceLock, Mutex};
use rusqlite::Connection;

const DB_PATH: &str = "rsod_sqlite.db";

static INIT: Once = Once::new();
// 存储从 Golang 传入的数据库路径
static DB_PATH_STORED: OnceLock<PathBuf> = OnceLock::new();
// 存储初始化错误（如果有）
static INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

pub fn get_db_path() -> PathBuf {
    // 优先返回存储的路径（如果存在），否则返回默认路径
    DB_PATH_STORED.get().cloned().unwrap_or_else(|| {
        // db文件放在当前目录下的.rsod_sqlite.db文件中
        // 如果无法获取当前目录，使用相对路径
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(DB_PATH)
    })
}

/// 初始化数据库，创建所有必要的表
/// 使用 Once 确保线程安全且只执行一次
/// 
/// # 使用示例
/// ```no_run
/// use rsod_storage::init_db;
/// 
/// fn main() {
///     // 程序启动时调用一次
///     init_db().expect("Failed to initialize database");
///     
///     // 之后可以安全地使用 Model 和 DataSet
/// }
/// ```
pub fn init_db(db_path_opts: Option<PathBuf>) -> rusqlite::Result<()> {
    // 清除之前的错误（如果锁被 poison，忽略错误继续执行）
    if let Ok(mut error_guard) = INIT_ERROR.lock() {
        *error_guard = None;
    }
    
    INIT.call_once(|| {
        let result = (|| -> Result<(), String> {
            let db_path = if let Some(path) = db_path_opts {
                path
            } else {
                // 如果未提供路径，使用默认路径
                let current_dir = std::env::current_dir()
                    .map_err(|e| format!("Failed to get current directory: {}", e))?;
                current_dir.join(DB_PATH)
            };
            
            // 确保父目录存在
            if let Some(parent) = db_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("Failed to create parent directory {:?}: {}", parent, e))?;
                }
            }
            
            // 存储数据库路径，以便后续 get_db_path() 可以获取
            DB_PATH_STORED.set(db_path.clone())
                .map_err(|_| "Failed to store database path".to_string())?;
            
            // 打开数据库连接（如果文件不存在会自动创建）
            let db = Connection::open(&db_path)
                .map_err(|e| format!("Failed to open database at {:?}: {}", db_path, e))?;
            
            // 创建 models 表（执行 SQL 语句会触发文件创建）
            db.execute(
                "CREATE TABLE IF NOT EXISTS models (
                    uuid TEXT PRIMARY KEY,
                    artifacts BLOB NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
                    updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
                )",
                [],
            ).map_err(|e| format!("Failed to create models table: {}", e))?;
            
            // 创建 data 表（用于 DataSet，如果存在）
            db.execute(
                "CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL
                )",
                [],
            ).map_err(|e| format!("Failed to create data table: {}", e))?;
            
            // 设置 WAL 模式（PRAGMA 返回结果，需要使用 query_row）
            db.query_row("PRAGMA journal_mode=WAL", [], |row| {
                let mode: String = row.get(0)?;
                Ok(mode)
            }).map_err(|e| format!("Failed to set journal mode: {}", e))?;
            
            Ok(())
        })();
        
        if let Err(e) = result {
            if let Ok(mut error_guard) = INIT_ERROR.lock() {
                *error_guard = Some(e);
            }
        }
    });
    
    // 检查是否有错误
    if let Ok(error_guard) = INIT_ERROR.lock() {
        if let Some(ref error) = *error_guard {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(error.clone()),
            ));
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_init_db() {
        let test_db = "/tmp/test_init.db";
        if std::path::Path::new(test_db).exists() {
            fs::remove_file(test_db).unwrap();
        }

        let db_path = PathBuf::from(test_db);
    
        init_db(Some(db_path)).unwrap();
        
        assert!(Path::new(test_db).exists());
        
        let db = Connection::open(&test_db).unwrap();
        let mut stmt = db.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('models', 'data')"
        ).unwrap();
        
        let tables: Vec<String> = stmt.query_map([], |row| {
            Ok(row.get::<_, String>(0)?)
        }).unwrap()
        .map(|r| r.unwrap())
        .collect();
        
        assert!(tables.contains(&"models".to_string()));
    }
}