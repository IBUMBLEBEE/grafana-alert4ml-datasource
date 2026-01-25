use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use std::io::Cursor;
use std::error::Error;
use polars::prelude::{DataType as PolarsDataType, DataFrame as PolarsDataFrame};
use arrow::array::{Array as ArrowArray, Float64Array, Int64Array, RecordBatch, TimestampNanosecondArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub fn arrow_to_matrix(data: &[u8]) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let cursor = Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| format!("Arrow parse failed: {}", e))?;

    let mut matrix = Vec::new();
    
    for batch_result in reader {
        let batch = match batch_result {
            Ok(b) => b,
            Err(e) => {
                return Err(format!("read RecordBatch error: {}", e).into());
            }
        };
        
        if batch.num_rows() == 0 {
            continue; 
        }
        
        matrix.reserve(batch.num_rows());

        let time_array = batch.column(0);
        let times: Vec<f64> = if let Some(ts) = time_array.as_any().downcast_ref::<TimestampNanosecondArray>() {
            ts.iter().map(|t| t.unwrap_or(0) as f64 / 1_000_000_000.0).collect()
        } else if let Some(i64s) = time_array.as_any().downcast_ref::<Int64Array>() {
            i64s.iter().map(|t| t.unwrap_or(0) as f64).collect()
        } else if let Some(f64s) = time_array.as_any().downcast_ref::<Float64Array>() {
            f64s.iter().map(|t| t.unwrap_or(0.0)).collect()
        } else {
            return Err(format!("timestamp,Int64 or Float64 type: {:?}", time_array.data_type()).into());
        };

        let value_array = batch.column(1);
        let values = value_array.as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| "first column must be Float64")?;

        for i in 0..batch.num_rows() {
            matrix.push([times[i], values.value(i)]);
        }
    }
    Ok(matrix)
}

pub fn df_to_arrow_ipc(df: &mut PolarsDataFrame) -> Result<Vec<u8>, Box<dyn Error>> {
    let df = df.align_chunks();

    let column_names = df.get_column_names();
    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn ArrowArray>> = Vec::new();
    for col_name in column_names {
        let series = df.column(col_name)
            .map_err(|e| format!("Failed to get column {}: {}", col_name, e))?;

        let (field, array): (Field, Arc<dyn ArrowArray>) = match series.dtype() {
            PolarsDataType::Float64 => {
                let ca = series.f64()
                    .map_err(|e| format!("Failed to get Float64 column {}: {}", col_name, e))?;
                let values: Vec<Option<f64>> = ca.into_iter().collect();
                let arrow_array = Float64Array::from(values);
                (
                    Field::new(col_name.to_string(), DataType::Float64, true),
                    Arc::new(arrow_array),
                )
            }
            PolarsDataType::Int64 => {
                let ca = series.i64()
                    .map_err(|e| format!("Failed to get Int64 column {}: {}", col_name, e))?;
                let values: Vec<Option<i64>> = ca.into_iter().collect();
                let arrow_array = Int64Array::from(values);
                (
                    Field::new(col_name.to_string(), DataType::Int64, true),
                    Arc::new(arrow_array),
                )
            }
            _ => {
                return Err(format!("Unsupported column type: {:?} for column {}", series.dtype(), col_name).into());
            }
        };

        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| format!("Failed to create RecordBatch: {}", e))?;

    let mut buffer = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buffer, &schema)
        .map_err(|e| format!("Failed to create StreamWriter: {}", e))?;
    
    writer.write(&batch)
        .map_err(|e| format!("Failed to write batch: {}", e))?;
    writer.finish()
        .map_err(|e| format!("Failed to finish writer: {}", e))?;

    Ok(buffer)
}
