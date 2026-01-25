use tonic::{Request, Response, Status};
use crate::rsod as rsodsvc;
use rsodsvc::{ForecasterOptions as ForecasterOptionsParams, ForecastRequest, ForecastResponse};
use rsod_forecaster::{forecast, ForecasterOptions};

use crate::utils::{arrow_to_matrix, df_to_arrow_ipc};

pub fn apply_defaults(opt: Option<ForecasterOptionsParams>) -> ForecasterOptions {
    let mut options = opt.unwrap_or_default();

    if options.model_name.is_empty() {
        options.model_name = "perpetual-ml".to_string();
    }
    
    if options.periods.is_empty() {
        options.periods = vec![24];
    }

    if options.budget == 0.0 {
        options.budget = 0.5;
    }
    
    if options.num_threads == 0{
        options.num_threads = 1;
    }

    if options.n_lags == 0 {
        options.n_lags = 24;
    }

    if options.std_dev_multiplier == 0.0 {
        options.std_dev_multiplier = 2.0;
    }

    ForecasterOptions {
        model_name: options.model_name,
        periods: options.periods.iter().map(|&x| x as usize).collect(),
        budget: Some(options.budget as f32),
        uuid: options.uuid,
        num_threads: Some(options.num_threads as usize),
        n_lags: Some(options.n_lags as usize),
        std_dev_multiplier: Some(options.std_dev_multiplier as f64),
        allow_negative_bounds: Some(options.allow_negative_bounds),
    }
}

pub async fn forecast_process(request: Request<ForecastRequest>) -> Result<Response<ForecastResponse>, Status> {
    let req = request.into_inner();
    
    let current_data = match arrow_to_matrix(&req.current_data) {
        Ok(data) => {
            data
        },
        Err(e) => {
            return Err(Status::invalid_argument(format!("Failed to convert the current data: {}", e)));
        }  
    };

    let history_data = match arrow_to_matrix(&req.history_data) {
        Ok(data) => {
            data
        },
        Err(e) => {
            return Err(Status::invalid_argument(format!("Failed to convert the history data: {}", e)));
        }  
    };

    let options = apply_defaults(req.options);
    
    let mut result = match forecast(&current_data, &history_data, &options) {
        Ok(res) => {
            res
        },
        Err(e) => {
            return Err(Status::internal(format!("predict failed: {}", e)));
        }
    };

    let ipc_data = match df_to_arrow_ipc(&mut result) {
        Ok(data) => {
            data
        },
        Err(e) => {
            return Err(Status::internal(format!("Arrow IPC error: {}", e)));
        }
    };

    Ok(Response::new(ForecastResponse { data: ipc_data, error_message: "".to_string() }))
}
