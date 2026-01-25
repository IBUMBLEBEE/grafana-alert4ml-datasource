use tonic::{Request, Response, Status};
use crate::rsod as rsodsvc;
use rsodsvc::{
    OutlierOptions as OutlierOptionsParams,
    DetectOutliersRequest,
    DetectOutliersResponse,
};
use rsod_outlier::{outlier, OutlierOptions};
use crate::utils::{arrow_to_matrix, df_to_arrow_ipc};

pub fn apply_defaults(opt: Option<OutlierOptionsParams>) -> OutlierOptions {
    let mut options = opt.unwrap_or_default();
    if options.model_name == "" {
        options.model_name = "outlier".into();
    }
    if options.periods.is_empty() {
        options.periods = vec![24];
    }

    OutlierOptions {
        model_name: options.model_name,
        periods: options.periods.iter().map(|&x | x as usize).collect(),
        uuid: options.uuid,
    }
}

pub async fn outlier_process(request: Request<DetectOutliersRequest>) -> Result<Response<DetectOutliersResponse>, Status> {
    let req = request.into_inner();
    let current_data = match arrow_to_matrix(&req.data) {
        Ok(data) => {
            data
        },
        Err(e) => {
            return Err(Status::invalid_argument(format!("Failed to convert the current data: {}", e)));
        }  
    };
    let options = apply_defaults(req.options);
    let mut result = match outlier(&current_data, &options.periods, &options.uuid) {
        Ok(v) => {
            v
        },
        Err(e) => {
            return  Err(Status::internal(format!("outlier detect error: {}", e)));
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

    Ok(Response::new(DetectOutliersResponse { data: ipc_data, error_message: "".to_string() }))
}