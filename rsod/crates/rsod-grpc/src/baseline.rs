use tonic::{Request, Response, Status};
use crate::rsod as rsodsvc;
use rsodsvc::{
    BaselineOptions as BaselineOptionsParams,
    DetectBaselineRequest,
    DetectBaselineResponse,
};
use rsod_baseline::{baseline_detect, BaselineOptions, TrendType};
use crate::utils::{arrow_to_matrix, df_to_arrow_ipc};

pub fn apply_defaults(opt: Option<BaselineOptionsParams>) -> BaselineOptions {
    let mut options = opt.unwrap_or_default();
    if options.confidence_level == 0.0 {
        options.confidence_level = 0.95;
    }
    if options.std_dev_multiplier == 0.0 {
        options.std_dev_multiplier = 0.2;
    }
    if options.interval_mins == 0 {
        options.interval_mins = 15;
    }
    
    // Convert protobuf TrendType (i32) to rsod_baseline::TrendType
    // protobuf enum values: Unspecified=0, Daily=1, Weekly=2, Monthly=3
    // rsod_baseline enum values: Daily, Weekly, Monthly, None
    let trend_type = match options.trend_type {
        0 | 1 => TrendType::Daily,
        2 => TrendType::Weekly,
        3 => TrendType::Monthly,
        _ => TrendType::Daily,
    };

    BaselineOptions {
        confidence_level: Some(options.confidence_level),
        std_dev_multiplier: Some(options.std_dev_multiplier),
        interval_mins: Some(options.interval_mins as u32),
        trend_type,
        allow_negative_bounds: Some(options.allow_negative_bounds),
        uuid: options.uuid,
    }
}

pub async fn baseline_process(request: Request<DetectBaselineRequest>) -> Result<Response<DetectBaselineResponse>, Status> {
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
    let mut result = match baseline_detect(&current_data, &history_data, &options) {
        Ok(res) => {
            res
        },
        Err(e) => {
            return Err(Status::internal(format!("baseline detect error: {}", e)));
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
    Ok(Response::new(DetectBaselineResponse { data: ipc_data, error_message: "".to_string() }))
}