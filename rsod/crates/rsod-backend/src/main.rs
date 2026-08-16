//! Alert4ML Grafana datasource backend plugin.
//!
//! Entry point for the plugin process. Grafana launches the `gpx_alert4ml`
//! binary, which speaks gRPC via the `grafana-plugin-sdk`.

mod client;
mod config;
mod contract;
mod frame_ops;
mod health;
mod history_cache;
mod pipeline;
mod plugin;
mod render;
mod tools;
mod uuid_util;

use plugin::PluginService;

#[grafana_plugin_sdk::main(services(data, diagnostics), init_subscriber = true)]
async fn plugin() -> PluginService {
    pipeline::assert_engines_registered();
    PluginService::new()
}
