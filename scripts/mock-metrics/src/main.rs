//! Alert4ML mock metrics API for Grafana Infinity.
//!
//! Deterministic scenario formulas (scheme 2): value(t) is a pure function of
//! scenario + unix time, so restarts and repeated queries stay reproducible.
//!
//! Infinity example URL:
//!   http://mock-metrics:9108/api/series?scenario=weekly&from=${__from}&to=${__to}&step=60000

mod http;
mod scenario;
mod series;

use clap::Parser;
use std::net::SocketAddr;

const PORT_DEFAULT: u16 = 9108;

#[derive(Debug, Parser)]
#[command(about = "Alert4ML mock metrics API for Grafana Infinity")]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = PORT_DEFAULT)]
    port: u16,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let addr: SocketAddr = match format!("{}:{}", args.host, args.port).parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("invalid --host/--port: {e}");
            std::process::exit(1);
        }
    };

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "alert4ml mock-metrics listening on http://{}:{} scenarios={}",
        args.host,
        args.port,
        scenario::SCENARIOS.join(",")
    );

    if let Err(e) = axum::serve(listener, http::app()).await {
        eprintln!("server error: {e}");
        std::process::exit(1);
    }
}
