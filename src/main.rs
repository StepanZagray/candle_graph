//! Standalone `candle-graph` binary — same trace commands as `cargo candle-graph`.

use anyhow::Result;
use clap::Parser;

use candle_graph::cli::args::Cli;

fn main() -> Result<()> {
    Cli::parse().run()
}
