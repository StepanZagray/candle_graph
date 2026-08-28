//! Cargo subcommand entrypoint: `cargo candle-graph …`.
//!
//! Installed as `cargo-candle-graph` so Cargo discovers it as `cargo candle-graph`.
//! Cargo forwards the subcommand name through argv, so the shared [`Cli`]
//! parser strips a leading `candle-graph` element before parsing.

use anyhow::Result;

use candle_graph::cli::args::Cli;

fn main() -> Result<()> {
    Cli::parse_as_cargo_subcommand().run()
}
