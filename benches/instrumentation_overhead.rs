//! Reproducible microbenchmarks for the disabled capture gate and active event serialization.

use std::hint::black_box;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_graph::{CaptureSelector, OpRecord, ProfileRun, TraceSession};

const ITERATIONS: u64 = 100_000;

fn main() -> anyhow::Result<()> {
    let selector = CaptureSelector::new(50_001)?;
    let started = Instant::now();
    let mut matches = 0u64;
    for invocation in 1..=ITERATIONS {
        if black_box(selector).is_selected(black_box(invocation)) {
            matches += 1;
        }
    }
    println!(
        "disabled capture gate: {:.2} ns/invocation ({matches} selected)",
        started.elapsed().as_nanos() as f64 / ITERATIONS as f64
    );

    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = std::env::temp_dir().join(format!(
        "candle-graph-overhead-{}-{nonce}.jsonl",
        std::process::id()
    ));
    let active_iterations = 10_000u64;
    let session = TraceSession::open(&path, ProfileRun::inference("bench::infer", 1, "cpu"))?;
    let measured = session.begin_measurement("bench/inference-1");
    let started = Instant::now();
    for _ in 0..active_iterations {
        session.record_op(
            measured.id(),
            OpRecord {
                op_name: "add",
                inputs: &[],
                output: None,
                shape: &[32, 32],
                dtype: "f32",
                device: "cpu",
                duration_ns: 1,
                timestamp_ns: 0,
                output_dense_bytes: Some(4096),
                input_dense_bytes: 8192,
            },
        )?;
    }
    let elapsed = started.elapsed();
    drop(measured);
    session.finish()?;
    println!(
        "active op event: {:.2} ns/event",
        elapsed.as_nanos() as f64 / active_iterations as f64
    );
    std::fs::remove_file(&path)?;
    Ok(())
}
