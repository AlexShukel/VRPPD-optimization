//! Reference values: best of `--runs` CEA runs (thesis settings, wall-time
//! budget) per instance, distance objective. Runs instances sequentially
//! because CEA is itself multi-threaded.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};
use vrppd_cea::{CeaConfig, solve_cea_seeded};
use vrppd_core::Objective;
use vrppd_lns::io::{cell_name, load_instances, resolve_cell_dir};

#[derive(Parser, Debug)]
#[command(
  name = "cea_ref",
  about = "CEA reference values (best of N runs) per instance."
)]
struct Args {
  #[arg(long)]
  cell: String,
  #[arg(long, default_value_t = 3)]
  runs: usize,
  #[arg(long, default_value_t = 60)]
  seconds: u64,
  #[arg(long, default_value_t = 6)]
  threads: usize,
  #[arg(long, default_value_t = 7)]
  seed: u64,
  /// Output JSON path (one record per instance).
  #[arg(long)]
  out: PathBuf,
  #[arg(long)]
  max_instances: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct RefRecord {
  cell: String,
  instance: String,
  n_orders: usize,
  n_vehicles: usize,
  runs: Vec<f64>,
  runtimes_ms: Vec<f64>,
  generations: Vec<u64>,
  exit_reasons: Vec<String>,
  best: f64,
}

fn main() {
  let args = Args::parse();
  let dir = resolve_cell_dir(&args.cell);
  let cell = cell_name(&dir);
  let instances = load_instances(&dir, args.max_instances).unwrap_or_else(|e| {
    eprintln!("{e}");
    std::process::exit(1)
  });
  if let Some(parent) = args.out.parent() {
    fs::create_dir_all(parent).expect("create out dir");
  }
  // Resume: keep records already written.
  let mut records: Vec<RefRecord> = fs::read_to_string(&args.out)
    .ok()
    .and_then(|t| serde_json::from_str(&t).ok())
    .unwrap_or_default();
  let threads = args.threads.clamp(1, 6);
  eprintln!(
    "cea_ref: cell={cell} instances={} runs={} seconds={} threads={threads}",
    instances.len(),
    args.runs,
    args.seconds
  );
  for inst in &instances {
    if records
      .iter()
      .any(|r| r.instance == inst.id && r.cell == cell)
    {
      continue;
    }
    let mut rec = RefRecord {
      cell: cell.clone(),
      instance: inst.id.clone(),
      n_orders: inst.problem.orders.len(),
      n_vehicles: inst.problem.vehicles.len(),
      runs: Vec::new(),
      runtimes_ms: Vec::new(),
      generations: Vec::new(),
      exit_reasons: Vec::new(),
      best: f64::INFINITY,
    };
    for r in 0..args.runs {
      let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(r as u64) ^ (inst.id.len() as u64);
      let cfg = CeaConfig {
        wall_time_cap_ms: Some(args.seconds * 1000),
        threads,
        ..CeaConfig::default()
      };
      let t = Instant::now();
      let solved = solve_cea_seeded(&inst.problem, Objective::Distance, cfg, seed);
      let v = solved.solution.total_distance;
      rec.runs.push(v);
      rec.runtimes_ms.push(t.elapsed().as_secs_f64() * 1000.0);
      rec.generations.push(solved.generations);
      rec.exit_reasons.push(format!("{:?}", solved.exit_reason));
      rec.best = rec.best.min(v);
    }
    eprintln!(
      "cea_ref: {cell}/{} best={:.3} runs={:?} times={:?}",
      inst.id,
      rec.best,
      rec
        .runs
        .iter()
        .map(|v| format!("{v:.1}"))
        .collect::<Vec<_>>(),
      rec
        .runtimes_ms
        .iter()
        .map(|v| format!("{:.0}", v / 1000.0))
        .collect::<Vec<_>>()
    );
    records.push(rec);
    let tmp = args.out.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_string_pretty(&records).unwrap()).expect("write");
    fs::rename(&tmp, &args.out).expect("rename");
  }
  eprintln!(
    "cea_ref: {} records -> {}",
    records.len(),
    args.out.display()
  );
}
