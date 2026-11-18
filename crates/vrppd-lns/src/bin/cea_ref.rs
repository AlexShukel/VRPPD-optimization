//! Reference values: best of `--runs` CEA runs (thesis settings, wall-time
//! budget) per instance in the chosen objective. Runs instances sequentially
//! because CEA is itself multi-threaded: it parallelises on rayon's global
//! pool, which this binary sizes to the recorded `threads` before any run.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};
use vrppd_cea::{CeaConfig, solve_cea_seeded};
use vrppd_core::Objective;
use vrppd_lns::io::{cell_name, init_threads, load_instances, resolve_cell_dir};
use vrppd_lns::{objective_name, parse_objective};

#[derive(Parser, Debug)]
#[command(
  name = "cea_ref",
  about = "CEA reference values (best of N runs) per instance."
)]
struct Args {
  #[arg(long)]
  cell: String,
  /// Objective the CEA minimises and the values are given in: distance | price | empty.
  #[arg(long, default_value = "distance", value_parser = parse_objective)]
  objective: Objective,
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
  /// Machine name written into every new record as `host` (pipeline bookkeeping).
  #[arg(long, default_value = "")]
  host_tag: String,
  /// CPU list the process is pinned to (e.g. the taskset list), written into
  /// every new record as `cpu_list`.
  #[arg(long, default_value = "")]
  cpu_list: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct RefRecord {
  cell: String,
  instance: String,
  /// Objective of `runs` and `best`; files written before the field existed
  /// are distance references.
  #[serde(default = "vrppd_lns::distance_name")]
  objective: String,
  n_orders: usize,
  n_vehicles: usize,
  runs: Vec<f64>,
  runtimes_ms: Vec<f64>,
  generations: Vec<u64>,
  exit_reasons: Vec<String>,
  best: f64,
  /// Machine, CEA threads and CPU list of the runs; empty / 0 in files written
  /// before these fields existed.
  #[serde(default)]
  host: String,
  #[serde(default)]
  threads: usize,
  #[serde(default)]
  cpu_list: String,
  /// Size of rayon's global pool the runs used (absent in records written
  /// before the pool was sized; equal to `threads` since).
  #[serde(default, skip_serializing_if = "Option::is_none")]
  rayon_threads: Option<usize>,
  /// Seed of each run, in the order of `runs` (absent in older records).
  #[serde(default, skip_serializing_if = "Vec::is_empty")]
  seeds: Vec<u64>,
}

fn main() {
  let args = Args::parse();
  // The CEA runs on rayon's global pool: size it before anything uses it
  // (RAYON_NUM_THREADS can only lower it), so `threads` is the worker count.
  let threads = init_threads(args.threads, 20);
  let rayon_threads = rayon::current_num_threads();
  if rayon_threads != threads {
    eprintln!("cea_ref: rayon's global pool has {rayon_threads} threads, expected {threads}");
    std::process::exit(1);
  }
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
  let objective = objective_name(args.objective);
  if let Some(other) = records.iter().find(|r| r.objective != objective) {
    eprintln!(
      "{} holds '{}' records but --objective is '{objective}'; use another --out",
      args.out.display(),
      other.objective
    );
    std::process::exit(1);
  }
  eprintln!(
    "cea_ref: cell={cell} objective={objective} instances={} runs={} seconds={} threads={threads}",
    instances.len(),
    args.runs,
    args.seconds
  );
  for inst in &instances {
    if records
      .iter()
      .any(|r| r.instance == inst.id && r.cell == cell && r.objective == objective)
    {
      continue;
    }
    let mut rec = RefRecord {
      cell: cell.clone(),
      instance: inst.id.clone(),
      objective: objective.to_string(),
      n_orders: inst.problem.orders.len(),
      n_vehicles: inst.problem.vehicles.len(),
      runs: Vec::new(),
      runtimes_ms: Vec::new(),
      generations: Vec::new(),
      exit_reasons: Vec::new(),
      best: f64::INFINITY,
      host: args.host_tag.clone(),
      threads,
      cpu_list: args.cpu_list.clone(),
      rayon_threads: Some(rayon_threads),
      seeds: Vec::new(),
    };
    for r in 0..args.runs {
      // The Stage 2 derivation, unchanged: the seed depends on the instance
      // only through the length of its id, so instances whose ids have the
      // same length share their run seeds.
      let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(r as u64) ^ (inst.id.len() as u64);
      let cfg = CeaConfig {
        wall_time_cap_ms: Some(args.seconds * 1000),
        threads,
        ..CeaConfig::default()
      };
      let t = Instant::now();
      let solved = solve_cea_seeded(&inst.problem, args.objective, cfg, seed);
      let v = args.objective.energy(&solved.solution);
      rec.runs.push(v);
      rec.seeds.push(seed);
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
