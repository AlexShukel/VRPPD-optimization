//! PSA grid-search tuner. Sweeps `initial_temp × cooling_rate × max_iterations` over a list of cells from `problem-gen/problems/<cell>/`, repeats each (config, instance) several times with distinct seeds, ranks configs by mean objective.

use std::fmt::Write;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;
use vrppd_benchmarks::{
  load_cell_instances, make_run_dir, mean, parse_cells, parse_objective, stdev, write_run_outputs,
};
use vrppd_core::Objective;
use vrppd_psa::{default_config_for, solve_seeded, OperatorWeights, SaConfig};

#[derive(Parser, Debug)]
#[command(name = "tune_psa", about = "Grid search over PSA hyperparameters.")]
struct Args {
  #[arg(long, value_name = "CELL_LIST")]
  cells: String,

  #[arg(long, default_value_t = 3)]
  reps: usize,

  #[arg(long, default_value = "distance")]
  objective: String,

  #[arg(long, value_delimiter = ',', num_args = 1..)]
  initial_temp: Option<Vec<f64>>,

  #[arg(long, value_delimiter = ',', num_args = 1..)]
  cooling_rate: Option<Vec<f64>>,

  #[arg(long, value_delimiter = ',', num_args = 1..)]
  max_iterations: Option<Vec<u64>>,

  #[arg(long, default_value_t = 1)]
  threads: usize,

  #[arg(long, default_value_t = false)]
  debug: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigSpec {
  initial_temp: f64,
  cooling_rate: f64,
  max_iterations: u64,
}

#[derive(Clone, Debug, Serialize)]
struct RunRecord {
  config_index: usize,
  config: ConfigSpec,
  cell: String,
  instance_id: String,
  rep: usize,
  seed: u64,
  objective_value: f64,
  runtime_ms: u128,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigSummary {
  config_index: usize,
  config: ConfigSpec,
  mean_objective: f64,
  stdev_objective: f64,
  mean_runtime_ms: f64,
  runs: usize,
}

#[derive(Serialize)]
struct ResultsJson {
  run_kind: &'static str,
  objective: String,
  cells: Vec<String>,
  reps: usize,
  configs: Vec<ConfigSpec>,
  records: Vec<RunRecord>,
  ranking: Vec<ConfigSummary>,
}

fn main() {
  let args = Args::parse();
  let objective = parse_objective(&args.objective).expect("--objective");
  let cells = parse_cells(&args.cells);
  if cells.is_empty() {
    eprintln!("--cells must contain at least one cell (e.g. --cells 10_10,20_50)");
    std::process::exit(2);
  }

  let initial_temps = args.initial_temp.unwrap_or_else(|| vec![200.0, 500.0, 1000.0]);
  let cooling_rates = args.cooling_rate.unwrap_or_else(|| vec![0.99, 0.995, 0.999]);
  let max_iters = args
    .max_iterations
    .unwrap_or_else(|| vec![5_000, 20_000, 50_000]);

  let mut configs: Vec<ConfigSpec> = Vec::new();
  for &t in &initial_temps {
    for &c in &cooling_rates {
      for &m in &max_iters {
        configs.push(ConfigSpec {
          initial_temp: t,
          cooling_rate: c,
          max_iterations: m,
        });
      }
    }
  }

  let mut instances = Vec::new();
  for cell in &cells {
    let mut loaded = load_cell_instances(cell).unwrap_or_else(|e| {
      eprintln!("failed to load cell {cell}: {e}");
      std::process::exit(1);
    });
    instances.append(&mut loaded);
  }
  eprintln!(
    "tune_psa: {} configs × {} instances × {} reps = {} runs",
    configs.len(),
    instances.len(),
    args.reps,
    configs.len() * instances.len() * args.reps,
  );

  let base = default_config_for(objective);
  let mut records: Vec<RunRecord> = Vec::new();

  for (cidx, cfg) in configs.iter().enumerate() {
    let sa_config = SaConfig {
      initial_temp: cfg.initial_temp,
      cooling_rate: cfg.cooling_rate,
      min_temp: base.min_temp,
      max_iterations: cfg.max_iterations,
      weights: OperatorWeights::default(),
      threads: args.threads.max(1),
      batch_size: base.batch_size,
      sync_interval: base.sync_interval,
      reheat_floor: base.reheat_floor,
    };

    for inst in &instances {
      for rep in 0..args.reps {
        let seed = (cidx as u64) * 1_000_003 + (rep as u64) * 7919 + inst_id_hash(&inst.id);
        let started = Instant::now();
        let solved = solve_seeded(&inst.problem, objective, sa_config, seed);
        let runtime_ms = started.elapsed().as_millis();
        let value = objective.energy(&solved.solution);
        if args.debug {
          eprintln!(
            "[debug] cfg#{cidx:02} cell={} id={} rep={} value={:.4} time={}ms",
            inst.cell, inst.id, rep, value, runtime_ms
          );
        }
        records.push(RunRecord {
          config_index: cidx,
          config: cfg.clone(),
          cell: inst.cell.clone(),
          instance_id: inst.id.clone(),
          rep,
          seed,
          objective_value: value,
          runtime_ms,
        });
      }
    }
  }

  let mut ranking: Vec<ConfigSummary> = configs
    .iter()
    .enumerate()
    .map(|(cidx, cfg)| {
      let bucket: Vec<&RunRecord> = records.iter().filter(|r| r.config_index == cidx).collect();
      let values: Vec<f64> = bucket.iter().map(|r| r.objective_value).collect();
      let times: Vec<f64> = bucket.iter().map(|r| r.runtime_ms as f64).collect();
      ConfigSummary {
        config_index: cidx,
        config: cfg.clone(),
        mean_objective: mean(&values),
        stdev_objective: stdev(&values),
        mean_runtime_ms: mean(&times),
        runs: bucket.len(),
      }
    })
    .collect();
  ranking.sort_by(|a, b| a.mean_objective.partial_cmp(&b.mean_objective).unwrap());

  let run_dir = make_run_dir("tune-psa").expect("create run dir");
  let summary = build_summary(&objective, &cells, args.reps, &ranking);
  let json = ResultsJson {
    run_kind: "tune-psa",
    objective: format!("{:?}", objective),
    cells: cells.clone(),
    reps: args.reps,
    configs: configs.clone(),
    records,
    ranking,
  };
  write_run_outputs(&run_dir, &serde_json::to_value(&json).unwrap(), &summary)
    .expect("write outputs");
  println!("wrote {}", run_dir.display());
}

fn inst_id_hash(id: &str) -> u64 {
  let mut h: u64 = 1469598103934665603;
  for b in id.bytes() {
    h ^= b as u64;
    h = h.wrapping_mul(1099511628211);
  }
  h
}

fn build_summary(
  objective: &Objective,
  cells: &[String],
  reps: usize,
  ranking: &[ConfigSummary],
) -> String {
  let mut s = String::new();
  writeln!(s, "# tune_psa — {:?}", objective).ok();
  writeln!(s, "Cells: {}", cells.join(", ")).ok();
  writeln!(s, "Reps per (config, instance): {reps}").ok();
  writeln!(s).ok();
  writeln!(
    s,
    "| Rank | cfg# | initial_temp | cooling_rate | max_iterations | mean | stdev | mean runtime (ms) |"
  )
  .ok();
  writeln!(s, "|---|---|---|---|---|---|---|---|").ok();
  for (rank, r) in ranking.iter().enumerate() {
    writeln!(
      s,
      "| {} | {} | {} | {} | {} | {:.4} | {:.4} | {:.1} |",
      rank + 1,
      r.config_index,
      r.config.initial_temp,
      r.config.cooling_rate,
      r.config.max_iterations,
      r.mean_objective,
      r.stdev_objective,
      r.mean_runtime_ms
    )
    .ok();
  }
  s
}
