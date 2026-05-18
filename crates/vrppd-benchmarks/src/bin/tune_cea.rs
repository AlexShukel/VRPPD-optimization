//! CEA grid-search tuner. Sweeps `population_size × p_reinsertion × p_crossover` over a list of cells, repeats each (config, instance) several times with distinct seeds, ranks configs by mean objective.

use std::fmt::Write;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;
use vrppd_benchmarks::{
  load_cell_instances, make_run_dir, mean, parse_cells, parse_objective, stdev, write_run_outputs,
};
use vrppd_cea::{solve_cea_seeded, CeaConfig};
use vrppd_core::Objective;

#[derive(Parser, Debug)]
#[command(name = "tune_cea", about = "Grid search over CEA hyperparameters.")]
struct Args {
  #[arg(long, value_name = "CELL_LIST")]
  cells: String,

  #[arg(long, default_value_t = 3)]
  reps: usize,

  #[arg(long, default_value = "distance")]
  objective: String,

  #[arg(long, value_delimiter = ',', num_args = 1..)]
  population_size: Option<Vec<usize>>,

  #[arg(long, value_delimiter = ',', num_args = 1..)]
  p_reinsertion: Option<Vec<f64>>,

  #[arg(long, value_delimiter = ',', num_args = 1..)]
  p_crossover: Option<Vec<f64>>,

  #[arg(long, default_value_t = 200)]
  conv_count: usize,

  #[arg(long, default_value_t = 60_000)]
  wall_time_cap_ms: u64,

  #[arg(long, default_value_t = 1)]
  threads: usize,

  #[arg(long, default_value_t = false)]
  debug: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigSpec {
  population_size: usize,
  p_reinsertion: f64,
  p_crossover: f64,
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
  generations: u64,
}

#[derive(Clone, Debug, Serialize)]
struct ConfigSummary {
  config_index: usize,
  config: ConfigSpec,
  mean_objective: f64,
  stdev_objective: f64,
  mean_runtime_ms: f64,
  mean_generations: f64,
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

  let pops = args.population_size.unwrap_or_else(|| vec![30, 50, 100]);
  let p_reins = args.p_reinsertion.unwrap_or_else(|| vec![0.3, 0.5, 0.7]);
  let p_cross = args.p_crossover.unwrap_or_else(|| vec![0.3, 0.5, 0.7]);

  let mut configs: Vec<ConfigSpec> = Vec::new();
  for &p in &pops {
    for &pr in &p_reins {
      for &pc in &p_cross {
        configs.push(ConfigSpec {
          population_size: p,
          p_reinsertion: pr,
          p_crossover: pc,
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
    "tune_cea: {} configs × {} instances × {} reps = {} runs",
    configs.len(),
    instances.len(),
    args.reps,
    configs.len() * instances.len() * args.reps,
  );

  let mut records: Vec<RunRecord> = Vec::new();

  for (cidx, cfg) in configs.iter().enumerate() {
    let cea_config = CeaConfig {
      population_size: cfg.population_size,
      conv_count: args.conv_count,
      wall_time_cap_ms: Some(args.wall_time_cap_ms),
      recombination_fraction_low: 0.1,
      recombination_fraction_high: 0.5,
      p_reinsertion: cfg.p_reinsertion,
      p_crossover: cfg.p_crossover,
      threads: args.threads.max(1),
    };

    for inst in &instances {
      for rep in 0..args.reps {
        let seed = (cidx as u64) * 1_000_003 + (rep as u64) * 7919 + inst_id_hash(&inst.id);
        let started = Instant::now();
        let solved = solve_cea_seeded(&inst.problem, objective, cea_config, seed);
        let runtime_ms = started.elapsed().as_millis();
        let value = objective.energy(&solved.solution);
        if args.debug {
          eprintln!(
            "[debug] cfg#{cidx:02} cell={} id={} rep={} value={:.4} gens={} time={}ms",
            inst.cell, inst.id, rep, value, solved.generations, runtime_ms
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
          generations: solved.generations,
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
      let gens: Vec<f64> = bucket.iter().map(|r| r.generations as f64).collect();
      ConfigSummary {
        config_index: cidx,
        config: cfg.clone(),
        mean_objective: mean(&values),
        stdev_objective: stdev(&values),
        mean_runtime_ms: mean(&times),
        mean_generations: mean(&gens),
        runs: bucket.len(),
      }
    })
    .collect();
  ranking.sort_by(|a, b| a.mean_objective.partial_cmp(&b.mean_objective).unwrap());

  let run_dir = make_run_dir("tune-cea").expect("create run dir");
  let summary = build_summary(&objective, &cells, args.reps, &ranking);
  let json = ResultsJson {
    run_kind: "tune-cea",
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
  writeln!(s, "# tune_cea — {:?}", objective).ok();
  writeln!(s, "Cells: {}", cells.join(", ")).ok();
  writeln!(s, "Reps per (config, instance): {reps}").ok();
  writeln!(s).ok();
  writeln!(
    s,
    "| Rank | cfg# | pop_size | p_reinsertion | p_crossover | mean | stdev | mean gens | mean runtime (ms) |"
  )
  .ok();
  writeln!(s, "|---|---|---|---|---|---|---|---|---|").ok();
  for (rank, r) in ranking.iter().enumerate() {
    writeln!(
      s,
      "| {} | {} | {} | {} | {} | {:.4} | {:.4} | {:.1} | {:.1} |",
      rank + 1,
      r.config_index,
      r.config.population_size,
      r.config.p_reinsertion,
      r.config.p_crossover,
      r.mean_objective,
      r.stdev_objective,
      r.mean_generations,
      r.mean_runtime_ms
    )
    .ok();
  }
  s
}
