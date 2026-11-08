//! LNS experiment CLI: run one destroy operator on every instance of a cell
//! with `--repeats` seeds, writing a JSON result and a CSV search curve per run.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use vrppd_lns::io::{cell_name, init_threads, load_instances, resolve_cell_dir};
use vrppd_lns::{Context, LnsConfig, build_operator, initial_solution, run_lns};

#[derive(Parser, Debug)]
#[command(
  name = "lns",
  about = "Large Neighbourhood Search runs for the VRPPD (distance objective)."
)]
struct Args {
  /// Thesis cell name (e.g. 10_20) or a directory of instance JSON files.
  #[arg(long)]
  cell: String,
  /// Destroy operator: random | worst | related | adaptive | neural.
  #[arg(long, default_value = "random")]
  operator: String,
  /// Iteration budget (0 = none).
  #[arg(long, default_value_t = 5000)]
  iters: u64,
  /// Wall-time budget in seconds (0 = none).
  #[arg(long, default_value_t = 0.0)]
  seconds: f64,
  /// Base seed; run r on instance i uses seed = seed + 1000*i + r.
  #[arg(long, default_value_t = 1)]
  seed: u64,
  /// Destroy size (0 = 15 % of orders, min 3).
  #[arg(long, default_value_t = 0)]
  k: usize,
  #[arg(long, default_value_t = 1)]
  repeats: usize,
  /// Output directory (files: <out>/<instance>_s<seed>.json and .curve.csv).
  #[arg(long)]
  out: PathBuf,
  /// MLP weights JSON (neural operator).
  #[arg(long)]
  weights: Option<PathBuf>,
  /// Softmax temperature for neural sampling (0 = deterministic top-k).
  #[arg(long, default_value_t = 0.0)]
  tau: f64,
  /// Randomisation exponent of worst removal (Ropke & Pisinger p).
  #[arg(long, default_value_t = 3.0)]
  worst_p: f64,
  /// Log a curve row every N iterations (plus every new best; 0 = off).
  #[arg(long, default_value_t = 100)]
  log_every: u64,
  /// Log a curve row every N milliseconds (0 = off).
  #[arg(long, default_value_t = 0.0)]
  log_every_ms: f64,
  /// Only the first N instances of the cell.
  #[arg(long)]
  max_instances: Option<usize>,
  /// Worker threads (hard-capped at 6).
  #[arg(long, default_value_t = 6)]
  threads: usize,
  /// Re-run even if the result file already exists (default: skip existing).
  #[arg(long, default_value_t = false)]
  force: bool,
}

#[derive(Serialize)]
struct RunFile<'a> {
  cell: String,
  instance: String,
  n_orders: usize,
  n_vehicles: usize,
  budget_iters: Option<u64>,
  budget_secs: Option<f64>,
  tau: f64,
  #[serde(flatten)]
  result: &'a vrppd_lns::LnsResult,
}

fn main() {
  let args = Args::parse();
  let threads = init_threads(args.threads, 6);
  let dir = resolve_cell_dir(&args.cell);
  let cell = cell_name(&dir);
  let instances = load_instances(&dir, args.max_instances).unwrap_or_else(|e| {
    eprintln!("{e}");
    std::process::exit(1)
  });
  fs::create_dir_all(&args.out).expect("create out dir");
  // Validate operator construction early (weights path etc.).
  build_operator(
    &args.operator,
    args.weights.as_deref(),
    args.tau,
    args.worst_p,
  )
  .unwrap_or_else(|e| {
    eprintln!("{e}");
    std::process::exit(2)
  });

  let cfg_base = LnsConfig {
    max_iters: if args.iters > 0 {
      Some(args.iters)
    } else {
      None
    },
    max_secs: if args.seconds > 0.0 {
      Some(args.seconds)
    } else {
      None
    },
    k: if args.k > 0 { Some(args.k) } else { None },
    log_every: args.log_every,
    log_every_ms: args.log_every_ms,
    ..Default::default()
  };
  if cfg_base.max_iters.is_none() && cfg_base.max_secs.is_none() {
    eprintln!("need --iters or --seconds");
    std::process::exit(2);
  }

  let jobs: Vec<(usize, usize)> = (0..instances.len())
    .flat_map(|i| (0..args.repeats).map(move |r| (i, r)))
    .collect();
  eprintln!(
    "lns: cell={cell} operator={} instances={} repeats={} jobs={} threads={threads} iters={:?} secs={:?}",
    args.operator,
    instances.len(),
    args.repeats,
    jobs.len(),
    cfg_base.max_iters,
    cfg_base.max_secs
  );
  let started = Instant::now();
  let done: usize = jobs
    .par_iter()
    .map(|&(i, r)| {
      let inst = &instances[i];
      let seed = args.seed + 1000 * i as u64 + r as u64;
      let stem = format!("{}_s{}", inst.id, seed);
      let json_path = args.out.join(format!("{stem}.json"));
      if !args.force && json_path.exists() {
        return 0;
      }
      let ctx = Context::new(&inst.problem);
      let mut op = build_operator(
        &args.operator,
        args.weights.as_deref(),
        args.tau,
        args.worst_p,
      )
      .unwrap();
      let cfg = LnsConfig {
        seed,
        ..cfg_base.clone()
      };
      let init = initial_solution(&ctx, seed);
      let res = run_lns(&ctx, init, op.as_mut(), &cfg, None);
      let mut csv = String::from("iteration,elapsed_ms,current,best,accepted\n");
      for row in &res.curve {
        csv.push_str(&format!(
          "{},{:.3},{:.6},{:.6},{}\n",
          row.iteration, row.elapsed_ms, row.current, row.best, row.accepted as u8
        ));
      }
      fs::write(args.out.join(format!("{stem}.curve.csv")), csv).expect("write curve");
      let file = RunFile {
        cell: cell.clone(),
        instance: inst.id.clone(),
        n_orders: inst.problem.orders.len(),
        n_vehicles: inst.problem.vehicles.len(),
        budget_iters: cfg.max_iters,
        budget_secs: cfg.max_secs,
        tau: args.tau,
        result: &res,
      };
      let tmp = args.out.join(format!("{stem}.json.tmp"));
      fs::write(&tmp, serde_json::to_string(&file).unwrap()).expect("write json");
      fs::rename(&tmp, &json_path).expect("rename");
      1
    })
    .sum();
  eprintln!(
    "lns: {} runs written ({} skipped) in {:.1}s -> {}",
    done,
    jobs.len() - done,
    started.elapsed().as_secs_f64(),
    args.out.display()
  );
}
