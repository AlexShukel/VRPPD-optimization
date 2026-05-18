//! Large-instance benchmark harness. Runs vrppd-milp, vrppd-psa, vrppd-cea, and vrppd-or-tools::routing (PCA and PCI variants) across the configured cells. MILP runs once per (cell, instance, objective); stochastic solvers (psa, cea, routing) get `--reps` seeded runs. Writes per-run records to results.json and an aggregated summary.md.

use std::collections::HashMap;
use std::fmt::Write;
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;
use vrppd_benchmarks::{
  load_cell_instances, make_run_dir, mean, parse_cells, stdev, write_run_outputs, Instance,
};
use vrppd_cea::{solve_cea_seeded, CeaConfig};
use vrppd_core::Objective;
use vrppd_milp::{solve_milp, MilpError, MilpStatus};
use vrppd_or_tools::{solve_routing_with_strategy, OrToolsError, OrToolsStatus, RoutingStrategy};
use vrppd_psa::{default_config_for, solve_seeded as solve_psa_seeded, SaConfig};

const ALL_ALGOS: &[&str] = &["milp", "psa", "cea", "routing-pca", "routing-pci"];

#[derive(Parser, Debug)]
#[command(
  name = "bench_large",
  about = "Run all algorithms on the large-instance benchmark suite."
)]
struct Args {
  #[arg(long, default_value = "10_10,10_20,20_50,30_100,50_200,100_500")]
  cells: String,

  #[arg(long, default_value_t = 5)]
  reps: usize,

  #[arg(long, default_value = "distance,price,empty")]
  objectives: String,

  #[arg(long, default_value = "milp,psa,cea,routing-pca,routing-pci")]
  algos: String,

  #[arg(long, default_value_t = 1800)]
  milp_timeout_secs: u64,

  #[arg(long, default_value_t = 60)]
  cea_timeout_secs: u64,

  #[arg(long, default_value_t = 60)]
  routing_timeout_secs: u64,

  #[arg(long)]
  threads: Option<usize>,

  #[arg(long, default_value_t = 20)]
  max_instances_per_cell: usize,

  #[arg(long, default_value_t = false)]
  debug: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RunRecord {
  cell: String,
  algo: String,
  objective: String,
  instance_id: String,
  rep: usize,
  seed: Option<u64>,
  value: Option<f64>,
  runtime_ms: u128,
  status: String,
  error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AlgoSummary {
  cell: String,
  objective: String,
  algo: String,
  runs: usize,
  successes: usize,
  mean_value: Option<f64>,
  stdev_value: Option<f64>,
  best_value: Option<f64>,
  mean_runtime_ms: f64,
}

#[derive(Serialize)]
struct ResultsJson {
  run_kind: &'static str,
  cells: Vec<String>,
  reps: usize,
  objectives: Vec<String>,
  algos: Vec<String>,
  budgets: BudgetReport,
  records: Vec<RunRecord>,
  summaries: Vec<AlgoSummary>,
}

#[derive(Serialize)]
struct BudgetReport {
  milp_timeout_secs: u64,
  cea_timeout_secs: u64,
  routing_timeout_secs: u64,
  threads: usize,
}

fn main() {
  let args = Args::parse();
  let cells = parse_cells(&args.cells);
  let objectives: Vec<Objective> = args
    .objectives
    .split(',')
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(|s| {
      vrppd_benchmarks::parse_objective(s).unwrap_or_else(|e| {
        eprintln!("--objectives: {e}");
        std::process::exit(2);
      })
    })
    .collect();
  let algos: Vec<String> = args
    .algos
    .split(',')
    .map(|s| s.trim().to_string())
    .filter(|s| !s.is_empty())
    .collect();
  for a in &algos {
    if !ALL_ALGOS.contains(&a.as_str()) {
      eprintln!("--algos: unknown {a:?}, valid: {:?}", ALL_ALGOS);
      std::process::exit(2);
    }
  }
  let threads = args.threads.unwrap_or_else(|| {
    std::thread::available_parallelism()
      .map(|n| n.get())
      .unwrap_or(1)
  });

  let mut all_instances: Vec<Instance> = Vec::new();
  for cell in &cells {
    let mut loaded = load_cell_instances(cell).unwrap_or_else(|e| {
      eprintln!("failed to load cell {cell}: {e}");
      std::process::exit(1);
    });
    loaded.truncate(args.max_instances_per_cell);
    all_instances.append(&mut loaded);
  }
  eprintln!(
    "bench_large: {} cells, {} instances total, algos={:?}, objectives={:?}, reps={}",
    cells.len(),
    all_instances.len(),
    algos,
    objectives.iter().map(|o| format!("{o:?}")).collect::<Vec<_>>(),
    args.reps,
  );

  let milp_timeout = Duration::from_secs(args.milp_timeout_secs);
  let cea_timeout = Duration::from_secs(args.cea_timeout_secs);
  let routing_timeout = Duration::from_secs(args.routing_timeout_secs);

  let mut records: Vec<RunRecord> = Vec::new();

  for inst in &all_instances {
    let cell_n = cell_n_orders(&inst.cell);

    for &obj in &objectives {
      if algos.iter().any(|a| a == "milp") {
        for rec in run_milp(inst, obj, cell_n, milp_timeout, threads) {
          debug_record(args.debug, &rec);
          records.push(rec);
        }
      }
      if algos.iter().any(|a| a == "psa") {
        for rep in 0..args.reps {
          let rec = run_psa(inst, obj, rep, threads);
          debug_record(args.debug, &rec);
          records.push(rec);
        }
      }
      if algos.iter().any(|a| a == "cea") {
        for rep in 0..args.reps {
          let rec = run_cea(inst, obj, rep, cea_timeout, threads);
          debug_record(args.debug, &rec);
          records.push(rec);
        }
      }
      if algos.iter().any(|a| a == "routing-pci") {
        for rep in 0..args.reps {
          let rec = run_routing(
            inst,
            obj,
            rep,
            routing_timeout,
            threads,
            RoutingStrategy::Pci,
            "routing-pci",
          );
          debug_record(args.debug, &rec);
          records.push(rec);
        }
      }
      if algos.iter().any(|a| a == "routing-pca") {
        for rep in 0..args.reps {
          let rec = run_routing(
            inst,
            obj,
            rep,
            routing_timeout,
            threads,
            RoutingStrategy::Pca,
            "routing-pca",
          );
          debug_record(args.debug, &rec);
          records.push(rec);
        }
      }
    }
  }

  let summaries = aggregate(&records);
  let run_dir = make_run_dir("bench-large").expect("create run dir");
  let summary_md = build_summary(&cells, &objectives, &algos, &summaries);
  let json = ResultsJson {
    run_kind: "bench-large",
    cells: cells.clone(),
    reps: args.reps,
    objectives: objectives.iter().map(|o| format!("{o:?}")).collect(),
    algos: algos.clone(),
    budgets: BudgetReport {
      milp_timeout_secs: args.milp_timeout_secs,
      cea_timeout_secs: args.cea_timeout_secs,
      routing_timeout_secs: args.routing_timeout_secs,
      threads,
    },
    records,
    summaries,
  };
  write_run_outputs(&run_dir, &serde_json::to_value(&json).unwrap(), &summary_md)
    .expect("write outputs");
  println!("wrote {}", run_dir.display());
}

fn debug_record(enabled: bool, rec: &RunRecord) {
  if !enabled {
    return;
  }
  let value = rec
    .value
    .map(|v| format!("{v:.4}"))
    .unwrap_or_else(|| "—".to_string());
  let extra = rec
    .error
    .as_ref()
    .map(|e| format!(" err={e}"))
    .unwrap_or_default();
  eprintln!(
    "[debug] cell={} algo={} obj={} id={} rep={} value={} status={} time={}ms{}",
    rec.cell, rec.algo, rec.objective, rec.instance_id, rec.rep, value, rec.status, rec.runtime_ms, extra,
  );
}

fn cell_n_orders(cell: &str) -> usize {
  cell
    .split('_')
    .nth(1)
    .and_then(|s| s.parse::<usize>().ok())
    .unwrap_or(usize::MAX)
}

fn obj_label(o: Objective) -> &'static str {
  match o {
    Objective::Distance => "distance",
    Objective::Price => "price",
    Objective::Empty => "empty",
  }
}

fn inst_id_hash(id: &str) -> u64 {
  let mut h: u64 = 1469598103934665603;
  for b in id.bytes() {
    h ^= b as u64;
    h = h.wrapping_mul(1099511628211);
  }
  h
}

fn skip(
  inst: &Instance,
  algo: &str,
  obj: Objective,
  rep: usize,
  reason: &str,
  seed: Option<u64>,
) -> RunRecord {
  RunRecord {
    cell: inst.cell.clone(),
    algo: algo.to_string(),
    objective: obj_label(obj).to_string(),
    instance_id: inst.id.clone(),
    rep,
    seed,
    value: None,
    runtime_ms: 0,
    status: "Skipped".to_string(),
    error: Some(reason.to_string()),
  }
}

fn run_milp(
  inst: &Instance,
  obj: Objective,
  n_orders: usize,
  timeout: Duration,
  threads: usize,
) -> Vec<RunRecord> {
  if matches!(obj, Objective::Empty) {
    return vec![skip(inst, "milp", obj, 0, "Empty unsupported", None)];
  }
  if n_orders > 200 {
    return vec![skip(
      inst,
      "milp",
      obj,
      0,
      "N > 200: HiGHS OOM expected at 54GB",
      None,
    )];
  }
  let started = Instant::now();
  let res = solve_milp(&inst.problem, obj, timeout, threads);
  let runtime_ms = started.elapsed().as_millis();
  let rec = match res {
    Ok(r) => RunRecord {
      cell: inst.cell.clone(),
      algo: "milp".to_string(),
      objective: obj_label(obj).to_string(),
      instance_id: inst.id.clone(),
      rep: 0,
      seed: None,
      value: Some(r.objective_value),
      runtime_ms,
      status: match r.status {
        MilpStatus::Optimal => "Optimal".to_string(),
        MilpStatus::TimedOut => "TimedOut".to_string(),
      },
      error: None,
    },
    Err(MilpError::UnsupportedObjective(_)) => {
      skip(inst, "milp", obj, 0, "Empty unsupported", None)
    }
    Err(e) => RunRecord {
      cell: inst.cell.clone(),
      algo: "milp".to_string(),
      objective: obj_label(obj).to_string(),
      instance_id: inst.id.clone(),
      rep: 0,
      seed: None,
      value: None,
      runtime_ms,
      status: "Failed".to_string(),
      error: Some(format!("{e}")),
    },
  };
  vec![rec]
}

fn run_psa(inst: &Instance, obj: Objective, rep: usize, threads: usize) -> RunRecord {
  let seed = (rep as u64).wrapping_mul(1_000_003) ^ inst_id_hash(&inst.id);
  let mut cfg: SaConfig = default_config_for(obj);
  cfg.threads = threads.max(1);
  let started = Instant::now();
  let solved = solve_psa_seeded(&inst.problem, obj, cfg, seed);
  let runtime_ms = started.elapsed().as_millis();
  let value = obj.energy(&solved.solution);
  RunRecord {
    cell: inst.cell.clone(),
    algo: "psa".to_string(),
    objective: obj_label(obj).to_string(),
    instance_id: inst.id.clone(),
    rep,
    seed: Some(seed),
    value: Some(value),
    runtime_ms,
    status: "Completed".to_string(),
    error: None,
  }
}

fn run_cea(
  inst: &Instance,
  obj: Objective,
  rep: usize,
  timeout: Duration,
  threads: usize,
) -> RunRecord {
  let seed = (rep as u64).wrapping_mul(1_000_003) ^ inst_id_hash(&inst.id) ^ 0xC0E0_C0E0;
  let cfg = CeaConfig {
    wall_time_cap_ms: Some(timeout.as_millis() as u64),
    threads: threads.max(1),
    ..CeaConfig::default()
  };
  let started = Instant::now();
  let solved = solve_cea_seeded(&inst.problem, obj, cfg, seed);
  let runtime_ms = started.elapsed().as_millis();
  let value = obj.energy(&solved.solution);
  RunRecord {
    cell: inst.cell.clone(),
    algo: "cea".to_string(),
    objective: obj_label(obj).to_string(),
    instance_id: inst.id.clone(),
    rep,
    seed: Some(seed),
    value: Some(value),
    runtime_ms,
    status: format!("{:?}", solved.exit_reason),
    error: None,
  }
}

fn run_routing(
  inst: &Instance,
  obj: Objective,
  rep: usize,
  timeout: Duration,
  threads: usize,
  strategy: RoutingStrategy,
  algo_label: &str,
) -> RunRecord {
  if matches!(obj, Objective::Empty) {
    return skip(inst, algo_label, obj, rep, "Empty unsupported", None);
  }
  let started = Instant::now();
  let res = solve_routing_with_strategy(&inst.problem, obj, timeout, threads, strategy);
  let runtime_ms = started.elapsed().as_millis();
  match res {
    Ok(r) => RunRecord {
      cell: inst.cell.clone(),
      algo: algo_label.to_string(),
      objective: obj_label(obj).to_string(),
      instance_id: inst.id.clone(),
      rep,
      seed: None,
      value: Some(r.objective_value),
      runtime_ms,
      status: match r.status {
        OrToolsStatus::Optimal => "Optimal".to_string(),
        OrToolsStatus::Feasible => "Feasible".to_string(),
        OrToolsStatus::TimedOut => "TimedOut".to_string(),
      },
      error: None,
    },
    Err(OrToolsError::UnsupportedObjective(_)) => {
      skip(inst, algo_label, obj, rep, "Empty unsupported", None)
    }
    Err(e) => RunRecord {
      cell: inst.cell.clone(),
      algo: algo_label.to_string(),
      objective: obj_label(obj).to_string(),
      instance_id: inst.id.clone(),
      rep,
      seed: None,
      value: None,
      runtime_ms,
      status: "Failed".to_string(),
      error: Some(format!("{e}")),
    },
  }
}

fn aggregate(records: &[RunRecord]) -> Vec<AlgoSummary> {
  let mut buckets: HashMap<(String, String, String), Vec<&RunRecord>> = HashMap::new();
  for r in records {
    buckets
      .entry((r.cell.clone(), r.objective.clone(), r.algo.clone()))
      .or_default()
      .push(r);
  }

  let mut summaries: Vec<AlgoSummary> = buckets
    .into_iter()
    .map(|((cell, obj, algo), recs)| {
      let values: Vec<f64> = recs.iter().filter_map(|r| r.value).collect();
      let times: Vec<f64> = recs.iter().map(|r| r.runtime_ms as f64).collect();
      let mean_value = if values.is_empty() {
        None
      } else {
        Some(mean(&values))
      };
      let stdev_value = if values.len() < 2 {
        None
      } else {
        Some(stdev(&values))
      };
      let best_value = if values.is_empty() {
        None
      } else {
        Some(values.iter().cloned().fold(f64::INFINITY, f64::min))
      };
      AlgoSummary {
        cell,
        objective: obj,
        algo,
        runs: recs.len(),
        successes: values.len(),
        mean_value,
        stdev_value,
        best_value,
        mean_runtime_ms: mean(&times),
      }
    })
    .collect();
  summaries.sort_by(|a, b| {
    a.cell
      .cmp(&b.cell)
      .then(a.objective.cmp(&b.objective))
      .then(a.algo.cmp(&b.algo))
  });
  summaries
}

fn build_summary(
  cells: &[String],
  objectives: &[Objective],
  algos: &[String],
  summaries: &[AlgoSummary],
) -> String {
  let mut s = String::new();
  writeln!(s, "# bench_large").ok();
  writeln!(s, "Cells: {}", cells.join(", ")).ok();
  writeln!(
    s,
    "Objectives: {}",
    objectives
      .iter()
      .map(|o| obj_label(*o))
      .collect::<Vec<_>>()
      .join(", ")
  )
  .ok();
  writeln!(s, "Algos: {}", algos.join(", ")).ok();
  writeln!(s).ok();

  for cell in cells {
    for obj in objectives {
      let header = format!("## Cell {cell} — objective {}", obj_label(*obj));
      writeln!(s, "{header}\n").ok();
      writeln!(
        s,
        "| algo | runs | ok | mean | stdev | best | mean runtime (ms) |"
      )
      .ok();
      writeln!(s, "|---|---|---|---|---|---|---|").ok();
      for algo in algos {
        if let Some(row) = summaries.iter().find(|r| {
          r.cell == *cell && r.objective == obj_label(*obj) && r.algo == *algo
        }) {
          writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {:.1} |",
            row.algo,
            row.runs,
            row.successes,
            fmt_opt(row.mean_value),
            fmt_opt(row.stdev_value),
            fmt_opt(row.best_value),
            row.mean_runtime_ms,
          )
          .ok();
        }
      }
      writeln!(s).ok();
    }
  }
  s
}

fn fmt_opt(v: Option<f64>) -> String {
  match v {
    Some(x) => format!("{x:.4}"),
    None => "—".to_string(),
  }
}
