//! LNS driver: destroy (pluggable) -> greedy cheapest-insertion repair ->
//! simulated-annealing acceptance, with iteration and wall-time budgets.
//! Repair and acceptance minimise the context's objective.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use vrppd_core::{Objective, StopKind, WorkingSolution};

use crate::destroy::{DestroyOperator, Outcome};
use crate::solution::{Context, destroy, objective_name, repair_greedy, stop_lists};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LnsConfig {
  /// Iteration budget (None = unlimited; at least one budget must be set).
  pub max_iters: Option<u64>,
  /// Wall-time budget in seconds (None = unlimited).
  pub max_secs: Option<f64>,
  pub seed: u64,
  /// Number of orders removed per iteration. None = `k_frac` of the orders, or
  /// 15 % if that is None too (min 3).
  pub k: Option<usize>,
  /// Destroy size as a fraction of the orders (see [`k_for`]); used when `k`
  /// is None.
  #[serde(default)]
  pub k_frac: Option<f64>,
  /// SA: a solution `start_temp_control` (fraction) worse than the initial one
  /// is accepted with probability 0.5 at the start.
  pub start_temp_control: f64,
  /// SA: final temperature as a fraction of the start temperature.
  pub end_temp_ratio: f64,
  /// Log a curve row every this many iterations (in addition to every new best).
  pub log_every: u64,
  /// Also log a curve row whenever this many milliseconds passed since the last row (0 = off).
  pub log_every_ms: f64,
}

impl Default for LnsConfig {
  fn default() -> Self {
    Self {
      max_iters: Some(5000),
      max_secs: None,
      seed: 0,
      k: None,
      k_frac: None,
      start_temp_control: 0.05,
      end_temp_ratio: 0.002,
      log_every: 100,
      log_every_ms: 0.0,
    }
  }
}

/// Destroy size for a fraction of the orders: `max(3, round(frac * n))`,
/// capped at `n`.
pub fn k_for(n_orders: usize, frac: f64) -> usize {
  ((frac * n_orders as f64).round() as usize)
    .max(3)
    .min(n_orders.max(1))
}

pub fn default_k(n_orders: usize) -> usize {
  k_for(n_orders, 0.15)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurveRow {
  pub iteration: u64,
  pub elapsed_ms: f64,
  pub current: f64,
  pub best: f64,
  pub accepted: bool,
}

fn distance_name() -> String {
  objective_name(Objective::Distance).to_string()
}

fn not_recorded() -> f64 {
  f64::NAN
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LnsResult {
  pub operator: String,
  /// Objective minimised: "distance", "price" or "empty". `initial`, `best`,
  /// `final_current` and the curve are in this objective. Files written
  /// before the field existed are distance runs.
  #[serde(default = "distance_name")]
  pub objective: String,
  pub seed: u64,
  pub k: usize,
  pub initial: f64,
  pub best: f64,
  /// All three objectives of the best solution (NaN when read from a file
  /// written before these fields existed).
  #[serde(default = "not_recorded")]
  pub best_distance: f64,
  #[serde(default = "not_recorded")]
  pub best_price: f64,
  #[serde(default = "not_recorded")]
  pub best_empty: f64,
  pub final_current: f64,
  pub iterations: u64,
  pub elapsed_ms: f64,
  pub accepted: u64,
  pub improved_current: u64,
  pub new_best_count: u64,
  pub best_iteration: u64,
  pub best_elapsed_ms: f64,
  pub curve: Vec<CurveRow>,
  /// Per vehicle (index order): list of (order id, "pickup"/"delivery").
  pub best_routes: Vec<Vec<(u32, StopKind)>>,
  /// Mean destroy-time share (ms spent in destroy / total), for cost analysis.
  pub destroy_ms: f64,
  pub repair_ms: f64,
  pub valid: bool,
}

/// Hook called after each iteration with the current solution and its
/// objective value (used for snapshotting search states during label generation).
pub type IterHook<'h> = dyn FnMut(u64, &WorkingSolution, f64) + 'h;

pub fn run_lns(
  ctx: &Context,
  initial: WorkingSolution,
  op: &mut dyn DestroyOperator,
  cfg: &LnsConfig,
  mut hook: Option<&mut IterHook<'_>>,
) -> LnsResult {
  let mut rng = Xoshiro256StarStar::seed_from_u64(cfg.seed);
  let n = ctx.n_orders();
  let k = cfg
    .k
    .unwrap_or_else(|| k_for(n, cfg.k_frac.unwrap_or(0.15)))
    .min(n.max(1));

  let started = Instant::now();
  let mut current = initial;
  let mut current_obj = ctx.objective_value(&current);
  let mut best = current.clone();
  let mut best_obj = current_obj;
  let initial_obj = current_obj;

  let t0 = cfg.start_temp_control * initial_obj.max(1e-9) / std::f64::consts::LN_2;
  let t_end_ratio = cfg.end_temp_ratio.max(1e-12);

  let mut curve = vec![CurveRow {
    iteration: 0,
    elapsed_ms: 0.0,
    current: current_obj,
    best: best_obj,
    accepted: true,
  }];
  let mut accepted = 0u64;
  let mut improved_current = 0u64;
  let mut new_best_count = 0u64;
  let mut best_iteration = 0u64;
  let mut best_elapsed_ms = 0.0;
  let mut destroy_ns = 0u128;
  let mut repair_ns = 0u128;
  let mut iter = 0u64;
  let mut last_log_ms = 0.0_f64;

  loop {
    let elapsed = started.elapsed().as_secs_f64();
    let mut progress: f64 = 0.0;
    if let Some(mi) = cfg.max_iters {
      if iter >= mi {
        break;
      }
      progress = progress.max(iter as f64 / mi as f64);
    }
    if let Some(ms) = cfg.max_secs {
      if elapsed >= ms {
        break;
      }
      progress = progress.max(elapsed / ms);
    }
    if cfg.max_iters.is_none() && cfg.max_secs.is_none() {
      panic!("LnsConfig needs max_iters or max_secs");
    }
    iter += 1;
    let temperature = t0 * t_end_ratio.powf(progress.min(1.0));

    let td = Instant::now();
    let removed = op.select(ctx, &current, k, &mut rng);
    let mut candidate = current.clone();
    destroy(ctx, &mut candidate, &removed);
    destroy_ns += td.elapsed().as_nanos();

    let tr = Instant::now();
    // Reinsert in random order (classic LNS greedy repair).
    let mut order = removed;
    shuffle(&mut order, &mut rng);
    repair_greedy(ctx, &mut candidate, &order);
    repair_ns += tr.elapsed().as_nanos();

    let cand_obj = ctx.objective_value(&candidate);
    let delta = cand_obj - current_obj;
    let accept = delta < -1e-9 || {
      let p: f64 = rng.r#gen();
      p < (-delta / temperature).exp()
    };

    let outcome = if cand_obj < best_obj - 1e-9 {
      Outcome::NewBest
    } else if delta < -1e-9 {
      Outcome::Improved
    } else if accept {
      Outcome::AcceptedWorse
    } else {
      Outcome::Rejected
    };
    op.feedback(outcome);

    let mut is_new_best = false;
    if accept {
      accepted += 1;
      if delta < -1e-9 {
        improved_current += 1;
      }
      current = candidate;
      current_obj = cand_obj;
      if current_obj < best_obj - 1e-9 {
        best_obj = current_obj;
        best = current.clone();
        new_best_count += 1;
        best_iteration = iter;
        best_elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        is_new_best = true;
      }
    }

    let now_ms = started.elapsed().as_secs_f64() * 1000.0;
    if is_new_best
      || (cfg.log_every > 0 && iter.is_multiple_of(cfg.log_every))
      || (cfg.log_every_ms > 0.0 && now_ms - last_log_ms >= cfg.log_every_ms)
    {
      last_log_ms = now_ms;
      curve.push(CurveRow {
        iteration: iter,
        elapsed_ms: now_ms,
        current: current_obj,
        best: best_obj,
        accepted: accept,
      });
    }

    if let Some(h) = hook.as_deref_mut() {
      h(iter, &current, current_obj);
    }
  }

  let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
  let valid = best.is_valid(ctx.problem);
  LnsResult {
    operator: op.name().to_string(),
    objective: objective_name(ctx.objective).to_string(),
    seed: cfg.seed,
    k,
    initial: initial_obj,
    best: best_obj,
    best_distance: best.total_distance,
    best_price: best.total_price,
    best_empty: best.empty_distance,
    final_current: current_obj,
    iterations: iter,
    elapsed_ms,
    accepted,
    improved_current,
    new_best_count,
    best_iteration,
    best_elapsed_ms,
    curve,
    best_routes: stop_lists(ctx.problem, &best),
    destroy_ms: destroy_ns as f64 / 1e6,
    repair_ms: repair_ns as f64 / 1e6,
    valid,
  }
}

fn shuffle<T>(v: &mut [T], rng: &mut Xoshiro256StarStar) {
  for i in (1..v.len()).rev() {
    let j = rng.gen_range(0..=i);
    v.swap(i, j);
  }
}

/// Start solution: the thesis' RSCIM construction with the given seed (built
/// for distance whatever the context's objective, so every objective starts
/// from the same solution).
pub fn initial_solution(ctx: &Context, seed: u64) -> WorkingSolution {
  vrppd_cea::generate_rscim_seeded(
    ctx.problem,
    &ctx.order_mat,
    &ctx.vstart_mat,
    Objective::Distance,
    seed,
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::destroy::{OPERATOR_NAMES, build_operator};
  use std::collections::HashMap;
  use vrppd_core::{OrderMatrix, Problem, VehicleStartMatrix, WorkingStop};

  fn load_problem(cell: &str, idx: usize) -> Problem {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("../../problem-gen/problems")
      .join(cell);
    let mut files: Vec<_> = std::fs::read_dir(dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .map(|e| e.path())
      .collect();
    files.sort();
    serde_json::from_str(&std::fs::read_to_string(&files[idx]).unwrap()).unwrap()
  }

  #[test]
  fn classical_operators_improve_and_stay_valid() {
    let problem = load_problem("10_20", 0);
    let ctx = Context::new(&problem);
    for name in OPERATOR_NAMES
      .iter()
      .filter(|n| !matches!(**n, "neural" | "adaptive_n"))
    {
      let mut op = build_operator(name, None, 0.0, 3.0).unwrap();
      let cfg = LnsConfig {
        max_iters: Some(300),
        seed: 5,
        ..Default::default()
      };
      let init = initial_solution(&ctx, 5);
      let res = run_lns(&ctx, init, op.as_mut(), &cfg, None);
      assert!(res.valid, "{name}: invalid best");
      assert!(
        res.best <= res.initial + 1e-9,
        "{name}: best worse than initial"
      );
      assert!(
        res.best < res.initial,
        "{name}: no improvement at all in 300 iterations"
      );
      assert_eq!(res.iterations, 300);
      let n_stops: usize = res.best_routes.iter().map(|r| r.len()).sum();
      assert_eq!(n_stops, 2 * problem.orders.len());
    }
  }

  #[test]
  fn same_seed_same_result() {
    let problem = load_problem("10_10", 3);
    let ctx = Context::new(&problem);
    let run = || {
      let mut op = build_operator("adaptive", None, 0.0, 3.0).unwrap();
      let cfg = LnsConfig {
        max_iters: Some(200),
        seed: 11,
        ..Default::default()
      };
      run_lns(&ctx, initial_solution(&ctx, 11), op.as_mut(), &cfg, None).best
    };
    assert_eq!(run(), run());
  }

  /// Rebuild a solution from `best_routes` and evaluate it from scratch.
  fn recompute(problem: &Problem, routes: &[Vec<(u32, StopKind)>]) -> WorkingSolution {
    let idx: HashMap<u32, usize> = problem
      .orders
      .iter()
      .enumerate()
      .map(|(i, o)| (o.id, i))
      .collect();
    let mut sol = WorkingSolution::empty(problem.vehicles.len());
    for (v_idx, route) in routes.iter().enumerate() {
      sol.routes[v_idx].stops = route
        .iter()
        .map(|&(id, kind)| WorkingStop {
          order_idx: idx[&id],
          kind,
        })
        .collect();
    }
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);
    sol.recalculate_all(problem, &order_mat, &vstart_mat);
    sol
  }

  fn value_of(objective: Objective, sol: &WorkingSolution) -> f64 {
    match objective {
      Objective::Distance => sol.total_distance,
      Objective::Price => sol.total_price,
      Objective::Empty => sol.empty_distance,
    }
  }

  fn run_with(problem: &Problem, objective: Objective, iters: u64, seed: u64) -> LnsResult {
    let ctx = Context::with_objective(problem, objective);
    let mut op = build_operator("random", None, 0.0, 3.0).unwrap();
    let cfg = LnsConfig {
      max_iters: Some(iters),
      seed,
      ..Default::default()
    };
    run_lns(&ctx, initial_solution(&ctx, seed), op.as_mut(), &cfg, None)
  }

  /// Recorded before the objective-generic refactor (20_50 instance 0, random
  /// destroy, seed 1, 2 000 iterations) on macOS/aarch64.
  const RECORDED_DISTANCE_BEST: f64 = 9411.860251651286;

  #[test]
  fn distance_run_is_unchanged() {
    let problem = load_problem("20_50", 0);
    let ctx = Context::new(&problem);
    let mut op = build_operator("random", None, 0.0, 3.0).unwrap();
    let cfg = LnsConfig {
      max_iters: Some(2000),
      seed: 1,
      ..Default::default()
    };
    let res = run_lns(&ctx, initial_solution(&ctx, 1), op.as_mut(), &cfg, None);
    assert_eq!(res.objective, "distance");
    assert_eq!(res.best_distance, res.best);
    // The distance matrix comes from the platform libm (sin, cos, acos), which
    // may differ in the last bit on other platforms, so only the recording
    // platform demands bit equality.
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
      assert_eq!(res.best, RECORDED_DISTANCE_BEST);
    } else {
      assert!(
        (res.best - RECORDED_DISTANCE_BEST).abs() <= 1e-9 * RECORDED_DISTANCE_BEST,
        "best {} vs recorded {RECORDED_DISTANCE_BEST}",
        res.best
      );
    }
  }

  fn assert_tracks_recomputed_value(objective: Objective, name: &str) {
    let problem = load_problem("10_20", 0);
    let res = run_with(&problem, objective, 300, 5);
    assert_eq!(res.objective, name);
    assert!(res.valid, "{name}: invalid best");
    let sol = recompute(&problem, &res.best_routes);
    let value = value_of(objective, &sol);
    assert!(
      (res.best - value).abs() < 1e-6,
      "{name}: best {} vs recomputed {value}",
      res.best
    );
    assert!((res.best_distance - sol.total_distance).abs() < 1e-6);
    assert!((res.best_price - sol.total_price).abs() < 1e-6);
    assert!((res.best_empty - sol.empty_distance).abs() < 1e-6);
    let ctx = Context::new(&problem);
    let init = initial_solution(&ctx, 5);
    assert!((res.initial - value_of(objective, &init)).abs() < 1e-6);
    assert!(res.best < res.initial, "{name}: no improvement");
    assert_eq!(res.curve.last().unwrap().best, res.best);
  }

  #[test]
  fn price_objective_tracks_recomputed_value() {
    assert_tracks_recomputed_value(Objective::Price, "price");
  }

  #[test]
  fn empty_objective_tracks_recomputed_value() {
    assert_tracks_recomputed_value(Objective::Empty, "empty");
  }

  #[test]
  fn price_run_differs_from_distance_run() {
    let problem = load_problem("10_20", 0);
    let price = problem.vehicles[0].price_km;
    assert!(
      problem
        .vehicles
        .iter()
        .any(|v| (v.price_km - price).abs() > 1e-9),
      "fixture needs unequal vehicle prices"
    );
    let by_distance = run_with(&problem, Objective::Distance, 300, 5);
    let by_price = run_with(&problem, Objective::Price, 300, 5);
    assert_ne!(by_distance.best_routes, by_price.best_routes);
  }

  #[test]
  fn result_without_objective_fields_still_parses() {
    let problem = load_problem("10_10", 0);
    let mut json = serde_json::to_value(run_with(&problem, Objective::Price, 10, 1)).unwrap();
    let obj = json.as_object_mut().unwrap();
    for key in ["objective", "best_distance", "best_price", "best_empty"] {
      obj.remove(key).unwrap();
    }
    let old: LnsResult = serde_json::from_value(json).unwrap();
    assert_eq!(old.objective, "distance");
    assert!(old.best_distance.is_nan() && old.best_price.is_nan() && old.best_empty.is_nan());
  }

  #[test]
  fn k_for_examples() {
    assert_eq!(k_for(20, 0.25), 5);
    assert_eq!(k_for(10, 0.05), 3);
    assert_eq!(k_for(50, 0.15), default_k(50));
    assert_eq!(k_for(2, 0.5), 2);
  }

  #[test]
  fn k_frac_sets_destroy_size_and_k_wins() {
    let problem = load_problem("10_20", 0);
    let ctx = Context::new(&problem);
    let run = |k: Option<usize>, k_frac: Option<f64>| {
      let mut op = build_operator("random", None, 0.0, 3.0).unwrap();
      let cfg = LnsConfig {
        max_iters: Some(5),
        k,
        k_frac,
        ..Default::default()
      };
      run_lns(&ctx, initial_solution(&ctx, 1), op.as_mut(), &cfg, None).k
    };
    assert_eq!(run(None, Some(0.5)), 10);
    assert_eq!(run(Some(4), Some(0.5)), 4);
    assert_eq!(run(None, None), default_k(20));
  }

  #[test]
  fn time_budget_stops() {
    let problem = load_problem("20_50", 0);
    let ctx = Context::new(&problem);
    let mut op = build_operator("random", None, 0.0, 3.0).unwrap();
    let cfg = LnsConfig {
      max_iters: None,
      max_secs: Some(0.2),
      seed: 1,
      ..Default::default()
    };
    let res = run_lns(&ctx, initial_solution(&ctx, 1), op.as_mut(), &cfg, None);
    assert!(res.elapsed_ms >= 200.0 && res.elapsed_ms < 1000.0);
    assert!(res.iterations > 10);
  }
}
