//! LNS driver: destroy (pluggable) -> greedy cheapest-insertion repair ->
//! simulated-annealing acceptance, with iteration and wall-time budgets.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use vrppd_core::{Objective, StopKind, WorkingSolution};

use crate::destroy::{DestroyOperator, Outcome};
use crate::solution::{Context, destroy, repair_greedy, stop_lists};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LnsConfig {
  /// Iteration budget (None = unlimited; at least one budget must be set).
  pub max_iters: Option<u64>,
  /// Wall-time budget in seconds (None = unlimited).
  pub max_secs: Option<f64>,
  pub seed: u64,
  /// Number of orders removed per iteration. None = 15 % of orders, min 3.
  pub k: Option<usize>,
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
      start_temp_control: 0.05,
      end_temp_ratio: 0.002,
      log_every: 100,
      log_every_ms: 0.0,
    }
  }
}

pub fn default_k(n_orders: usize) -> usize {
  ((0.15 * n_orders as f64).round() as usize)
    .max(3)
    .min(n_orders.max(1))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurveRow {
  pub iteration: u64,
  pub elapsed_ms: f64,
  pub current: f64,
  pub best: f64,
  pub accepted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LnsResult {
  pub operator: String,
  pub seed: u64,
  pub k: usize,
  pub initial: f64,
  pub best: f64,
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

/// Hook called after each iteration with the current solution (used for
/// snapshotting search states during label generation).
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
  let k = cfg.k.unwrap_or_else(|| default_k(n)).min(n.max(1));

  let started = Instant::now();
  let mut current = initial;
  let mut current_obj = current.total_distance;
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

    let cand_obj = candidate.total_distance;
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
    seed: cfg.seed,
    k,
    initial: initial_obj,
    best: best_obj,
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

/// Start solution: the thesis' RSCIM construction with the given seed.
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
  use vrppd_core::Problem;

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
    for name in OPERATOR_NAMES.iter().filter(|n| **n != "neural") {
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
