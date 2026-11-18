//! Supervised label generation: snapshot search states from a random-removal
//! LNS run and score every order by the mean improvement of random removal
//! sets that contained it.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};
use vrppd_core::{StopKind, WorkingSolution};

use crate::destroy::RandomDestroy;
use crate::features::compute_features;
use crate::lns::{LnsConfig, default_k, initial_solution, run_lns};
use crate::solution::{Context, destroy, repair_greedy, stop_lists};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelConfig {
  /// Iterations of the random-removal LNS that produces the states.
  pub lns_iters: u64,
  /// Number of states to snapshot per instance (spread evenly over the run,
  /// including the final iteration).
  pub states_per_instance: usize,
  /// Random removal sets tried per state.
  pub m: usize,
  /// Removal size (None = 15 %, min 3).
  pub k: Option<usize>,
  pub seed: u64,
  /// Snapshot schedule: "log" (geometric spacing from iteration 1 to lns_iters,
  /// so that early states with many improving removals and late, converged
  /// states are both covered) or "even" (lns_iters * i / S).
  pub schedule: String,
  /// Fraction of the M tries counted as the top set for `top_share`.
  pub top_frac: f64,
}

impl Default for LabelConfig {
  fn default() -> Self {
    Self {
      lns_iters: 2000,
      states_per_instance: 10,
      m: 100,
      k: None,
      seed: 0,
      schedule: "log".to_string(),
      top_frac: 0.1,
    }
  }
}

/// Iterations at which states are snapshotted (sorted, deduplicated, last = lns_iters).
pub fn snapshot_iterations(lns_iters: u64, states: usize, schedule: &str) -> Vec<u64> {
  let s = states.max(1) as u64;
  let mut v: Vec<u64> = if schedule == "even" {
    (1..=s).map(|i| lns_iters * i / s).collect()
  } else {
    (1..=s)
      .map(|i| ((lns_iters as f64).powf(i as f64 / s as f64)).round() as u64)
      .collect()
  };
  v.iter_mut()
    .for_each(|x| *x = (*x).clamp(1, lns_iters.max(1)));
  v.sort_unstable();
  v.dedup();
  v
}

/// One labelled search state (one JSONL line).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LabelledState {
  pub instance: String,
  pub cell: String,
  pub split: String,
  pub iteration: u64,
  pub objective: f64,
  pub m: usize,
  pub k: usize,
  pub seed: u64,
  pub n_orders: usize,
  pub n_vehicles: usize,
  /// Per vehicle: list of (order id, pickup/delivery).
  pub routes: Vec<Vec<(u32, StopKind)>>,
  /// Order ids in feature/score row order (= problem.orders order).
  pub order_ids: Vec<u32>,
  /// n_orders x 8, row-major.
  pub features: Vec<Vec<f64>>,
  /// Mean improvement over the tries containing the order (null if never sampled).
  pub scores: Vec<Option<f64>>,
  /// Number of tries containing the order.
  pub counts: Vec<u32>,
  /// Whether the order was in the best try.
  pub in_best: Vec<bool>,
  /// Improvement (drop in total distance) of every try.
  pub try_improvements: Vec<f64>,
  /// Removed order indices of the best try.
  pub best_try: Vec<usize>,
  /// Share of the tries containing the order that were in the top set
  /// (null if never sampled). Absent in Stage 2 label files.
  #[serde(default)]
  pub top_share: Vec<Option<f64>>,
}

/// (scores, counts, in_best, try_improvements, best_try, top_share) of one labelled state.
pub type StateLabels = (
  Vec<Option<f64>>,
  Vec<u32>,
  Vec<bool>,
  Vec<f64>,
  Vec<usize>,
  Vec<Option<f64>>,
);

/// For every order: (tries containing it that are among the `ceil(top_frac * M)`
/// largest improvements) / (tries containing it). Ties go to the lower try index.
/// `None` if no try contains the order.
pub fn top_share(
  try_sets: &[Vec<usize>],
  improvements: &[f64],
  n_orders: usize,
  top_frac: f64,
) -> Vec<Option<f64>> {
  let m = try_sets.len();
  let n_top = ((top_frac * m as f64).ceil() as usize).min(m);
  let mut ranked: Vec<usize> = (0..m).collect();
  ranked.sort_by(|&a, &b| improvements[b].total_cmp(&improvements[a]).then(a.cmp(&b)));
  let mut is_top = vec![false; m];
  for &t in &ranked[..n_top] {
    is_top[t] = true;
  }
  let mut top = vec![0u32; n_orders];
  let mut all = vec![0u32; n_orders];
  for (t, set) in try_sets.iter().enumerate() {
    for &o in set {
      all[o] += 1;
      if is_top[t] {
        top[o] += 1;
      }
    }
  }
  (0..n_orders)
    .map(|o| (all[o] > 0).then(|| top[o] as f64 / all[o] as f64))
    .collect()
}

/// Evaluate one state: M random removal sets of size k, greedy repair.
pub fn label_state(
  ctx: &Context,
  sol: &WorkingSolution,
  m: usize,
  k: usize,
  top_frac: f64,
  rng: &mut Xoshiro256StarStar,
) -> StateLabels {
  let n = ctx.n_orders();
  let mut sum = vec![0.0; n];
  let mut counts = vec![0u32; n];
  let mut improvements = Vec::with_capacity(m);
  let mut try_sets: Vec<Vec<usize>> = Vec::with_capacity(m);
  let mut best_imp = f64::NEG_INFINITY;
  let mut best_set: Vec<usize> = Vec::new();
  let mut op = RandomDestroy;
  use crate::destroy::DestroyOperator;
  for _ in 0..m {
    let removed = op.select(ctx, sol, k, rng);
    let mut cand = sol.clone();
    destroy(ctx, &mut cand, &removed);
    let mut order = removed.clone();
    order.shuffle(rng);
    repair_greedy(ctx, &mut cand, &order);
    let imp = sol.total_distance - cand.total_distance;
    improvements.push(imp);
    for &o in &removed {
      sum[o] += imp;
      counts[o] += 1;
    }
    if imp > best_imp {
      best_imp = imp;
      best_set = removed.clone();
    }
    try_sets.push(removed);
  }
  let shares = top_share(&try_sets, &improvements, n, top_frac);
  let scores = (0..n)
    .map(|o| {
      if counts[o] > 0 {
        Some(sum[o] / counts[o] as f64)
      } else {
        None
      }
    })
    .collect();
  let mut in_best = vec![false; n];
  for &o in &best_set {
    in_best[o] = true;
  }
  (scores, counts, in_best, improvements, best_set, shares)
}

/// Run the random-removal LNS on one instance, snapshot states and label them.
pub fn label_instance(
  ctx: &Context,
  instance: &str,
  cell: &str,
  split: &str,
  cfg: &LabelConfig,
) -> Vec<LabelledState> {
  let n = ctx.n_orders();
  let k = cfg.k.unwrap_or_else(|| default_k(n)).min(n.max(1));
  let snap_iters = snapshot_iterations(cfg.lns_iters, cfg.states_per_instance, &cfg.schedule);

  let mut states: Vec<(u64, WorkingSolution, f64)> = Vec::new();
  {
    let mut hook = |iter: u64, sol: &WorkingSolution, obj: f64| {
      if snap_iters.contains(&iter) {
        states.push((iter, sol.clone(), obj));
      }
    };
    let lns_cfg = LnsConfig {
      max_iters: Some(cfg.lns_iters),
      max_secs: None,
      seed: cfg.seed,
      k: Some(k),
      log_every: 0,
      ..Default::default()
    };
    let mut op = RandomDestroy;
    let init = initial_solution(ctx, cfg.seed);
    run_lns(ctx, init, &mut op, &lns_cfg, Some(&mut hook));
  }

  let mut out = Vec::with_capacity(states.len());
  for (si, (iter, sol, obj)) in states.into_iter().enumerate() {
    let mut rng = Xoshiro256StarStar::seed_from_u64(
      cfg.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(si as u64 + 1)),
    );
    let (scores, counts, in_best, try_improvements, best_try, top_share) =
      label_state(ctx, &sol, cfg.m, k, cfg.top_frac, &mut rng);
    let feats = compute_features(ctx, &sol);
    let features: Vec<Vec<f64>> = (0..n).map(|o| feats[o * 8..(o + 1) * 8].to_vec()).collect();
    out.push(LabelledState {
      instance: instance.to_string(),
      cell: cell.to_string(),
      split: split.to_string(),
      iteration: iter,
      objective: obj,
      m: cfg.m,
      k,
      seed: cfg.seed,
      n_orders: n,
      n_vehicles: ctx.n_vehicles(),
      routes: stop_lists(ctx.problem, &sol),
      order_ids: ctx.problem.orders.iter().map(|o| o.id).collect(),
      features,
      scores,
      counts,
      in_best,
      try_improvements,
      best_try,
      top_share,
    });
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use vrppd_core::Problem;

  #[test]
  fn labels_have_expected_shape() {
    let dir =
      std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../problem-gen/problems/10_20");
    let mut files: Vec<_> = std::fs::read_dir(dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .map(|e| e.path())
      .collect();
    files.sort();
    let problem: Problem =
      serde_json::from_str(&std::fs::read_to_string(&files[0]).unwrap()).unwrap();
    let ctx = Context::new(&problem);
    let cfg = LabelConfig {
      lns_iters: 50,
      states_per_instance: 3,
      m: 8,
      k: None,
      seed: 1,
      schedule: "even".to_string(),
      top_frac: 0.1,
    };
    let states = label_instance(&ctx, "x", "10_20", "train", &cfg);
    assert_eq!(states.len(), 3);
    assert_eq!(
      states.iter().map(|s| s.iteration).collect::<Vec<_>>(),
      vec![16, 33, 50]
    );
    assert_eq!(snapshot_iterations(5000, 4, "log"), vec![8, 71, 595, 5000]);
    assert_eq!(snapshot_iterations(10, 20, "log").last(), Some(&10));
    for s in &states {
      assert_eq!(s.features.len(), 20);
      assert_eq!(s.scores.len(), 20);
      assert_eq!(s.try_improvements.len(), 8);
      assert_eq!(s.k, 3);
      assert_eq!(s.in_best.iter().filter(|b| **b).count(), 3);
      let total: u32 = s.counts.iter().sum();
      assert_eq!(total, 24);
      assert_eq!(s.top_share.len(), 20);
    }
  }

  #[test]
  fn top_share_small_example() {
    let sets = vec![vec![0, 1], vec![1, 2], vec![0, 2], vec![2, 3]];
    let imps = [5.0, 1.0, 4.0, 0.0];
    let got = top_share(&sets, &imps, 5, 0.5);
    assert_eq!(
      got,
      vec![Some(1.0), Some(0.5), Some(1.0 / 3.0), Some(0.0), None]
    );
  }

  #[test]
  fn top_share_ties_go_to_lower_try_index() {
    let sets = vec![vec![0], vec![1]];
    let got = top_share(&sets, &[1.0, 1.0], 2, 0.5);
    assert_eq!(got, vec![Some(1.0), Some(0.0)]);
  }

  #[test]
  fn labelled_state_without_top_share_still_parses() {
    let line = include_str!("../tests/fixtures/stage2_label_line.json");
    assert!(!line.contains("top_share"));
    let s: LabelledState = serde_json::from_str(line).unwrap();
    assert!(s.top_share.is_empty());
    assert_eq!(s.n_orders, 20);
    assert_eq!(s.n_vehicles, 10);
  }
}
