//! Destroy operators: which `k` orders to remove from the current solution.

use rand::Rng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar;
use vrppd_core::WorkingSolution;

use crate::scorer::{Scorer, load_scorer};
use crate::solution::{Context, placements, removal_saving};

/// What happened to the solution produced after this operator's destroy + repair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
  /// New global best.
  NewBest,
  /// Better than the current solution (but not a new best).
  Improved,
  /// Worse than current but accepted by the acceptance criterion.
  AcceptedWorse,
  /// Rejected.
  Rejected,
}

pub trait DestroyOperator {
  fn name(&self) -> &str;

  /// Choose the orders to remove. Must return exactly `k` distinct routed orders
  /// (or fewer if there are not enough orders).
  fn select(
    &mut self,
    ctx: &Context,
    sol: &WorkingSolution,
    k: usize,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize>;

  /// Feedback after the iteration (used by the adaptive operator).
  fn feedback(&mut self, _outcome: Outcome) {}

  /// Operator used in the last iteration (for logging; adaptive reports the sub-operator).
  fn last_used(&self) -> &str {
    self.name()
  }
}

fn routed_orders(ctx: &Context, sol: &WorkingSolution) -> Vec<usize> {
  placements(sol, ctx.n_orders())
    .iter()
    .enumerate()
    .filter_map(|(i, p)| p.map(|_| i))
    .collect()
}

/// Random removal: `k` orders chosen uniformly at random.
#[derive(Default)]
pub struct RandomDestroy;

impl DestroyOperator for RandomDestroy {
  fn name(&self) -> &str {
    "random"
  }

  fn select(
    &mut self,
    ctx: &Context,
    sol: &WorkingSolution,
    k: usize,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize> {
    let mut orders = routed_orders(ctx, sol);
    orders.shuffle(rng);
    orders.truncate(k);
    orders
  }
}

/// Worst (most expensive) removal: orders with the largest route saving first,
/// with the Ropke & Pisinger randomisation (pick index floor(y^p * |L|)) so that
/// consecutive iterations do not remove exactly the same set. `p = 1` would be
/// uniform random; larger `p` is closer to deterministic worst-first.
pub struct WorstDestroy {
  pub p: f64,
}

impl Default for WorstDestroy {
  fn default() -> Self {
    Self { p: 3.0 }
  }
}

impl DestroyOperator for WorstDestroy {
  fn name(&self) -> &str {
    "worst"
  }

  fn select(
    &mut self,
    ctx: &Context,
    sol: &WorkingSolution,
    k: usize,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize> {
    let pl = placements(sol, ctx.n_orders());
    let mut scored: Vec<(f64, usize)> = pl
      .iter()
      .enumerate()
      .filter_map(|(o, p)| p.map(|p| (removal_saving(ctx, sol, p, o), o)))
      .collect();
    // descending by saving
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::with_capacity(k);
    while out.len() < k && !scored.is_empty() {
      let y: f64 = rng.r#gen();
      let idx = ((y.powf(self.p)) * scored.len() as f64).floor() as usize;
      let idx = idx.min(scored.len() - 1);
      out.push(scored.remove(idx).1);
    }
    out
  }
}

/// Shaw related removal: a random seed order plus the `k-1` orders most
/// related to it (pickup-pickup + delivery-delivery distance).
#[derive(Default)]
pub struct RelatedDestroy;

impl DestroyOperator for RelatedDestroy {
  fn name(&self) -> &str {
    "related"
  }

  fn select(
    &mut self,
    ctx: &Context,
    sol: &WorkingSolution,
    k: usize,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize> {
    let orders = routed_orders(ctx, sol);
    if orders.is_empty() || k == 0 {
      return Vec::new();
    }
    let seed = orders[rng.gen_range(0..orders.len())];
    let mut others: Vec<(f64, usize)> = orders
      .iter()
      .filter(|&&o| o != seed)
      .map(|&o| (ctx.related(seed, o), o))
      .collect();
    others.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![seed];
    out.extend(others.into_iter().take(k.saturating_sub(1)).map(|(_, o)| o));
    out
  }
}

/// Adaptive selection (Ropke & Pisinger 2006): roulette over sub-operators with
/// weights updated at the end of every segment from the scores collected.
pub struct AdaptiveDestroy {
  name: String,
  pub ops: Vec<Box<dyn DestroyOperator>>,
  pub weights: Vec<f64>,
  scores: Vec<f64>,
  uses: Vec<usize>,
  pub segment_len: usize,
  pub reaction: f64,
  pub sigma: [f64; 3],
  iter_in_segment: usize,
  last: usize,
}

impl AdaptiveDestroy {
  pub fn new(ops: Vec<Box<dyn DestroyOperator>>) -> Self {
    let n = ops.len();
    Self {
      name: "adaptive".to_string(),
      ops,
      weights: vec![1.0; n],
      scores: vec![0.0; n],
      uses: vec![0; n],
      segment_len: 100,
      reaction: 0.1,
      sigma: [33.0, 9.0, 13.0],
      iter_in_segment: 0,
      last: 0,
    }
  }

  /// Same selection and feedback mechanism under another operator name.
  pub fn named(mut self, name: &str) -> Self {
    self.name = name.to_string();
    self
  }

  pub fn classical() -> Self {
    Self::new(vec![
      Box::new(RandomDestroy),
      Box::new(WorstDestroy::default()),
      Box::new(RelatedDestroy),
    ])
  }
}

impl DestroyOperator for AdaptiveDestroy {
  fn name(&self) -> &str {
    &self.name
  }

  fn select(
    &mut self,
    ctx: &Context,
    sol: &WorkingSolution,
    k: usize,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize> {
    let total: f64 = self.weights.iter().sum();
    let mut r: f64 = rng.r#gen::<f64>() * total;
    let mut chosen = self.weights.len() - 1;
    for (i, w) in self.weights.iter().enumerate() {
      if r < *w {
        chosen = i;
        break;
      }
      r -= w;
    }
    self.last = chosen;
    self.uses[chosen] += 1;
    self.ops[chosen].select(ctx, sol, k, rng)
  }

  fn feedback(&mut self, outcome: Outcome) {
    let s = match outcome {
      Outcome::NewBest => self.sigma[0],
      Outcome::Improved => self.sigma[1],
      Outcome::AcceptedWorse => self.sigma[2],
      Outcome::Rejected => 0.0,
    };
    self.scores[self.last] += s;
    self.iter_in_segment += 1;
    if self.iter_in_segment >= self.segment_len {
      for i in 0..self.weights.len() {
        if self.uses[i] > 0 {
          let pi = self.scores[i] / self.uses[i] as f64;
          self.weights[i] = (1.0 - self.reaction) * self.weights[i] + self.reaction * pi;
          // keep every operator selectable
          self.weights[i] = self.weights[i].max(0.05);
        }
        self.scores[i] = 0.0;
        self.uses[i] = 0;
      }
      self.iter_in_segment = 0;
    }
  }

  fn last_used(&self) -> &str {
    self.ops[self.last].name()
  }
}

/// Neural destroy: score every order with a [`Scorer`] (B0: MLP on the eight features) and
/// remove the top-k (tau = 0) or sample without replacement from
/// softmax(score / tau) (tau > 0).
pub struct NeuralDestroy {
  pub scorer: Box<dyn Scorer>,
  pub tau: f64,
  pub last_scores: Vec<f64>,
}

impl NeuralDestroy {
  pub fn new(scorer: Box<dyn Scorer>, tau: f64) -> Self {
    Self {
      scorer,
      tau,
      last_scores: Vec::new(),
    }
  }

  /// Select `k` indices from `scores` (top-k if tau == 0, otherwise
  /// sampling without replacement proportional to exp(score / tau)).
  pub fn pick(
    scores: &[f64],
    candidates: &[usize],
    k: usize,
    tau: f64,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize> {
    if candidates.is_empty() || k == 0 {
      return Vec::new();
    }
    if tau <= 0.0 {
      let mut idx: Vec<usize> = candidates.to_vec();
      idx.sort_by(|&a, &b| {
        scores[b]
          .partial_cmp(&scores[a])
          .unwrap_or(std::cmp::Ordering::Equal)
      });
      idx.truncate(k);
      return idx;
    }
    let mut remaining: Vec<usize> = candidates.to_vec();
    let mut out = Vec::with_capacity(k);
    while out.len() < k && !remaining.is_empty() {
      let max = remaining
        .iter()
        .map(|&i| scores[i])
        .fold(f64::NEG_INFINITY, f64::max);
      let weights: Vec<f64> = remaining
        .iter()
        .map(|&i| ((scores[i] - max) / tau).exp())
        .collect();
      let total: f64 = weights.iter().sum();
      let mut r = rng.r#gen::<f64>() * total;
      let mut chosen = remaining.len() - 1;
      for (j, w) in weights.iter().enumerate() {
        if r < *w {
          chosen = j;
          break;
        }
        r -= w;
      }
      out.push(remaining.swap_remove(chosen));
    }
    out
  }
}

impl DestroyOperator for NeuralDestroy {
  fn name(&self) -> &str {
    "neural"
  }

  fn select(
    &mut self,
    ctx: &Context,
    sol: &WorkingSolution,
    k: usize,
    rng: &mut Xoshiro256StarStar,
  ) -> Vec<usize> {
    self.last_scores = self.scorer.score(ctx, sol);
    let candidates = routed_orders(ctx, sol);
    Self::pick(&self.last_scores, &candidates, k, self.tau, rng)
  }
}

/// Build a destroy operator by name. `neural` needs `weights`.
pub fn build_operator(
  name: &str,
  weights: Option<&std::path::Path>,
  tau: f64,
  worst_p: f64,
) -> Result<Box<dyn DestroyOperator>, String> {
  Ok(match name {
    "random" => Box::new(RandomDestroy),
    "worst" => Box::new(WorstDestroy { p: worst_p }),
    "related" => Box::new(RelatedDestroy),
    "adaptive" => Box::new(AdaptiveDestroy::new(vec![
      Box::new(RandomDestroy),
      Box::new(WorstDestroy { p: worst_p }),
      Box::new(RelatedDestroy),
    ])),
    "neural" => {
      let path = weights.ok_or("neural operator needs --weights")?;
      Box::new(NeuralDestroy::new(load_scorer(path)?, tau))
    }
    "adaptive_n" => {
      let path = weights.ok_or("adaptive_n operator needs --weights")?;
      Box::new(
        AdaptiveDestroy::new(vec![
          Box::new(RandomDestroy),
          Box::new(WorstDestroy { p: worst_p }),
          Box::new(RelatedDestroy),
          Box::new(NeuralDestroy::new(load_scorer(path)?, tau)),
        ])
        .named("adaptive_n"),
      )
    }
    other => return Err(format!("unknown operator {other}")),
  })
}

pub const OPERATOR_NAMES: [&str; 6] = [
  "random",
  "worst",
  "related",
  "adaptive",
  "neural",
  "adaptive_n",
];

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;

  #[test]
  fn pick_topk_is_deterministic_and_sorted() {
    let scores = [0.1, 5.0, 2.0, 3.0];
    let mut rng = Xoshiro256StarStar::seed_from_u64(1);
    let got = NeuralDestroy::pick(&scores, &[0, 1, 2, 3], 2, 0.0, &mut rng);
    assert_eq!(got, vec![1, 3]);
  }

  #[test]
  fn pick_sampling_returns_distinct() {
    let scores = [0.1, 5.0, 2.0, 3.0, -1.0];
    let mut rng = Xoshiro256StarStar::seed_from_u64(1);
    for _ in 0..50 {
      let got = NeuralDestroy::pick(&scores, &[0, 1, 2, 3, 4], 3, 1.0, &mut rng);
      assert_eq!(got.len(), 3);
      let mut s = got.clone();
      s.sort();
      s.dedup();
      assert_eq!(s.len(), 3);
    }
  }

  fn fixture_weights() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlp_fixture_weights.json")
  }

  fn load_problem(cell: &str, idx: usize) -> vrppd_core::Problem {
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
  fn adaptive_n_uses_all_four_rules() {
    use crate::lns::{LnsConfig, initial_solution, run_lns};
    use std::collections::BTreeSet;
    let problem = load_problem("10_20", 0);
    let ctx = Context::new(&problem);
    let mut op = build_operator("adaptive_n", Some(&fixture_weights()), 1.0, 3.0).unwrap();
    assert_eq!(op.name(), "adaptive_n");
    let cfg = LnsConfig {
      max_iters: Some(2000),
      seed: 3,
      ..Default::default()
    };
    let mut seen = BTreeSet::new();
    {
      let op_ref: &mut dyn DestroyOperator = op.as_mut();
      // Wrap the operator to record which rule ran each iteration.
      struct Spy<'a>(&'a mut dyn DestroyOperator, &'a mut BTreeSet<String>);
      impl DestroyOperator for Spy<'_> {
        fn name(&self) -> &str {
          self.0.name()
        }
        fn select(
          &mut self,
          ctx: &Context,
          sol: &WorkingSolution,
          k: usize,
          rng: &mut Xoshiro256StarStar,
        ) -> Vec<usize> {
          let out = self.0.select(ctx, sol, k, rng);
          self.1.insert(self.0.last_used().to_string());
          out
        }
        fn feedback(&mut self, outcome: Outcome) {
          self.0.feedback(outcome)
        }
      }
      let mut spy = Spy(op_ref, &mut seen);
      run_lns(&ctx, initial_solution(&ctx, 3), &mut spy, &cfg, None);
    }
    let expect: BTreeSet<String> = ["random", "worst", "related", "neural"]
      .iter()
      .map(|s| s.to_string())
      .collect();
    assert_eq!(seen, expect);
  }

  #[test]
  fn adaptive_n_needs_weights() {
    assert!(build_operator("adaptive_n", None, 1.0, 3.0).is_err());
  }

  #[test]
  fn neural_scores_unchanged_by_cache() {
    use crate::features::compute_features;
    use crate::lns::{LnsConfig, initial_solution, run_lns};
    use crate::mlp::Mlp;
    // Stage 2 path: full feature matrix, then the MLP on all rows. The cached
    // scorer must agree bit for bit at every step of a search.
    let problem = load_problem("10_20", 0);
    let ctx = Context::new(&problem);
    let mlp = Mlp::load(&fixture_weights()).unwrap();
    let mut neural = NeuralDestroy::new(load_scorer(&fixture_weights()).unwrap(), 0.0);
    let mut rng = Xoshiro256StarStar::seed_from_u64(9);
    let mut checked = 0;
    let mut hook = |_: u64, sol: &WorkingSolution, _: f64| {
      neural.select(&ctx, sol, 3, &mut rng);
      let expected = mlp.score_all(&compute_features(&ctx, sol), ctx.n_orders());
      assert_eq!(neural.last_scores, expected);
      checked += 1;
    };
    let mut op = RandomDestroy;
    let cfg = LnsConfig {
      max_iters: Some(200),
      seed: 4,
      ..Default::default()
    };
    run_lns(
      &ctx,
      initial_solution(&ctx, 4),
      &mut op,
      &cfg,
      Some(&mut hook),
    );
    assert_eq!(checked, 200);
  }
}
