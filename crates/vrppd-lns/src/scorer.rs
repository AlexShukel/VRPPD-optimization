//! Order scorers for the neural destroy operator. A scorer gives every order
//! a removal score from the current solution; the operator picks the best.

use std::path::Path;

use vrppd_core::WorkingSolution;

use crate::encoder::{Encoder, InstanceEncoding, Workspace};
use crate::features::FeatureCache;
use crate::mlp::Mlp;
use crate::solution::Context;

pub trait Scorer {
  /// One score per order (problem order); higher = remove first. Scorers may
  /// keep state between calls, so use one scorer per search run (it resets itself when it sees another instance).
  fn score(&mut self, ctx: &Context, sol: &WorkingSolution) -> Vec<f64>;

  /// Model family name as written in the weights file ("b0", ...).
  fn model(&self) -> &str;
}

/// The B0 model: an MLP on the eight hand-made features of each order. The
/// features and the per-order scores are cached, so only orders on routes
/// that changed since the last call are rescored.
pub struct MlpScorer {
  mlp: Mlp,
  instance: Option<u64>,
  cache: FeatureCache,
  scores: Vec<f64>,
  scratch_a: Vec<f64>,
  scratch_b: Vec<f64>,
}

impl MlpScorer {
  pub fn new(mlp: Mlp) -> Self {
    Self {
      mlp,
      instance: None,
      cache: FeatureCache::empty(),
      scores: Vec::new(),
      scratch_a: Vec::new(),
      scratch_b: Vec::new(),
    }
  }
}

impl Scorer for MlpScorer {
  fn score(&mut self, ctx: &Context, sol: &WorkingSolution) -> Vec<f64> {
    let n = ctx.n_orders();
    // First call, or a scorer reused on another instance: start from scratch.
    let full = self.instance != Some(ctx.fingerprint);
    if full {
      self.instance = Some(ctx.fingerprint);
      self.cache = FeatureCache::new(ctx);
      self.scores = vec![0.0; n];
    }
    self.cache.update(ctx, sol);
    let feats = self.cache.features();
    let nf = self.mlp.n_features;
    let mut rescore = |o: usize| {
      self.scores[o] = self.mlp.score_row(
        &feats[o * nf..(o + 1) * nf],
        &mut self.scratch_a,
        &mut self.scratch_b,
      );
    };
    if full {
      (0..n).for_each(&mut rescore);
    } else {
      self.cache.changed().iter().copied().for_each(&mut rescore);
    }
    self.scores.clone()
  }

  fn model(&self) -> &str {
    "b0"
  }
}

/// The V1 model: the encoder over stop tokens and routes. The part before the
/// route layers depends only on the instance, so it is computed once per
/// instance (keyed on the context's fingerprint); each call runs the rest.
pub struct EncoderScorer {
  encoder: Encoder,
  encoded: Option<(u64, InstanceEncoding)>,
  ws: Workspace,
}

impl EncoderScorer {
  pub fn new(encoder: Encoder) -> Self {
    Self {
      encoder,
      encoded: None,
      ws: Workspace::default(),
    }
  }
}

impl Scorer for EncoderScorer {
  fn score(&mut self, ctx: &Context, sol: &WorkingSolution) -> Vec<f64> {
    if !matches!(&self.encoded, Some((fp, _)) if *fp == ctx.fingerprint) {
      let encoded = self.encoder.encode_instance(ctx.problem, &mut self.ws);
      self.encoded = Some((ctx.fingerprint, encoded));
    }
    let (_, encoded) = self.encoded.as_ref().unwrap();
    self.encoder.score_encoded(encoded, sol, &mut self.ws)
  }

  fn model(&self) -> &str {
    "v1"
  }
}

/// Load a scorer from a weights JSON. The optional `"model"` key picks the
/// family; files without it (Stage 2) are B0.
pub fn load_scorer(path: &Path) -> Result<Box<dyn Scorer>, String> {
  let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
  let value: serde_json::Value =
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
  match value.get("model").and_then(|m| m.as_str()) {
    None | Some("b0") => Ok(Box::new(MlpScorer::new(Mlp::load(path)?))),
    Some("v1") => Ok(Box::new(EncoderScorer::new(Encoder::load(path)?))),
    Some(other) => Err(format!("unknown model '{other}' in {}", path.display())),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("tests/fixtures")
      .join(name)
  }

  #[test]
  fn b0_weights_without_model_key_load() {
    let scorer = load_scorer(&fixture("mlp_fixture_weights.json")).unwrap();
    assert_eq!(scorer.model(), "b0");
  }

  fn load_10_20(idx: usize) -> vrppd_core::Problem {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../problem-gen/problems/10_20");
    let mut files: Vec<_> = std::fs::read_dir(dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .map(|e| e.path())
      .collect();
    files.sort();
    serde_json::from_str(&std::fs::read_to_string(&files[idx]).unwrap()).unwrap()
  }

  #[test]
  fn reused_scorer_matches_fresh_scorer_on_same_shape_instance() {
    use crate::lns::initial_solution;
    let (pa, pb) = (load_10_20(0), load_10_20(1));
    assert_eq!(pa.orders.len(), pb.orders.len());
    assert_eq!(pa.vehicles.len(), pb.vehicles.len());
    let (ca, cb) = (Context::new(&pa), Context::new(&pb));
    assert_ne!(ca.fingerprint, cb.fingerprint);
    let (sa, sb) = (initial_solution(&ca, 1), initial_solution(&cb, 1));
    for name in ["mlp_fixture_weights.json", "v1_fixture.json"] {
      let weights = fixture(name);
      let mut reused = load_scorer(&weights).unwrap();
      reused.score(&ca, &sa);
      let got = reused.score(&cb, &sb);
      let fresh = load_scorer(&weights).unwrap().score(&cb, &sb);
      assert_eq!(got, fresh, "{name}");
    }
  }

  #[test]
  fn v1_cached_scorer_matches_full_forward_pass() {
    use crate::destroy::RandomDestroy;
    use crate::lns::{LnsConfig, initial_solution, run_lns};
    // EncoderScorer keeps the route-independent part per instance; at every
    // step of a search it must agree bit for bit with the uncached pass.
    let problem = load_10_20(0);
    let ctx = Context::new(&problem);
    let weights = fixture("v1_fixture.json");
    let encoder = Encoder::load(&weights).unwrap();
    let mut scorer = load_scorer(&weights).unwrap();
    assert_eq!(scorer.model(), "v1");
    let mut checked = 0;
    let mut hook = |_: u64, sol: &WorkingSolution, _: f64| {
      assert_eq!(scorer.score(&ctx, sol), encoder.score(&ctx, sol));
      checked += 1;
    };
    let cfg = LnsConfig {
      max_iters: Some(30),
      seed: 2,
      ..Default::default()
    };
    run_lns(
      &ctx,
      initial_solution(&ctx, 2),
      &mut RandomDestroy,
      &cfg,
      Some(&mut hook),
    );
    assert_eq!(checked, 30);
  }

  #[test]
  fn model_key_selects_family() {
    let dir = std::env::temp_dir().join(format!("vrppd-scorer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let text = std::fs::read_to_string(fixture("mlp_fixture_weights.json")).unwrap();
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    for (model, ok) in [("b0", true), ("v1", false), ("zzz", false)] {
      value["model"] = model.into();
      let path = dir.join(format!("{model}.json"));
      std::fs::write(&path, value.to_string()).unwrap();
      let got = load_scorer(&path);
      assert_eq!(got.is_ok(), ok, "{model}");
      if model == "v1" {
        // Dispatched to the V1 loader, which finds no V1 config in a B0 file.
        let err = got.err().unwrap();
        assert!(err.contains("missing field `config`"), "{err}");
      }
    }
    let v1 = load_scorer(&fixture("v1_fixture.json")).unwrap();
    assert_eq!(v1.model(), "v1");
    std::fs::remove_dir_all(&dir).unwrap();
  }
}
