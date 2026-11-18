//! Order scorers for the neural destroy operator. A scorer gives every order
//! a removal score from the current solution; the operator picks the best.

use std::path::Path;

use vrppd_core::WorkingSolution;

use crate::features::FeatureCache;
use crate::mlp::Mlp;
use crate::solution::Context;

pub trait Scorer {
  /// One score per order (problem order); higher = remove first. Scorers may
  /// keep state between calls, so use one scorer per search run.
  fn score(&mut self, ctx: &Context, sol: &WorkingSolution) -> Vec<f64>;

  /// Model family name as written in the weights file ("b0", ...).
  fn model(&self) -> &str;
}

/// The B0 model: an MLP on the eight hand-made features of each order. The
/// features and the per-order scores are cached, so only orders on routes
/// that changed since the last call are rescored.
pub struct MlpScorer {
  mlp: Mlp,
  shape: Option<(usize, usize)>,
  cache: FeatureCache,
  scores: Vec<f64>,
  scratch_a: Vec<f64>,
  scratch_b: Vec<f64>,
}

impl MlpScorer {
  pub fn new(mlp: Mlp) -> Self {
    Self {
      mlp,
      shape: None,
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
    let shape = (n, ctx.n_vehicles());
    // First call (or a scorer reused on another instance): start from scratch.
    let full = self.shape != Some(shape);
    if full {
      self.shape = Some(shape);
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

/// Load a scorer from a weights JSON. The optional `"model"` key picks the
/// family; files without it (Stage 2) are B0.
pub fn load_scorer(path: &Path) -> Result<Box<dyn Scorer>, String> {
  let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
  let value: serde_json::Value =
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
  match value.get("model").and_then(|m| m.as_str()) {
    None | Some("b0") => Ok(Box::new(MlpScorer::new(Mlp::load(path)?))),
    Some("v1") => Err("model v1 not supported yet".to_string()),
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
        assert_eq!(got.err().unwrap(), "model v1 not supported yet");
      }
    }
    std::fs::remove_dir_all(&dir).unwrap();
  }
}
