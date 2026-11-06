//! Tiny MLP forward pass in pure Rust, loading weights exported by
//! `python/vrppd_nn` as JSON.
//!
//! JSON schema:
//! ```json
//! {
//!   "n_features": 8,
//!   "feature_mean": [..8..],
//!   "feature_std": [..8..],
//!   "activation": "relu",
//!   "layers": [ {"weight": [[out x in]], "bias": [out]}, ... ]
//! }
//! ```
//! Every layer except the last is followed by ReLU; the last layer is linear
//! with a single output (the score).

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
  /// out x in
  pub weight: Vec<Vec<f64>>,
  pub bias: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mlp {
  pub n_features: usize,
  pub feature_mean: Vec<f64>,
  pub feature_std: Vec<f64>,
  #[serde(default = "default_activation")]
  pub activation: String,
  pub layers: Vec<Layer>,
}

fn default_activation() -> String {
  "relu".to_string()
}

impl Mlp {
  pub fn load(path: &Path) -> Result<Self, String> {
    let text =
      std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mlp: Mlp =
      serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    mlp.validate()?;
    Ok(mlp)
  }

  pub fn validate(&self) -> Result<(), String> {
    if self.activation != "relu" {
      return Err(format!("unsupported activation {}", self.activation));
    }
    if self.feature_mean.len() != self.n_features || self.feature_std.len() != self.n_features {
      return Err("feature_mean/std length mismatch".into());
    }
    let mut width = self.n_features;
    for (i, l) in self.layers.iter().enumerate() {
      if l.weight.len() != l.bias.len() {
        return Err(format!("layer {i}: weight rows != bias len"));
      }
      for row in &l.weight {
        if row.len() != width {
          return Err(format!(
            "layer {i}: expected {} inputs, got {}",
            width,
            row.len()
          ));
        }
      }
      width = l.bias.len();
    }
    if width != 1 {
      return Err(format!("last layer must have 1 output, has {width}"));
    }
    Ok(())
  }

  /// Score one raw (unstandardised) feature row.
  pub fn score_row(&self, raw: &[f64], scratch_a: &mut Vec<f64>, scratch_b: &mut Vec<f64>) -> f64 {
    scratch_a.clear();
    for (i, x) in raw.iter().take(self.n_features).enumerate() {
      let s = if self.feature_std[i] > 0.0 {
        self.feature_std[i]
      } else {
        1.0
      };
      scratch_a.push((x - self.feature_mean[i]) / s);
    }
    let n_layers = self.layers.len();
    for (li, layer) in self.layers.iter().enumerate() {
      scratch_b.clear();
      for (w_row, b) in layer.weight.iter().zip(&layer.bias) {
        let mut acc = *b;
        for (w, x) in w_row.iter().zip(scratch_a.iter()) {
          acc += w * x;
        }
        if li + 1 < n_layers && acc < 0.0 {
          acc = 0.0;
        }
        scratch_b.push(acc);
      }
      std::mem::swap(scratch_a, scratch_b);
    }
    scratch_a[0]
  }

  /// Score every order given the raw feature matrix (n x n_features).
  pub fn score_all(&self, features: &[f64], n: usize) -> Vec<f64> {
    let mut a = Vec::new();
    let mut b = Vec::new();
    (0..n)
      .map(|i| {
        self.score_row(
          &features[i * self.n_features..(i + 1) * self.n_features],
          &mut a,
          &mut b,
        )
      })
      .collect()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn forward_matches_python_fixture() {
    // Fixture produced by python/vrppd_nn/make_fixture.py (numpy reference).
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mlp = Mlp::load(&dir.join("mlp_fixture_weights.json")).unwrap();
    let text = std::fs::read_to_string(dir.join("mlp_fixture_io.json")).unwrap();
    let io: serde_json::Value = serde_json::from_str(&text).unwrap();
    let inputs = io["inputs"].as_array().unwrap();
    let outputs = io["outputs"].as_array().unwrap();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for (x, y) in inputs.iter().zip(outputs) {
      let row: Vec<f64> = x
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
      let expected = y.as_f64().unwrap();
      let got = mlp.score_row(&row, &mut a, &mut b);
      assert!(
        (got - expected).abs() < 1e-6,
        "got {got}, expected {expected}"
      );
    }
  }

  #[test]
  fn hand_built_two_layer() {
    let mlp = Mlp {
      n_features: 2,
      feature_mean: vec![0.0, 0.0],
      feature_std: vec![1.0, 1.0],
      activation: "relu".into(),
      layers: vec![
        Layer {
          weight: vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
          bias: vec![0.0, 0.0],
        },
        Layer {
          weight: vec![vec![1.0, 2.0]],
          bias: vec![0.5],
        },
      ],
    };
    mlp.validate().unwrap();
    // x = (3, 1): h = relu(2, -2) = (2, 0) -> 2 + 0.5 = 2.5
    let s = mlp.score_all(&[3.0, 1.0, 1.0, 3.0], 2);
    assert!((s[0] - 2.5).abs() < 1e-12);
    // x = (1, 3): h = relu(-2, 2) = (0, 2) -> 4 + 0.5 = 4.5
    assert!((s[1] - 4.5).abs() < 1e-12);
  }
}
