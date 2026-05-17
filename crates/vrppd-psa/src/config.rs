//! Simulated-annealing hyperparameters and per-objective tuned defaults.

use vrppd_core::Objective;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperatorWeights {
  pub shift: f64,
  pub swap: f64,
  pub shuffle: f64,
}

impl Default for OperatorWeights {
  fn default() -> Self {
    Self {
      shift: 0.4,
      swap: 0.3,
      shuffle: 0.3,
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaConfig {
  pub initial_temp: f64,
  pub cooling_rate: f64,
  pub min_temp: f64,
  pub max_iterations: u64,
  pub weights: OperatorWeights,
  pub threads: usize,
  pub batch_size: u64,
  pub sync_interval: u64,
  pub reheat_floor: f64,
}

pub fn default_config_for(target: Objective) -> SaConfig {
  let threads = num_cpus_or(2);
  match target {
    Objective::Empty => SaConfig {
      initial_temp: 500.0,
      cooling_rate: 0.99,
      min_temp: 0.1,
      max_iterations: 1_000,
      weights: OperatorWeights::default(),
      threads,
      batch_size: 50,
      sync_interval: 10,
      reheat_floor: 50.0,
    },
    Objective::Distance => SaConfig {
      initial_temp: 500.0,
      cooling_rate: 0.999,
      min_temp: 0.1,
      max_iterations: 10_000,
      weights: OperatorWeights::default(),
      threads,
      batch_size: 100,
      sync_interval: 4,
      reheat_floor: 50.0,
    },
    Objective::Price => SaConfig {
      initial_temp: 1500.0,
      cooling_rate: 0.999,
      min_temp: 0.1,
      max_iterations: 10_000,
      weights: OperatorWeights::default(),
      threads,
      batch_size: 200,
      sync_interval: 4,
      reheat_floor: 50.0,
    },
  }
}

#[inline]
fn num_cpus_or(min: usize) -> usize {
  std::thread::available_parallelism()
    .map(|n| n.get().max(min))
    .unwrap_or(min)
}
