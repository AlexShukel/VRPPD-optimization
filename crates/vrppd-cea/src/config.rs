//! CEA hyperparameters and defaults.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CeaConfig {
  pub population_size: usize,

  pub conv_count: usize,

  pub wall_time_cap_ms: Option<u64>,

  pub recombination_fraction_low: f64,
  pub recombination_fraction_high: f64,

  pub p_reinsertion: f64,

  pub p_crossover: f64,

  pub threads: usize,
}

impl Default for CeaConfig {
  fn default() -> Self {
    Self {
      population_size: 50,
      conv_count: 500,
      wall_time_cap_ms: Some(30 * 60 * 1000),
      recombination_fraction_low: 0.1,
      recombination_fraction_high: 0.5,
      p_reinsertion: 0.5,
      p_crossover: 0.5,
      threads: num_cpus_or(2),
    }
  }
}

impl CeaConfig {
  pub fn small_for_tests() -> Self {
    Self {
      population_size: 10,
      conv_count: 50,
      wall_time_cap_ms: Some(5_000),
      threads: 1,
      ..Self::default()
    }
  }
}

#[inline]
fn num_cpus_or(min: usize) -> usize {
  std::thread::available_parallelism()
    .map(|n| n.get().max(min))
    .unwrap_or(min)
}
