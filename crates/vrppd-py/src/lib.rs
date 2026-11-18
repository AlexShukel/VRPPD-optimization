//! Python module `vrppd_py`: the LNS destroy, greedy repair and
//! simulated-annealing acceptance of `vrppd-lns` as a batch of environments,
//! one per instance, for training a destroy policy (distance objective).
//!
//! Order indices are positions in the instance's order list (0..N-1), not
//! order ids.

use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::ThreadPool;
use rayon::prelude::*;
use self_cell::self_cell;
use vrppd_core::{Problem, StopKind, WorkingSolution};
use vrppd_lns::Context;
use vrppd_lns::lns::{Annealing, LnsConfig, initial_solution, k_for, sa_accept, shuffle};
use vrppd_lns::solution::{destroy, repair_greedy};

self_cell!(
  /// A problem together with its LNS context, which borrows it.
  struct Instance {
    owner: Problem,
    #[covariant]
    dependent: Context,
  }
);

/// The search state of one instance.
struct Env {
  instance: Instance,
  k: usize,
  /// Stream of `reset` and `step`: start solutions, repair order, acceptance.
  rng: Xoshiro256StarStar,
  current: WorkingSolution,
  annealing: Annealing,
  iteration: u64,
  /// Seed of `evaluate` in the current state, drawn from `rng` whenever the
  /// state changes so that `evaluate` never advances `rng`.
  eval_seed: u64,
}

impl Env {
  fn new(problem: Problem, k_frac: f64, rng: Xoshiro256StarStar) -> Self {
    let k = k_for(problem.orders.len(), k_frac);
    let n_vehicles = problem.vehicles.len();
    let mut env = Env {
      instance: Instance::new(problem, |p| Context::new(p)),
      k,
      rng,
      current: WorkingSolution::empty(n_vehicles),
      annealing: Annealing::new(&LnsConfig::default(), 0.0),
      iteration: 0,
      eval_seed: 0,
    };
    env.reset();
    env
  }

  fn ctx(&self) -> &Context<'_> {
    self.instance.borrow_dependent()
  }

  fn objective(&self) -> f64 {
    self.ctx().objective_value(&self.current)
  }

  /// RSCIM start solution, iteration counter 0; the schedule of `run_lns`
  /// with the default configuration, scaled to the start objective.
  fn reset(&mut self) {
    let ctx = self.instance.borrow_dependent();
    self.current = initial_solution(ctx, self.rng.next_u64());
    self.annealing = Annealing::new(&LnsConfig::default(), ctx.objective_value(&self.current));
    self.iteration = 0;
    self.eval_seed = self.rng.next_u64();
  }

  /// Percentage improvement `100 * (before - after) / before` of each
  /// removal set; leaves the state unchanged.
  fn evaluate(&self, sets: &[Vec<usize>]) -> Vec<f64> {
    let before = self.objective();
    sets
      .par_iter()
      .enumerate()
      .map(|(j, removed)| {
        let mut rng = Xoshiro256StarStar::seed_from_u64(sample_seed(self.eval_seed, j));
        let ctx = self.ctx();
        let after = ctx.objective_value(&repaired(ctx, &self.current, removed, &mut rng));
        100.0 * (before - after) / before
      })
      .collect()
  }

  /// One `run_lns` iteration with the given removal set, cooling over
  /// `horizon` iterations; returns whether the candidate was accepted.
  fn step(&mut self, removed: &[usize], horizon: u64) -> bool {
    let ctx = self.instance.borrow_dependent();
    let progress = self.iteration as f64 / horizon as f64;
    self.iteration += 1;
    let temperature = self.annealing.temperature(progress);
    let candidate = repaired(ctx, &self.current, removed, &mut self.rng);
    let delta = ctx.objective_value(&candidate) - ctx.objective_value(&self.current);
    let accepted = sa_accept(delta, temperature, &mut self.rng);
    if accepted {
      self.current = candidate;
    }
    self.eval_seed = self.rng.next_u64();
    accepted
  }

  fn routes(&self) -> Vec<Vec<(usize, &'static str)>> {
    self
      .current
      .routes
      .iter()
      .map(|r| {
        r.stops
          .iter()
          .map(|s| {
            let kind = match s.kind {
              StopKind::Pickup => "pickup",
              StopKind::Delivery => "delivery",
            };
            (s.order_idx, kind)
          })
          .collect()
      })
      .collect()
  }
}

/// A copy of `current` with `removed` destroyed and greedily reinserted in
/// random order, as in `run_lns`.
fn repaired(
  ctx: &Context,
  current: &WorkingSolution,
  removed: &[usize],
  rng: &mut Xoshiro256StarStar,
) -> WorkingSolution {
  let mut candidate = current.clone();
  destroy(ctx, &mut candidate, removed);
  let mut order = removed.to_vec();
  shuffle(&mut order, rng);
  repair_greedy(ctx, &mut candidate, &order);
  candidate
}

/// Seed of evaluation sample `j`: output `j + 1` of a SplitMix64 generator
/// started at `eval_seed`, so samples differ from each other and from `step`.
fn sample_seed(eval_seed: u64, j: usize) -> u64 {
  let mut z = eval_seed.wrapping_add((j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^ (z >> 31)
}

fn load_problem(path: &Path) -> PyResult<Problem> {
  let text = std::fs::read_to_string(path)
    .map_err(|e| PyOSError::new_err(format!("read {}: {e}", path.display())))?;
  let problem: Problem = serde_json::from_str(&text)
    .map_err(|e| PyValueError::new_err(format!("parse {}: {e}", path.display())))?;
  if problem.orders.is_empty() || problem.vehicles.is_empty() {
    return Err(PyValueError::new_err(format!(
      "{}: an instance needs at least one order and one vehicle",
      path.display()
    )));
  }
  Ok(problem)
}

fn check_count(got: usize, want: usize) -> PyResult<()> {
  if got != want {
    return Err(PyValueError::new_err(format!(
      "expected one entry per environment ({want}), got {got}"
    )));
  }
  Ok(())
}

/// Validate one removal set of environment `env` (indices in range, no
/// duplicates).
fn removal_set(raw: &[i64], n_orders: usize, env: usize) -> PyResult<Vec<usize>> {
  let mut seen = vec![false; n_orders];
  raw
    .iter()
    .map(|&i| {
      let idx = usize::try_from(i)
        .ok()
        .filter(|&idx| idx < n_orders)
        .ok_or_else(|| {
          PyValueError::new_err(format!(
            "environment {env}: order index {i} out of range 0..{n_orders}"
          ))
        })?;
      if std::mem::replace(&mut seen[idx], true) {
        return Err(PyValueError::new_err(format!(
          "environment {env}: order index {i} appears twice in one removal set"
        )));
      }
      Ok(idx)
    })
    .collect()
}

/// Run `f` with the GIL released, on `pool` or else on rayon's global pool.
fn parallel<T: Send>(py: Python<'_>, pool: Option<&ThreadPool>, f: impl FnOnce() -> T + Send) -> T {
  py.detach(|| match pool {
    Some(pool) => pool.install(f),
    None => f(),
  })
}

/// A batch of LNS environments over the distance objective, one per
/// instance, stepped and evaluated in parallel. The environments start
/// reset. Each owns a random stream: the stream of `seed` jumped ahead by
/// 2^128 draws per environment index. `threads = 0` runs on rayon's global
/// pool with its default size; `threads > 0` gives this batch its own pool,
/// so the global pool is never configured. Results do not depend on threads.
#[pyclass(module = "vrppd_py")]
struct BatchEnv {
  envs: Vec<Env>,
  horizon: u64,
  pool: Option<ThreadPool>,
}

#[pymethods]
impl BatchEnv {
  #[new]
  #[pyo3(signature = (instance_paths, seed, k_frac = 0.15, horizon = 200, threads = 0))]
  fn new(
    py: Python<'_>,
    instance_paths: Vec<PathBuf>,
    seed: u64,
    k_frac: f64,
    horizon: u64,
    threads: usize,
  ) -> PyResult<Self> {
    if instance_paths.is_empty() {
      return Err(PyValueError::new_err("instance_paths is empty"));
    }
    if !(k_frac > 0.0 && k_frac <= 1.0) {
      return Err(PyValueError::new_err(format!(
        "k_frac must be in (0, 1], got {k_frac}"
      )));
    }
    if horizon == 0 {
      return Err(PyValueError::new_err("horizon must be at least 1"));
    }
    let problems = instance_paths
      .iter()
      .map(|p| load_problem(p))
      .collect::<PyResult<Vec<_>>>()?;
    let mut stream = Xoshiro256StarStar::seed_from_u64(seed);
    let rngs: Vec<_> = (0..problems.len())
      .map(|_| {
        let rng = stream.clone();
        stream.jump();
        rng
      })
      .collect();
    let pool = match threads {
      0 => None,
      n => Some(
        rayon::ThreadPoolBuilder::new()
          .num_threads(n)
          .build()
          .map_err(|e| PyOSError::new_err(format!("thread pool: {e}")))?,
      ),
    };
    let envs = parallel(py, pool.as_ref(), || {
      problems
        .into_par_iter()
        .zip(rngs)
        .map(|(problem, rng)| Env::new(problem, k_frac, rng))
        .collect()
    });
    Ok(BatchEnv {
      envs,
      horizon,
      pool,
    })
  }

  /// New RSCIM start solutions; iteration counters back to 0.
  fn reset(&mut self, py: Python<'_>) {
    let envs = &mut self.envs;
    parallel(py, self.pool.as_ref(), || {
      envs.par_iter_mut().for_each(Env::reset)
    });
  }

  fn n_vehicles(&self) -> Vec<usize> {
    self.envs.iter().map(|e| e.ctx().n_vehicles()).collect()
  }

  fn n_orders(&self) -> Vec<usize> {
    self.envs.iter().map(|e| e.ctx().n_orders()).collect()
  }

  /// Destroy size per environment: `max(3, round(k_frac * n_orders))`,
  /// capped at `n_orders`.
  fn k(&self) -> Vec<usize> {
    self.envs.iter().map(|e| e.k).collect()
  }

  /// Per environment, per vehicle: the stops as (order index, "pickup" | "delivery").
  fn routes(&self) -> Vec<Vec<Vec<(usize, &'static str)>>> {
    self.envs.iter().map(Env::routes).collect()
  }

  /// Total distance of each environment's current solution.
  fn objective(&self) -> Vec<f64> {
    self.envs.iter().map(Env::objective).collect()
  }

  /// For every environment and removal set, destroy and greedily repair a
  /// copy of the current solution and return `100 * (before - after) /
  /// before`. Deterministic given the state, which is not changed.
  fn evaluate(&self, py: Python<'_>, removals: Vec<Vec<Vec<i64>>>) -> PyResult<Vec<Vec<f64>>> {
    check_count(removals.len(), self.envs.len())?;
    let sets = removals
      .iter()
      .zip(&self.envs)
      .enumerate()
      .map(|(e, (env_sets, env))| {
        env_sets
          .iter()
          .map(|raw| removal_set(raw, env.ctx().n_orders(), e))
          .collect::<PyResult<Vec<_>>>()
      })
      .collect::<PyResult<Vec<_>>>()?;
    let envs = &self.envs;
    Ok(parallel(py, self.pool.as_ref(), || {
      envs
        .par_iter()
        .zip(&sets)
        .map(|(env, env_sets)| env.evaluate(env_sets))
        .collect()
    }))
  }

  /// Apply one removal set per environment with the simulated-annealing
  /// acceptance of `run_lns`, cooling over `horizon` iterations; returns
  /// whether each candidate was accepted. Nothing is applied if any set is
  /// invalid.
  fn step(&mut self, py: Python<'_>, removals: Vec<Vec<i64>>) -> PyResult<Vec<bool>> {
    check_count(removals.len(), self.envs.len())?;
    let sets = removals
      .iter()
      .zip(&self.envs)
      .enumerate()
      .map(|(e, (raw, env))| removal_set(raw, env.ctx().n_orders(), e))
      .collect::<PyResult<Vec<_>>>()?;
    let (envs, horizon) = (&mut self.envs, self.horizon);
    Ok(parallel(py, self.pool.as_ref(), || {
      envs
        .par_iter_mut()
        .zip(&sets)
        .map(|(env, removed)| env.step(removed, horizon))
        .collect()
    }))
  }
}

#[pymodule]
fn vrppd_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<BatchEnv>()
}
