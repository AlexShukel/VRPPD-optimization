//! Single-threaded simulated-annealing driver.

use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::time::Instant;

use vrppd_core::{Objective, Problem, ProblemSolution};
use vrppd_core::{OrderMatrix, VehicleStartMatrix, WorkingSolution};

use crate::config::SaConfig;
use crate::operators::generate_neighbor;
use crate::rcrs::generate_rcrs;

#[derive(Clone, Debug)]
pub struct ConvergencePoint {
  pub time_ms: f64,
  pub iteration: u64,
  pub total_distance: f64,
  pub empty_distance: f64,
  pub total_price: f64,
}

impl ConvergencePoint {
  pub(crate) fn from_solution(time_ms: f64, iteration: u64, sol: &WorkingSolution) -> Self {
    Self {
      time_ms,
      iteration,
      total_distance: sol.total_distance,
      empty_distance: sol.empty_distance,
      total_price: sol.total_price,
    }
  }
}

#[derive(Clone, Debug)]
pub struct Solved {
  pub solution: ProblemSolution,
  pub history: Vec<ConvergencePoint>,
}

pub fn solve(problem: &Problem, target: Objective, config: SaConfig) -> Solved {
  let mut rng = thread_rng();
  let seed: u64 = rng.r#gen();
  solve_seeded(problem, target, config, seed)
}

pub fn solve_seeded(problem: &Problem, target: Objective, config: SaConfig, seed: u64) -> Solved {
  let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

  let order_mat = OrderMatrix::build(&problem.orders);
  let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

  let current = generate_rcrs(problem, &order_mat, &vstart_mat, target, &mut rng);
  let mut history = Vec::new();
  history.push(ConvergencePoint::from_solution(0.0, 0, &current));

  let started = Instant::now();
  let best = anneal(
    current,
    problem,
    &order_mat,
    &vstart_mat,
    target,
    &config,
    &mut rng,
    started,
    &mut history,
  );

  Solved {
    solution: best.into_problem_solution(problem),
    history,
  }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn anneal<R: Rng + ?Sized>(
  initial: WorkingSolution,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &SaConfig,
  rng: &mut R,
  started: Instant,
  history: &mut Vec<ConvergencePoint>,
) -> WorkingSolution {
  let mut current = initial;
  let mut current_energy = target.energy_working(&current);
  let mut best = current.clone();
  let mut best_energy = current_energy;
  let mut temperature = config.initial_temp;

  for iter in 1..=config.max_iterations {
    if temperature < config.min_temp {
      break;
    }

    if let Some(neighbor) = generate_neighbor(
      &current,
      problem,
      order_mat,
      vstart_mat,
      config.weights,
      rng,
    ) {
      let neighbor_energy = target.energy_working(&neighbor);
      let delta = neighbor_energy - current_energy;

      let accept = delta < 0.0 || {
        let p: f64 = rng.r#gen();
        p < (-delta / temperature).exp()
      };

      if accept {
        current = neighbor;
        current_energy = neighbor_energy;

        if current_energy < best_energy {
          best_energy = current_energy;
          best = current.clone();
          history.push(ConvergencePoint::from_solution(
            started.elapsed().as_secs_f64() * 1_000.0,
            iter,
            &best,
          ));
        }
      }
    }

    temperature *= config.cooling_rate;
  }

  best
}

trait WorkingEnergy {
  fn energy_working(self, sol: &WorkingSolution) -> f64;
}

impl WorkingEnergy for Objective {
  #[inline(always)]
  fn energy_working(self, sol: &WorkingSolution) -> f64 {
    match self {
      Objective::Empty => sol.empty_distance,
      Objective::Distance => sol.total_distance,
      Objective::Price => sol.total_price,
    }
  }
}
