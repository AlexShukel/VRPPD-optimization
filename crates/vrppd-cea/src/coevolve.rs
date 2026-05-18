//! Top-level CEA evolution loop with elitism, recombination, crossover, and local improvement.

use std::time::Instant;

use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

use vrppd_core::{
  Objective, OrderMatrix, Problem, ProblemSolution, VehicleStartMatrix, WorkingSolution,
};

use crate::config::CeaConfig;
use crate::crossover::crossover;
use crate::fitness::{fitness_values, roulette_select};
use crate::local_improvement::local_improve;
use crate::population::{Individual, Population};
use crate::recombination::recombine;
use crate::reproduction::reproduce_elite;
use crate::rscim::generate_rscim;

#[derive(Clone, Debug)]
pub struct ConvergencePoint {
  pub time_ms: f64,
  pub generation: u64,
  pub total_distance: f64,
  pub empty_distance: f64,
  pub total_price: f64,
}

impl ConvergencePoint {
  fn from_solution(time_ms: f64, generation: u64, sol: &WorkingSolution) -> Self {
    Self {
      time_ms,
      generation,
      total_distance: sol.total_distance,
      empty_distance: sol.empty_distance,
      total_price: sol.total_price,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExitReason {
  Converged,
  WallTimeCap,
}

#[derive(Clone, Debug)]
pub struct Solved {
  pub solution: ProblemSolution,
  pub history: Vec<ConvergencePoint>,
  pub generations: u64,
  pub exit_reason: ExitReason,
}

pub fn solve_cea(problem: &Problem, target: Objective, config: CeaConfig) -> Solved {
  let mut rng = thread_rng();
  let seed: u64 = rng.r#gen();
  solve_cea_seeded(problem, target, config, seed)
}

pub fn solve_cea_seeded(
  problem: &Problem,
  target: Objective,
  config: CeaConfig,
  seed: u64,
) -> Solved {
  let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

  let order_mat = OrderMatrix::build(&problem.orders);
  let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

  let n = config.population_size.max(1);
  let mut pop1 = Population::new(
    (0..n)
      .map(|_| {
        Individual::new(generate_rscim(
          problem,
          &order_mat,
          &vstart_mat,
          target,
          &mut rng,
        ))
      })
      .collect(),
  );
  let mut pop2 = pop1.clone();

  let mut best = best_individual(&pop1, &pop2, target)
    .expect("non-empty population")
    .clone();
  let mut best_energy = energy(&best.solution, target);

  let started = Instant::now();
  let mut history = vec![ConvergencePoint::from_solution(0.0, 0, &best.solution)];

  let mut stagnant: usize = 0;
  let mut generation: u64 = 0;
  let exit_reason;

  loop {
    if stagnant >= config.conv_count {
      exit_reason = ExitReason::Converged;
      break;
    }
    if let Some(cap) = config.wall_time_cap_ms {
      if started.elapsed().as_millis() as u64 >= cap {
        exit_reason = ExitReason::WallTimeCap;
        break;
      }
    }

    generation += 1;

    pop1 = evolve_pop1(
      &pop1,
      problem,
      &order_mat,
      &vstart_mat,
      target,
      &config,
      &mut rng,
    );
    pop2 = evolve_pop2(
      &pop2,
      &pop1,
      problem,
      &order_mat,
      &vstart_mat,
      target,
      &config,
      &mut rng,
    );

    let candidate = best_individual(&pop1, &pop2, target)
      .expect("non-empty population")
      .clone();
    let candidate_energy = energy(&candidate.solution, target);

    if candidate_energy + 1e-12 < best_energy {
      best = candidate;
      best_energy = candidate_energy;
      stagnant = 0;
      history.push(ConvergencePoint::from_solution(
        started.elapsed().as_secs_f64() * 1_000.0,
        generation,
        &best.solution,
      ));
    } else {
      stagnant += 1;
    }
  }

  Solved {
    solution: best.solution.into_problem_solution(problem),
    history,
    generations: generation,
    exit_reason,
  }
}

#[inline]
fn energy(sol: &WorkingSolution, target: Objective) -> f64 {
  match target {
    Objective::Empty => sol.empty_distance,
    Objective::Distance => sol.total_distance,
    Objective::Price => sol.total_price,
  }
}

fn best_individual<'a>(
  pop1: &'a Population,
  pop2: &'a Population,
  target: Objective,
) -> Option<&'a Individual> {
  match (pop1.best(target), pop2.best(target)) {
    (Some(a), Some(b)) => {
      if energy(&a.solution, target) <= energy(&b.solution, target) {
        Some(a)
      } else {
        Some(b)
      }
    }
    (Some(a), None) => Some(a),
    (None, Some(b)) => Some(b),
    (None, None) => None,
  }
}

fn evolve_pop1<R: Rng + ?Sized>(
  pop: &Population,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &CeaConfig,
  rng: &mut R,
) -> Population {
  if config.threads > 1 {
    evolve_pop1_parallel(pop, problem, order_mat, vstart_mat, target, config, rng)
  } else {
    evolve_pop1_sequential(pop, problem, order_mat, vstart_mat, target, config, rng)
  }
}

fn evolve_pop1_sequential<R: Rng + ?Sized>(
  pop: &Population,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &CeaConfig,
  rng: &mut R,
) -> Population {
  let n = pop.len();
  let target_offspring = 2 * n;

  let mut offspring: Vec<Individual> = Vec::with_capacity(target_offspring);
  if let Some(elite) = reproduce_elite(pop, target) {
    offspring.push(elite);
  }

  let parent_fitness = fitness_values(&pop.individuals, target);

  while offspring.len() < target_offspring {
    let picks = roulette_select(&parent_fitness, 1, rng);
    let parent = &pop.individuals[picks[0]];
    let child = recombine(
      &parent.solution,
      problem,
      order_mat,
      vstart_mat,
      target,
      config.recombination_fraction_low,
      config.recombination_fraction_high,
      rng,
    );
    offspring.push(Individual::new(child));
  }

  survive_n(offspring, n, target, rng)
}

fn evolve_pop1_parallel<R: Rng + ?Sized>(
  pop: &Population,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &CeaConfig,
  rng: &mut R,
) -> Population {
  let n = pop.len();
  let target_offspring = 2 * n;

  let mut offspring: Vec<Individual> = Vec::with_capacity(target_offspring);
  if let Some(elite) = reproduce_elite(pop, target) {
    offspring.push(elite);
  }

  let remaining = target_offspring - offspring.len();
  let parent_fitness = fitness_values(&pop.individuals, target);

  let base: u64 = rng.r#gen();

  let parallel_children: Vec<Individual> = (0..remaining)
    .into_par_iter()
    .map(|i| {
      let mut local_rng = Xoshiro256StarStar::seed_from_u64(base.wrapping_add(i as u64));
      let picks = roulette_select(&parent_fitness, 1, &mut local_rng);
      let parent = &pop.individuals[picks[0]];
      let child = recombine(
        &parent.solution,
        problem,
        order_mat,
        vstart_mat,
        target,
        config.recombination_fraction_low,
        config.recombination_fraction_high,
        &mut local_rng,
      );
      Individual::new(child)
    })
    .collect();

  offspring.extend(parallel_children);
  survive_n(offspring, n, target, rng)
}

#[allow(clippy::too_many_arguments)]
fn evolve_pop2<R: Rng + ?Sized>(
  pop2: &Population,
  pop1: &Population,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &CeaConfig,
  rng: &mut R,
) -> Population {
  if config.threads > 1 {
    evolve_pop2_parallel(
      pop2, pop1, problem, order_mat, vstart_mat, target, config, rng,
    )
  } else {
    evolve_pop2_sequential(
      pop2, pop1, problem, order_mat, vstart_mat, target, config, rng,
    )
  }
}

#[allow(clippy::too_many_arguments)]
fn evolve_pop2_sequential<R: Rng + ?Sized>(
  pop2: &Population,
  pop1: &Population,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &CeaConfig,
  rng: &mut R,
) -> Population {
  let n = pop2.len();
  let target_offspring = 2 * n;

  let mut offspring: Vec<Individual> = Vec::with_capacity(target_offspring);
  if let Some(elite) = reproduce_elite(pop2, target) {
    offspring.push(elite);
  }
  if let Some(migrant) = reproduce_elite(pop1, target) {
    offspring.push(migrant);
  }

  let parent_fitness = fitness_values(&pop2.individuals, target);

  while offspring.len() < target_offspring {
    let r: f64 = rng.r#gen();
    if r < config.p_crossover && pop2.len() >= 2 {
      let picks = roulette_select(&parent_fitness, 2, rng);
      let p1 = &pop2.individuals[picks[0]].solution;
      let p2 = &pop2.individuals[picks[1]].solution;
      let child = crossover(p1, p2, problem, order_mat, vstart_mat, target, rng);
      offspring.push(Individual::new(child));
    } else {
      let picks = roulette_select(&parent_fitness, 1, rng);
      let parent = &pop2.individuals[picks[0]];
      let mut child = parent.solution.clone();
      local_improve(
        &mut child,
        problem,
        order_mat,
        vstart_mat,
        target,
        config.p_reinsertion,
        rng,
      );
      offspring.push(Individual::new(child));
    }
  }

  survive_n(offspring, n, target, rng)
}

#[allow(clippy::too_many_arguments)]
fn evolve_pop2_parallel<R: Rng + ?Sized>(
  pop2: &Population,
  pop1: &Population,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  config: &CeaConfig,
  rng: &mut R,
) -> Population {
  let n = pop2.len();
  let target_offspring = 2 * n;

  let mut offspring: Vec<Individual> = Vec::with_capacity(target_offspring);
  if let Some(elite) = reproduce_elite(pop2, target) {
    offspring.push(elite);
  }
  if let Some(migrant) = reproduce_elite(pop1, target) {
    offspring.push(migrant);
  }

  let remaining = target_offspring - offspring.len();
  let parent_fitness = fitness_values(&pop2.individuals, target);
  let p_crossover = config.p_crossover;
  let p_reinsertion = config.p_reinsertion;
  let can_cross = pop2.len() >= 2;
  let individuals = &pop2.individuals;

  let base: u64 = rng.r#gen();

  let parallel_children: Vec<Individual> = (0..remaining)
    .into_par_iter()
    .map(|i| {
      let mut local_rng = Xoshiro256StarStar::seed_from_u64(base.wrapping_add(i as u64));
      let r: f64 = local_rng.r#gen();
      if r < p_crossover && can_cross {
        let picks = roulette_select(&parent_fitness, 2, &mut local_rng);
        let p1 = &individuals[picks[0]].solution;
        let p2 = &individuals[picks[1]].solution;
        let child = crossover(
          p1,
          p2,
          problem,
          order_mat,
          vstart_mat,
          target,
          &mut local_rng,
        );
        Individual::new(child)
      } else {
        let picks = roulette_select(&parent_fitness, 1, &mut local_rng);
        let parent = &individuals[picks[0]];
        let mut child = parent.solution.clone();
        local_improve(
          &mut child,
          problem,
          order_mat,
          vstart_mat,
          target,
          p_reinsertion,
          &mut local_rng,
        );
        Individual::new(child)
      }
    })
    .collect();

  offspring.extend(parallel_children);
  survive_n(offspring, n, target, rng)
}

fn survive_n<R: Rng + ?Sized>(
  offspring: Vec<Individual>,
  n: usize,
  target: Objective,
  rng: &mut R,
) -> Population {
  if offspring.len() <= n {
    return Population::new(offspring);
  }

  let pop = Population::new(offspring);
  let elite_idx = pop.best_index(target).unwrap();

  let fitness = fitness_values(&pop.individuals, target);

  let mut masked = fitness.clone();
  masked[elite_idx] = 0.0;

  let mut picks = roulette_select(&masked, n.saturating_sub(1), rng);
  picks.insert(0, elite_idx);

  Population::new(
    picks
      .into_iter()
      .map(|i| pop.individuals[i].clone())
      .collect(),
  )
}
