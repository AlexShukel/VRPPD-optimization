//! Reproduction operator (elitism) shared by both populations.

use vrppd_core::Objective;

use crate::population::{Individual, Population};

pub fn reproduce_elite(pop: &Population, target: Objective) -> Option<Individual> {
  pop.best(target).cloned()
}

#[cfg(test)]
mod tests {
  use super::*;
  use vrppd_core::WorkingSolution;

  fn ind(d: f64) -> Individual {
    let mut sol = WorkingSolution::empty(0);
    sol.total_distance = d;
    sol.empty_distance = d;
    sol.total_price = d;
    Individual { solution: sol }
  }

  #[test]
  fn reproduce_elite_returns_minimum_energy_individual() {
    let pop = Population::new(vec![ind(20.0), ind(10.0), ind(15.0)]);
    let elite = reproduce_elite(&pop, Objective::Distance).unwrap();
    assert_eq!(elite.solution.total_distance, 10.0);
  }

  #[test]
  fn reproduce_elite_on_empty_returns_none() {
    let pop = Population::new(vec![]);
    assert!(reproduce_elite(&pop, Objective::Distance).is_none());
  }
}
