//! Rank-based fitness function and roulette-wheel sampling.

use rand::Rng;

use vrppd_core::Objective;

use crate::population::Individual;

pub type Fitness = Vec<f64>;

pub fn fitness_values(individuals: &[Individual], target: Objective) -> Fitness {
  let n = individuals.len();
  let total_n = n;
  let mut indexed: Vec<(usize, f64)> = individuals
    .iter()
    .enumerate()
    .map(|(i, ind)| (i, energy(ind, target)))
    .collect();

  indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

  let high = 2 * total_n + 1;

  let mut out = vec![0.0; n];
  for (rank0, (orig_idx, _)) in indexed.iter().enumerate() {
    let rank1 = rank0 + 1;
    let f = (high as i64 - rank1 as i64) as f64;
    out[*orig_idx] = f.max(1.0);
  }
  out
}

pub fn roulette_select<R: Rng + ?Sized>(
  fitness: &Fitness,
  count: usize,
  rng: &mut R,
) -> Vec<usize> {
  assert!(
    count <= fitness.len(),
    "cannot draw more than population size"
  );

  let mut available: Vec<(usize, f64)> = fitness.iter().enumerate().map(|(i, &f)| (i, f)).collect();

  let mut picked = Vec::with_capacity(count);
  for _ in 0..count {
    let total: f64 = available.iter().map(|(_, f)| *f).sum();
    if total <= 0.0 {
      let idx = rng.gen_range(0..available.len());
      picked.push(available.remove(idx).0);
      continue;
    }
    let r: f64 = rng.r#gen::<f64>() * total;
    let mut acc = 0.0;
    let mut chosen_pos = available.len() - 1;
    for (pos, (_, f)) in available.iter().enumerate() {
      acc += *f;
      if r <= acc {
        chosen_pos = pos;
        break;
      }
    }
    picked.push(available.remove(chosen_pos).0);
  }
  picked
}

#[inline]
fn energy(ind: &Individual, target: Objective) -> f64 {
  match target {
    Objective::Empty => ind.solution.empty_distance,
    Objective::Distance => ind.solution.total_distance,
    Objective::Price => ind.solution.total_price,
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;
  use vrppd_core::WorkingSolution;

  fn ind_with_distance(d: f64) -> Individual {
    let mut sol = WorkingSolution::empty(0);
    sol.total_distance = d;
    sol.empty_distance = d;
    sol.total_price = d;
    Individual { solution: sol }
  }

  #[test]
  fn fitness_ranks_min_energy_highest() {
    let inds = vec![
      ind_with_distance(10.0),
      ind_with_distance(20.0),
      ind_with_distance(15.0),
    ];
    let f = fitness_values(&inds, Objective::Distance);
    assert_eq!(f, vec![6.0, 4.0, 5.0]);
  }

  #[test]
  fn roulette_picks_each_index_at_most_once() {
    let f = vec![5.0, 4.0, 3.0, 2.0];
    let mut rng = Xoshiro256StarStar::seed_from_u64(42);
    let picked = roulette_select(&f, 4, &mut rng);
    let mut sorted = picked.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3], "picked = {picked:?}");
  }

  #[test]
  fn roulette_favours_higher_fitness() {
    let f = vec![100.0, 1.0, 1.0, 1.0];
    let mut rng = Xoshiro256StarStar::seed_from_u64(7);
    let mut hits = 0;
    for _ in 0..1000 {
      let p = roulette_select(&f, 1, &mut rng);
      if p[0] == 0 {
        hits += 1;
      }
    }
    assert!(
      hits > 900,
      "expected high-fitness index dominant; got {hits}/1000"
    );
  }
}
