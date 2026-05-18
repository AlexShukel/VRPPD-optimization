//! End-to-end CEA quality tests anchored against the brute-force oracle on tiny fixtures.

use std::path::PathBuf;

use vrppd_cea::{solve_cea_seeded, CeaConfig};
use vrppd_core::{Objective, Problem};

fn load_fixture(name: &str) -> Problem {
  let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  path.push("tests/fixtures");
  path.push(name);
  let raw = std::fs::read_to_string(&path).unwrap();
  serde_json::from_str(&raw).unwrap()
}

#[test]
fn cea_matches_brute_force_on_single_order() {
  let problem = load_fixture("single_vehicle_single_order.json");
  let bf = vrppd_brute_force::solve(&problem);
  let bf_dist = bf.best_distance_solution.total_distance;

  for seed in [3_u64, 17, 91] {
    let solved = solve_cea_seeded(
      &problem,
      Objective::Distance,
      CeaConfig::small_for_tests(),
      seed,
    );
    assert!(
      (solved.solution.total_distance - bf_dist).abs() < 1e-9,
      "seed {seed}: CEA total_distance {} != BF {}",
      solved.solution.total_distance,
      bf_dist
    );
  }
}

#[test]
fn cea_finds_brute_force_optimum_on_three_orders() {
  let problem = load_fixture("two_vehicles_three_orders.json");
  let bf = vrppd_brute_force::solve(&problem);

  for target in [Objective::Distance, Objective::Empty, Objective::Price] {
    let bf_energy = target.energy(match target {
      Objective::Distance => &bf.best_distance_solution,
      Objective::Empty => &bf.best_empty_solution,
      Objective::Price => &bf.best_price_solution,
    });

    let mut best_rpd = f64::INFINITY;
    for seed in 0..6_u64 {
      let solved = solve_cea_seeded(&problem, target, CeaConfig::small_for_tests(), seed);
      let cea_energy = target.energy(&solved.solution);
      let rpd = (cea_energy - bf_energy) / bf_energy.max(1e-9);
      best_rpd = best_rpd.min(rpd);
    }
    assert!(
      best_rpd < 0.05,
      "objective {:?}: best RPD across 6 seeds was {:.4}, expected < 5%",
      target,
      best_rpd
    );
  }
}

#[test]
fn cea_history_is_monotone_improving() {
  let problem = load_fixture("two_vehicles_three_orders.json");
  let solved = solve_cea_seeded(
    &problem,
    Objective::Distance,
    CeaConfig::small_for_tests(),
    2026,
  );

  let mut prev = f64::INFINITY;
  for point in &solved.history {
    let e = point.total_distance;
    assert!(
      e <= prev + 1e-9,
      "convergence trace not monotone: {e} > {prev} at gen {}",
      point.generation
    );
    prev = e;
  }
}

#[test]
fn parallel_cea_produces_valid_solution() {
  let problem = load_fixture("two_vehicles_three_orders.json");
  let mut config = CeaConfig::small_for_tests();
  config.threads = 4;
  config.wall_time_cap_ms = Some(5_000);

  for seed in [0_u64, 1, 2] {
    let solved = solve_cea_seeded(&problem, Objective::Distance, config, seed);
    assert!(
      solved.solution.total_distance > 0.0,
      "parallel CEA returned zero distance for seed {seed}"
    );
  }
}
