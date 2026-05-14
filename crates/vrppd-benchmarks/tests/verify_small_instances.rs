//! 3-way global-optimum verification on the frozen small-instance fixtures.
//!
//! For each fixture under `fixtures/small-instances/` this test runs:
//!   - `vrppd-brute-force` (oracle, supports all 3 objectives)
//!   - `vrppd-milp`                 — TODO: enable when the crate lands
//!   - `vrppd-or-tools::cp_sat`     — TODO: enable when the crate lands
//!
//! and asserts that all three produce the same optimal objective value for
//! `Distance` and `Price`. Any disagreement is a solver bug or model bug, not
//! a tuning issue.
//!
//! Triggered manually via `.github/workflows/verify-small-instances.yml`.
//! Today the test runs BF on every fixture and leaves the MILP / CP-SAT
//! assertions as commented-out scaffolding.

use std::fs;
use std::path::PathBuf;

use vrppd_brute_force::solve as bf_solve;
use vrppd_core::Problem;

#[allow(dead_code)]
const EPS: f64 = 1e-6;

#[test]
fn verify_small_instances_three_way() {
  let fixtures = workspace_root().join("fixtures/small-instances");
  assert!(
    fixtures.exists(),
    "fixtures dir missing: {}",
    fixtures.display()
  );

  let mut verified = 0usize;
  for cell in read_dir_sorted(&fixtures) {
    if !cell.is_dir() {
      continue;
    }
    for instance in read_dir_sorted(&cell) {
      if instance.extension().and_then(|s| s.to_str()) != Some("json") {
        continue;
      }

      let raw = fs::read_to_string(&instance)
        .unwrap_or_else(|e| panic!("read {}: {e}", instance.display()));
      let problem: Problem = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("parse {}: {e}", instance.display()));

      let bf = bf_solve(&problem);
      let _bf_distance = bf.best_distance_solution.total_distance;
      let _bf_price = bf.best_price_solution.total_price;

      // TODO: enable when `vrppd-milp` lands.
      //
      //   let milp_d = vrppd_milp::solve_milp_default(&problem, Objective::Distance)
      //     .expect("MILP must solve a small instance");
      //   assert_eq!(milp_d.status, MilpStatus::Optimal);
      //   assert!(
      //     (milp_d.objective_value - _bf_distance).abs() < EPS,
      //     "MILP/BF disagree on Distance for {}: milp={} bf={}",
      //     instance.display(), milp_d.objective_value, _bf_distance,
      //   );
      //   // ... same for Price.

      // TODO: enable when `vrppd-or-tools::cp_sat` lands.
      //
      //   let cps_d = vrppd_or_tools::solve_cp_sat_default(&problem, Objective::Distance)
      //     .expect("CP-SAT must solve a small instance");
      //   assert_eq!(cps_d.status, OrToolsStatus::Optimal);
      //   assert!((cps_d.objective_value - _bf_distance).abs() < EPS);
      //   // ... same for Price.

      verified += 1;
    }
  }

  eprintln!(
    "verified {verified} small-instance fixtures with BF (MILP / CP-SAT assertions pending crates)"
  );
  assert!(verified > 0, "no small-instance fixtures found");
}

fn read_dir_sorted(dir: &std::path::Path) -> Vec<PathBuf> {
  let mut entries: Vec<PathBuf> = fs::read_dir(dir)
    .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
    .map(|r| r.unwrap().path())
    .collect();
  entries.sort();
  entries
}

fn workspace_root() -> PathBuf {
  // CARGO_MANIFEST_DIR is the crate dir (crates/vrppd-benchmarks); go up two.
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .unwrap()
    .parent()
    .unwrap()
    .to_path_buf()
}
