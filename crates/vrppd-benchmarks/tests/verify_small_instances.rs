//! 3-way global-optimum verification on the frozen small-instance fixtures.
//!
//! For each fixture under `fixtures/small-instances/` this test runs:
//!   - `vrppd-brute-force` (oracle, supports all 3 objectives)
//!   - `vrppd-milp`                 (proves Optimal at this scale)
//!   - `vrppd-or-tools::cp_sat`     — TODO: enable when the crate lands
//!
//! and asserts that all three produce the same optimal objective value for
//! `Distance` and `Price`. Any disagreement is a solver bug or model bug, not
//! a tuning issue.
//!
//! Triggered manually via `.github/workflows/verify-small-instances.yml`.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use vrppd_brute_force::solve as bf_solve;
use vrppd_core::{Objective, Problem};
use vrppd_milp::{MilpStatus, solve_milp};

const EPS: f64 = 1e-3;

/// Wall-clock cap per MILP run. At the small-instance scale HiGHS proves
/// optimum in seconds; this is a safety net so a regression doesn't hang CI.
const MILP_BUDGET: Duration = Duration::from_secs(600);

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
      let problem: Problem =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", instance.display()));

      let bf = bf_solve(&problem);
      let bf_distance = bf.best_distance_solution.total_distance;
      let bf_price = bf.best_price_solution.total_price;

      let milp_d = solve_milp(&problem, Objective::Distance, MILP_BUDGET, 1)
        .unwrap_or_else(|e| panic!("MILP Distance on {}: {e}", instance.display()));
      assert_eq!(
        milp_d.status,
        MilpStatus::Optimal,
        "MILP Distance timed out on {} (raise MILP_BUDGET)",
        instance.display()
      );
      assert!(
        (milp_d.objective_value - bf_distance).abs() < EPS,
        "MILP/BF Distance disagree on {}: milp={} bf={}",
        instance.display(),
        milp_d.objective_value,
        bf_distance,
      );

      let milp_p = solve_milp(&problem, Objective::Price, MILP_BUDGET, 1)
        .unwrap_or_else(|e| panic!("MILP Price on {}: {e}", instance.display()));
      assert_eq!(
        milp_p.status,
        MilpStatus::Optimal,
        "MILP Price timed out on {} (raise MILP_BUDGET)",
        instance.display()
      );
      assert!(
        (milp_p.objective_value - bf_price).abs() < EPS,
        "MILP/BF Price disagree on {}: milp={} bf={}",
        instance.display(),
        milp_p.objective_value,
        bf_price,
      );

      // TODO: enable when `vrppd-or-tools::cp_sat` lands.
      //
      //   let cps_d = vrppd_or_tools::solve_cp_sat_default(&problem, Objective::Distance)
      //     .expect("CP-SAT must solve a small instance");
      //   assert_eq!(cps_d.status, OrToolsStatus::Optimal);
      //   assert!((cps_d.objective_value - bf_distance).abs() < EPS);
      //   // ... same for Price.

      verified += 1;
    }
  }

  eprintln!(
    "verified {verified} small-instance fixtures with BF and MILP (CP-SAT assertion pending crate)"
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
