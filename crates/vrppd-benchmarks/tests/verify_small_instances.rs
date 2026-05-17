//! 3-way global-optimum verification on the frozen small-instance fixtures.

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use vrppd_brute_force::solve as bf_solve;
use vrppd_core::{Objective, Problem};
use vrppd_milp::{solve_milp_with_warm_start, MilpStatus};
use vrppd_or_tools::{solve_cp_sat, OrToolsStatus};

const EPS: f64 = 1e-3;
const MILP_BUDGET: Duration = Duration::from_secs(120);
const CP_SAT_BUDGET: Duration = Duration::from_secs(120);

#[test]
fn verify_small_instances_three_way() {
  let fixtures = workspace_root().join("fixtures/small-instances");
  assert!(
    fixtures.exists(),
    "fixtures dir missing: {}",
    fixtures.display()
  );

  let cp_sat_threads = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(1);

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

      let milp_d = solve_milp_with_warm_start(
        &problem,
        Objective::Distance,
        MILP_BUDGET,
        1,
        &bf.best_distance_solution,
      )
      .unwrap_or_else(|e| panic!("MILP Distance on {}: {e}", instance.display()));
      assert_eq!(
        milp_d.status,
        MilpStatus::Optimal,
        "MILP Distance did not prove optimum on {} given BF warm-start (raise MILP_BUDGET)",
        instance.display()
      );
      assert!(
        (milp_d.objective_value - bf_distance).abs() < EPS,
        "MILP/BF Distance disagree on {}: milp={} bf={}",
        instance.display(),
        milp_d.objective_value,
        bf_distance,
      );

      let milp_p = solve_milp_with_warm_start(
        &problem,
        Objective::Price,
        MILP_BUDGET,
        1,
        &bf.best_price_solution,
      )
      .unwrap_or_else(|e| panic!("MILP Price on {}: {e}", instance.display()));
      assert_eq!(
        milp_p.status,
        MilpStatus::Optimal,
        "MILP Price did not prove optimum on {} given BF warm-start (raise MILP_BUDGET)",
        instance.display()
      );
      assert!(
        (milp_p.objective_value - bf_price).abs() < EPS,
        "MILP/BF Price disagree on {}: milp={} bf={}",
        instance.display(),
        milp_p.objective_value,
        bf_price,
      );

      let cps_d = solve_cp_sat(&problem, Objective::Distance, CP_SAT_BUDGET, cp_sat_threads)
        .unwrap_or_else(|e| panic!("CP-SAT Distance on {}: {e}", instance.display()));
      assert_eq!(
        cps_d.status,
        OrToolsStatus::Optimal,
        "CP-SAT Distance did not prove optimum on {} (status={:?})",
        instance.display(),
        cps_d.status,
      );
      assert!(
        (cps_d.objective_value - bf_distance).abs() < EPS,
        "CP-SAT/BF Distance disagree on {}: cp_sat={} bf={}",
        instance.display(),
        cps_d.objective_value,
        bf_distance,
      );

      let cps_p = solve_cp_sat(&problem, Objective::Price, CP_SAT_BUDGET, cp_sat_threads)
        .unwrap_or_else(|e| panic!("CP-SAT Price on {}: {e}", instance.display()));
      assert_eq!(
        cps_p.status,
        OrToolsStatus::Optimal,
        "CP-SAT Price did not prove optimum on {} (status={:?})",
        instance.display(),
        cps_p.status,
      );
      assert!(
        (cps_p.objective_value - bf_price).abs() < EPS,
        "CP-SAT/BF Price disagree on {}: cp_sat={} bf={}",
        instance.display(),
        cps_p.objective_value,
        bf_price,
      );

      verified += 1;
    }
  }

  eprintln!("verified {verified} small-instance fixtures with BF, MILP (BF-warm-started), and CP-SAT");
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
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .unwrap()
    .parent()
    .unwrap()
    .to_path_buf()
}
