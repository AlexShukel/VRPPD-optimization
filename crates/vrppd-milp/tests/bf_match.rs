//! Tightness check: the MILP optimum must coincide with the brute-force
//! optimum on the small fixtures, for every supported objective.

use std::fs;
use std::path::{Path, PathBuf};

use vrppd_core::{Objective, Problem};

fn workspace_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .unwrap()
    .parent()
    .unwrap()
    .to_path_buf()
}

fn load_grid_problem(v: usize, n: usize) -> Problem {
  let dir = workspace_root()
    .join("fixtures/small-instances")
    .join(format!("{v}_{n}"));
  let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
    .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
    .filter_map(|r| r.ok().map(|e| e.path()))
    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
    .collect();
  entries.sort();
  let path: &Path = entries
    .first()
    .unwrap_or_else(|| panic!("no fixture in {}", dir.display()));
  let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn check_cell(v: usize, n: usize) {
  let problem = load_grid_problem(v, n);
  let bf = vrppd_brute_force::solve(&problem);

  for (target, bf_optimum) in [
    (
      Objective::Distance,
      bf.best_distance_solution.total_distance,
    ),
    (Objective::Price, bf.best_price_solution.total_price),
  ] {
    let cold = vrppd_milp::solve_milp(&problem, target, std::time::Duration::from_secs(600), 1)
      .unwrap_or_else(|e| panic!("MILP cold v={v} n={n} target={target:?}: {e}"));
    assert_eq!(
      cold.status,
      vrppd_milp::MilpStatus::Optimal,
      "cold MILP timed out on v={v} n={n} target={target:?} (raise the 600s budget or mark this cell #[ignore])"
    );
    assert!(
      (cold.objective_value - bf_optimum).abs() < 1e-3,
      "cold MILP {} != BF {} on v={v} n={n} target={target:?}",
      cold.objective_value,
      bf_optimum
    );
  }
}

macro_rules! grid_cell {
  (run, $v:literal, $n:literal) => {
    paste::paste! {
      #[test]
      fn [<milp_matches_bf_v $v _n $n _cold>]() {
        check_cell($v, $n);
      }
    }
  };
  (ignore, $v:literal, $n:literal) => {
    paste::paste! {
      #[test]
      #[ignore]
      fn [<milp_matches_bf_v $v _n $n _cold>]() {
        check_cell($v, $n);
      }
    }
  };
}

grid_cell!(run, 1, 1);
grid_cell!(run, 1, 2);
grid_cell!(run, 1, 3);
grid_cell!(run, 1, 4);
grid_cell!(run, 1, 5);
grid_cell!(run, 1, 6);
grid_cell!(run, 1, 7);

grid_cell!(run, 2, 1);
grid_cell!(run, 2, 2);
grid_cell!(run, 2, 3);
grid_cell!(run, 2, 4);
grid_cell!(run, 2, 5);
grid_cell!(run, 2, 6);
grid_cell!(ignore, 2, 7);

grid_cell!(run, 3, 1);
grid_cell!(run, 3, 2);
grid_cell!(run, 3, 3);
grid_cell!(run, 3, 4);
grid_cell!(run, 3, 5);
grid_cell!(ignore, 3, 6);
grid_cell!(ignore, 3, 7);

grid_cell!(run, 4, 1);
grid_cell!(run, 4, 2);
grid_cell!(run, 4, 3);
grid_cell!(run, 4, 4);
grid_cell!(run, 4, 5);
grid_cell!(ignore, 4, 6);
grid_cell!(ignore, 4, 7);

grid_cell!(run, 5, 1);
grid_cell!(run, 5, 2);
grid_cell!(run, 5, 3);
grid_cell!(run, 5, 4);
grid_cell!(run, 5, 5);
grid_cell!(ignore, 5, 6);
grid_cell!(ignore, 5, 7);

grid_cell!(run, 6, 1);
grid_cell!(run, 6, 2);
grid_cell!(run, 6, 3);
grid_cell!(run, 6, 4);
grid_cell!(ignore, 6, 5);
grid_cell!(ignore, 6, 6);
grid_cell!(ignore, 6, 7);

grid_cell!(run, 7, 1);
grid_cell!(run, 7, 2);
grid_cell!(run, 7, 3);
grid_cell!(ignore, 7, 4);
grid_cell!(ignore, 7, 5);
grid_cell!(ignore, 7, 6);
grid_cell!(ignore, 7, 7);
