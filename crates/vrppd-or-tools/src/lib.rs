//! OR-Tools CP-SAT baseline for the adapted VRPPD. Shells out to a Python
//! subprocess (`python/solver.py`) that runs `ortools.sat.python.cp_model`.

mod wire;

use std::io::{ErrorKind, Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use vrppd_core::{Objective, Problem};

use crate::wire::{SolverRequest, SolverResponse, WireOrder, WireProblem, WireVehicle};

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug)]
pub enum OrToolsError {
  UnsupportedObjective(Objective),
  PythonNotFound,
  OrtoolsImportFailed(String),
  SolverFailed(String),
  SolverInternal(String),
  Infeasible,
}

impl std::fmt::Display for OrToolsError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      OrToolsError::UnsupportedObjective(o) => {
        write!(f, "OR-Tools does not support objective {o:?}")
      }
      OrToolsError::PythonNotFound => write!(f, "python3 not found on PATH"),
      OrToolsError::OrtoolsImportFailed(msg) => write!(f, "ortools import failed: {msg}"),
      OrToolsError::SolverFailed(msg) => write!(f, "OR-Tools solver failed: {msg}"),
      OrToolsError::SolverInternal(msg) => write!(f, "OR-Tools internal error: {msg}"),
      OrToolsError::Infeasible => write!(f, "Model is infeasible"),
    }
  }
}

impl std::error::Error for OrToolsError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrToolsStatus {
  Optimal,
  Feasible,
  TimedOut,
}

#[derive(Clone, Copy, Debug)]
pub struct OrToolsResult {
  pub objective_value: f64,
  pub status: OrToolsStatus,
  pub solve_time_ms: u64,
}

pub fn solve_cp_sat(
  problem: &Problem,
  target: Objective,
  timeout: Duration,
  threads: usize,
) -> Result<OrToolsResult, OrToolsError> {
  if matches!(target, Objective::Empty) {
    return Err(OrToolsError::UnsupportedObjective(target));
  }
  if problem.orders.is_empty() || problem.vehicles.is_empty() {
    return Ok(OrToolsResult {
      objective_value: 0.0,
      status: OrToolsStatus::Optimal,
      solve_time_ms: 0,
    });
  }

  let started = Instant::now();
  let objective_str = match target {
    Objective::Distance => "DISTANCE",
    Objective::Price => "PRICE",
    Objective::Empty => unreachable!("guarded above"),
  };

  let request = SolverRequest {
    objective: objective_str,
    timeout_ms: timeout.as_millis() as u64,
    threads: threads.max(1),
    problem: WireProblem {
      vehicles: problem
        .vehicles
        .iter()
        .map(WireVehicle::from_core)
        .collect(),
      orders: problem.orders.iter().map(WireOrder::from_core).collect(),
    },
  };

  let response = run_python(&request)?;
  let solve_time_ms = response
    .solver_runtime_ms
    .unwrap_or(started.elapsed().as_millis() as u64);

  if !response.ok {
    let kind = response.error_kind.as_deref().unwrap_or("");
    let msg = response.error_msg.unwrap_or_default();
    return Err(match kind {
      "ortools_import" => OrToolsError::OrtoolsImportFailed(msg),
      "invalid_request" => OrToolsError::SolverFailed(msg),
      _ => OrToolsError::SolverInternal(msg),
    });
  }

  let status_str = response.status.as_deref().unwrap_or("");
  let status = match status_str {
    "OPTIMAL" => OrToolsStatus::Optimal,
    "FEASIBLE" => OrToolsStatus::Feasible,
    "TIMED_OUT" => OrToolsStatus::TimedOut,
    "INFEASIBLE" => return Err(OrToolsError::Infeasible),
    "FAILED" => return Err(OrToolsError::SolverFailed("FAILED".into())),
    other => {
      return Err(OrToolsError::SolverFailed(format!(
        "unknown status: {other}"
      )));
    }
  };

  let objective_value = response.objective_value.unwrap_or(0.0).max(0.0);
  Ok(OrToolsResult {
    objective_value,
    status,
    solve_time_ms,
  })
}

pub fn solve_cp_sat_default(
  problem: &Problem,
  target: Objective,
) -> Result<OrToolsResult, OrToolsError> {
  let threads = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(1);
  solve_cp_sat(problem, target, DEFAULT_TIMEOUT, threads)
}

fn script_path() -> String {
  std::env::var("VRPPD_ORTOOLS_PY")
    .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/python/solver.py").to_string())
}

fn python_bin() -> String {
  if let Ok(v) = std::env::var("VRPPD_PYTHON3") {
    return v;
  }
  let workspace_venv = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.venv/bin/python3");
  if std::path::Path::new(workspace_venv).exists() {
    return workspace_venv.to_string();
  }
  "python3".to_string()
}

fn run_python(request: &SolverRequest) -> Result<SolverResponse, OrToolsError> {
  let script = script_path();
  let python = python_bin();
  let mut child = match Command::new(&python)
    .arg(&script)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
  {
    Ok(c) => c,
    Err(e) if e.kind() == ErrorKind::NotFound => return Err(OrToolsError::PythonNotFound),
    Err(e) => return Err(OrToolsError::SolverFailed(format!("spawn {python}: {e}"))),
  };

  {
    let stdin = child
      .stdin
      .as_mut()
      .ok_or_else(|| OrToolsError::SolverFailed("python3 stdin not piped".into()))?;
    let json = serde_json::to_vec(request)
      .map_err(|e| OrToolsError::SolverFailed(format!("serialize request: {e}")))?;
    stdin
      .write_all(&json)
      .map_err(|e| OrToolsError::SolverFailed(format!("write stdin: {e}")))?;
  }
  drop(child.stdin.take());

  let mut stdout = String::new();
  if let Some(mut s) = child.stdout.take() {
    s.read_to_string(&mut stdout)
      .map_err(|e| OrToolsError::SolverFailed(format!("read stdout: {e}")))?;
  }
  let mut stderr = String::new();
  if let Some(mut s) = child.stderr.take() {
    let _ = s.read_to_string(&mut stderr);
  }

  let status = child
    .wait()
    .map_err(|e| OrToolsError::SolverFailed(format!("wait: {e}")))?;

  match serde_json::from_str(&stdout) {
    Ok(v) => Ok(v),
    Err(e) => {
      let exit = status
        .code()
        .map(|c| c.to_string())
        .unwrap_or_else(|| "signal".into());
      Err(OrToolsError::SolverFailed(format!(
        "parse stdout (exit={exit}): {e}; stdout={stdout:?}; stderr={stderr:?}"
      )))
    }
  }
}

#[cfg(test)]
mod shortcircuit_tests {
  use super::*;
  use vrppd_core::{Location, Objective, Order, Problem, Vehicle};

  fn loc(lat: f64, lon: f64) -> Location {
    Location {
      hash: format!("{lat},{lon}"),
      latitude: lat,
      longitude: lon,
    }
  }

  fn one_vehicle_one_order() -> Problem {
    Problem {
      vehicles: vec![Vehicle {
        id: 1,
        start_location: loc(0.0, 0.0),
        price_km: 1.0,
      }],
      orders: vec![Order {
        id: 1,
        pickup_location: loc(0.0, 0.0),
        delivery_location: loc(0.0, 0.5),
        load_factor: 1.0,
      }],
    }
  }

  #[test]
  fn empty_problem_returns_zero_optimal() {
    let p = Problem {
      vehicles: vec![],
      orders: vec![],
    };
    let r = solve_cp_sat(&p, Objective::Distance, Duration::from_secs(1), 1).unwrap();
    assert_eq!(r.objective_value, 0.0);
    assert_eq!(r.status, OrToolsStatus::Optimal);
    assert_eq!(r.solve_time_ms, 0);
  }

  #[test]
  fn empty_objective_rejected() {
    let p = one_vehicle_one_order();
    let err = solve_cp_sat(&p, Objective::Empty, Duration::from_secs(1), 1).unwrap_err();
    assert!(matches!(
      err,
      OrToolsError::UnsupportedObjective(Objective::Empty)
    ));
  }
}
