//! Exact MILP solver for the adapted VRPPD variant.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use highs::{HighsModelStatus, RowProblem, Sense};
use vrppd_core::{Objective, Problem, haversine_km};

mod warm_start;

/// 30-minute default budget per instance.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Debug)]
pub enum MilpError {
  SolverFailed(String),
  Infeasible,
  UnsupportedObjective(Objective),
}

impl std::fmt::Display for MilpError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      MilpError::SolverFailed(msg) => write!(f, "HiGHS failed: {msg}"),
      MilpError::Infeasible => write!(f, "MILP is infeasible"),
      MilpError::UnsupportedObjective(o) => {
        write!(f, "MILP for objective {o:?} is not supported")
      }
    }
  }
}

impl std::error::Error for MilpError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MilpStatus {
  /// HiGHS proved the returned `objective_value` is optimal.
  Optimal,
  /// The wall-clock budget elapsed before optimality was proven.
  TimedOut,
}

#[derive(Clone, Copy, Debug)]
pub struct MilpResult {
  pub objective_value: f64,
  pub status: MilpStatus,
  pub solve_time_ms: u64,
}

pub fn solve_milp(
  problem: &Problem,
  target: Objective,
  timeout: Duration,
  threads: usize,
) -> Result<MilpResult, MilpError> {
  if target == Objective::Empty {
    return Err(MilpError::UnsupportedObjective(target));
  }
  if problem.orders.is_empty() || problem.vehicles.is_empty() {
    return Ok(MilpResult {
      objective_value: 0.0,
      status: MilpStatus::Optimal,
      solve_time_ms: 0,
    });
  }

  let started = Instant::now();
  let model = build_milp(problem, target);
  let mut hm = model.problem.optimise(Sense::Minimise);
  hm.set_option("time_limit", timeout.as_secs_f64());
  hm.set_option("output_flag", false);
  hm.set_option("threads", threads.max(1) as i32);

  let solved = hm.solve();
  let status = match solved.status() {
    HighsModelStatus::Optimal => MilpStatus::Optimal,
    HighsModelStatus::ReachedTimeLimit => MilpStatus::TimedOut,
    HighsModelStatus::Infeasible => return Err(MilpError::Infeasible),
    other => return Err(MilpError::SolverFailed(format!("status={other:?}"))),
  };

  let solution = solved.get_solution();
  let mut z = 0.0;
  for (col, coef) in &model.objective_coeffs {
    z += coef * solution[*col];
  }
  Ok(MilpResult {
    objective_value: z.max(0.0),
    status,
    solve_time_ms: started.elapsed().as_millis() as u64,
  })
}

/// Convenience wrapper using `DEFAULT_TIMEOUT` (30 minutes) and all available threads.
pub fn solve_milp_default(problem: &Problem, target: Objective) -> Result<MilpResult, MilpError> {
  let threads = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(1);
  solve_milp(problem, target, DEFAULT_TIMEOUT, threads)
}

pub fn solve_milp_with_warm_start(
  problem: &Problem,
  target: Objective,
  timeout: Duration,
  threads: usize,
  warm_start: &vrppd_core::ProblemSolution,
) -> Result<MilpResult, MilpError> {
  if target == Objective::Empty {
    return Err(MilpError::UnsupportedObjective(target));
  }
  if problem.orders.is_empty() || problem.vehicles.is_empty() {
    return Ok(MilpResult {
      objective_value: 0.0,
      status: MilpStatus::Optimal,
      solve_time_ms: 0,
    });
  }

  let started = Instant::now();
  let model = build_milp(problem, target);
  let mut hm = model.problem.optimise(Sense::Minimise);
  hm.set_option("time_limit", timeout.as_secs_f64());
  hm.set_option("output_flag", false);
  hm.set_option("threads", threads.max(1) as i32);

  let col_values = warm_start::decode(problem, warm_start, &model.layout);
  hm.set_solution(Some(&col_values), None, None, None);

  let solved = hm.solve();
  let status = match solved.status() {
    HighsModelStatus::Optimal => MilpStatus::Optimal,
    HighsModelStatus::ReachedTimeLimit => MilpStatus::TimedOut,
    HighsModelStatus::Infeasible => return Err(MilpError::Infeasible),
    other => return Err(MilpError::SolverFailed(format!("status={other:?}"))),
  };

  let solution = solved.get_solution();
  let mut z = 0.0;
  for (col, coef) in &model.objective_coeffs {
    z += coef * solution[*col];
  }
  Ok(MilpResult {
    objective_value: z.max(0.0),
    status,
    solve_time_ms: started.elapsed().as_millis() as u64,
  })
}

struct MilpModel {
  problem: RowProblem,
  objective_coeffs: Vec<(highs::Col, f64)>,
  layout: ModelLayout,
}

struct ModelLayout {
  ix: NodeIndex,
  /// y_ov: (order_idx, vehicle_idx) → column index
  y: HashMap<(usize, usize), usize>,
  /// x_ijv: (node_i, node_j, vehicle_idx) → column index
  x: HashMap<(usize, usize, usize), usize>,
  /// q_iv: (node_i, vehicle_idx) → column index
  q: HashMap<(usize, usize), usize>,
  /// u_iv: (node_i, vehicle_idx) → column index
  u: HashMap<(usize, usize), usize>,
  /// Total number of columns in the model.
  num_cols: usize,
}

#[derive(Clone, Copy)]
struct NodeIndex {
  num_vehicles: usize,
  num_orders: usize,
}

impl NodeIndex {
  fn start(&self, v: usize) -> usize {
    v
  }
  fn pickup(&self, o: usize) -> usize {
    self.num_vehicles + o
  }
  fn delivery(&self, o: usize) -> usize {
    self.num_vehicles + self.num_orders + o
  }
  fn is_pickup(&self, node: usize) -> Option<usize> {
    let lo = self.num_vehicles;
    let hi = lo + self.num_orders;
    (lo..hi).contains(&node).then(|| node - lo)
  }
  fn is_delivery(&self, node: usize) -> Option<usize> {
    let lo = self.num_vehicles + self.num_orders;
    let hi = lo + self.num_orders;
    (lo..hi).contains(&node).then(|| node - lo)
  }
  fn service_nodes(&self) -> impl Iterator<Item = usize> {
    self.num_vehicles..self.num_vehicles + 2 * self.num_orders
  }
  fn vehicle_nodes(&self, v: usize) -> impl Iterator<Item = usize> {
    std::iter::once(self.start(v)).chain(self.service_nodes())
  }
}

fn build_milp(problem: &Problem, target: Objective) -> MilpModel {
  let v_count = problem.vehicles.len();
  let n_count = problem.orders.len();
  let ix = NodeIndex {
    num_vehicles: v_count,
    num_orders: n_count,
  };

  let mut pb = RowProblem::default();

  let mut y_col: HashMap<(usize, usize), highs::Col> = HashMap::new();
  let mut x_col: HashMap<(usize, usize, usize), highs::Col> = HashMap::new();
  let mut q_col: HashMap<(usize, usize), highs::Col> = HashMap::new();
  let mut u_col: HashMap<(usize, usize), highs::Col> = HashMap::new();

  let mut y_idx: HashMap<(usize, usize), usize> = HashMap::new();
  let mut x_idx: HashMap<(usize, usize, usize), usize> = HashMap::new();
  let mut q_idx: HashMap<(usize, usize), usize> = HashMap::new();
  let mut u_idx: HashMap<(usize, usize), usize> = HashMap::new();

  let mut objective_coeffs: Vec<(highs::Col, f64)> = Vec::new();

  // y_ov
  for o in 0..n_count {
    for v in 0..v_count {
      let idx = pb.num_cols();
      let col = pb.add_integer_column(0.0, 0.0..=1.0);
      y_col.insert((o, v), col);
      y_idx.insert((o, v), idx);
    }
  }

  // x_ijv
  for v in 0..v_count {
    let nodes: Vec<usize> = ix.vehicle_nodes(v).collect();
    let s = ix.start(v);
    for &i in &nodes {
      for &j in &nodes {
        if i == j {
          continue;
        }
        let cost = if j == s {
          0.0
        } else {
          arc_distance(problem, &ix, v, i, j) * objective_weight(problem, target, v)
        };
        let idx: usize = pb.num_cols();
        let col = pb.add_integer_column(cost, 0.0..=1.0);
        x_col.insert((i, j, v), col);
        x_idx.insert((i, j, v), idx);
        if cost != 0.0 {
          objective_coeffs.push((col, cost));
        }
      }
    }
  }

  // q_iv
  for v in 0..v_count {
    let idx = pb.num_cols();
    let s_col = pb.add_column(0.0, 0.0..=0.0);
    q_col.insert((ix.start(v), v), s_col);
    q_idx.insert((ix.start(v), v), idx);
    for i in ix.service_nodes() {
      let idx = pb.num_cols();
      let col = pb.add_column(0.0, 0.0..=1.0);
      q_col.insert((i, v), col);
      q_idx.insert((i, v), idx);
    }
  }

  // u_iv
  let mtz_max = 2.0 * n_count as f64;
  for v in 0..v_count {
    for i in ix.service_nodes() {
      let idx = pb.num_cols();
      let col = pb.add_column(0.0, 0.0..=mtz_max);
      u_col.insert((i, v), col);
      u_idx.insert((i, v), idx);
    }
  }

  // 1. Order assignment
  for o in 0..n_count {
    let factors: Vec<(highs::Col, f64)> = (0..v_count).map(|v| (y_col[&(o, v)], 1.0)).collect();
    pb.add_row(1.0..=1.0, &factors);
  }

  // 2. Tour start: each vehicle leaves its start at most once.
  for v in 0..v_count {
    let s = ix.start(v);
    let factors: Vec<(highs::Col, f64)> = ix
      .service_nodes()
      .filter_map(|j| x_col.get(&(s, j, v)).copied().map(|xv| (xv, 1.0)))
      .collect();
    if !factors.is_empty() {
      pb.add_row(f64::NEG_INFINITY..=1.0, &factors);
    }
  }

  // 3. Order servicing: enter and leave pickup and delivery iff y_ov
  for o in 0..n_count {
    for v in 0..v_count {
      let p = ix.pickup(o);
      let d = ix.delivery(o);
      let yov = y_col[&(o, v)];
      let nodes: Vec<usize> = ix.vehicle_nodes(v).collect();

      let mut into_p: Vec<(highs::Col, f64)> = Vec::new();
      let mut out_p: Vec<(highs::Col, f64)> = Vec::new();
      let mut into_d: Vec<(highs::Col, f64)> = Vec::new();
      let mut out_d: Vec<(highs::Col, f64)> = Vec::new();
      for &k in &nodes {
        if k != p {
          if let Some(&xv) = x_col.get(&(k, p, v)) {
            into_p.push((xv, 1.0));
          }
          if let Some(&xv) = x_col.get(&(p, k, v)) {
            out_p.push((xv, 1.0));
          }
        }
        if k != d {
          if let Some(&xv) = x_col.get(&(k, d, v)) {
            into_d.push((xv, 1.0));
          }
          if let Some(&xv) = x_col.get(&(d, k, v)) {
            out_d.push((xv, 1.0));
          }
        }
      }
      for mut row in [into_p, out_p, into_d, out_d] {
        row.push((yov, -1.0));
        pb.add_row(0.0..=0.0, &row);
      }
    }
  }

  // 4. Pickup-before-delivery
  let m_n = 2.0 * n_count as f64;
  for o in 0..n_count {
    for v in 0..v_count {
      let p = ix.pickup(o);
      let d = ix.delivery(o);
      let factors = [
        (u_col[&(p, v)], 1.0),
        (u_col[&(d, v)], -1.0),
        (y_col[&(o, v)], m_n),
      ];
      pb.add_row(f64::NEG_INFINITY..=(m_n - 1.0), factors);
    }
  }

  // 5. Capacity flow
  let m_q = 2.0_f64;
  for v in 0..v_count {
    let nodes: Vec<usize> = ix.vehicle_nodes(v).collect();
    for &i in &nodes {
      for &j in &nodes {
        if i == j {
          continue;
        }
        if ix.is_pickup(j).is_none() && ix.is_delivery(j).is_none() {
          continue; // no flow into a start node
        }
        let xij = x_col[&(i, j, v)];
        let qj = q_col[&(j, v)];

        let (delta_y_col, delta_y_coef): (Option<highs::Col>, f64) =
          if let Some(o) = ix.is_pickup(i) {
            (Some(y_col[&(o, v)]), 1.0 / problem.orders[o].load_factor)
          } else if let Some(o) = ix.is_delivery(i) {
            (Some(y_col[&(o, v)]), -1.0 / problem.orders[o].load_factor)
          } else {
            (None, 0.0)
          };

        let mut lower: Vec<(highs::Col, f64)> = Vec::with_capacity(4);
        let mut upper: Vec<(highs::Col, f64)> = Vec::with_capacity(4);
        lower.push((qj, 1.0));
        upper.push((qj, 1.0));
        if i != ix.start(v) {
          lower.push((q_col[&(i, v)], -1.0));
          upper.push((q_col[&(i, v)], -1.0));
        }
        if let Some(yc) = delta_y_col {
          lower.push((yc, -delta_y_coef));
          upper.push((yc, -delta_y_coef));
        }
        lower.push((xij, -m_q));
        upper.push((xij, m_q));
        pb.add_row(-m_q..=f64::INFINITY, &lower);
        pb.add_row(f64::NEG_INFINITY..=m_q, &upper);
      }
    }
  }

  // 6. MTZ subtour elimination
  for v in 0..v_count {
    let svc: Vec<usize> = ix.service_nodes().collect();
    for &i in &svc {
      for &j in &svc {
        if i == j {
          continue;
        }
        if let Some(&xij) = x_col.get(&(i, j, v)) {
          let factors = [(u_col[&(i, v)], 1.0), (u_col[&(j, v)], -1.0), (xij, m_n)];
          pb.add_row(f64::NEG_INFINITY..=(m_n - 1.0), factors);
        }
      }
    }
  }

  let num_cols = pb.num_cols();
  MilpModel {
    problem: pb,
    objective_coeffs,
    layout: ModelLayout {
      ix,
      y: y_idx,
      x: x_idx,
      q: q_idx,
      u: u_idx,
      num_cols,
    },
  }
}

fn objective_weight(problem: &Problem, target: Objective, v: usize) -> f64 {
  match target {
    Objective::Distance | Objective::Empty => 1.0,
    Objective::Price => problem.vehicles[v].price_km,
  }
}

fn arc_distance(problem: &Problem, ix: &NodeIndex, v: usize, i: usize, j: usize) -> f64 {
  haversine_km(
    node_location(problem, ix, v, i),
    node_location(problem, ix, v, j),
  )
}

fn node_location<'a>(
  problem: &'a Problem,
  ix: &NodeIndex,
  v: usize,
  node: usize,
) -> &'a vrppd_core::Location {
  if node == ix.start(v) {
    return &problem.vehicles[v].start_location;
  }
  if let Some(o) = ix.is_pickup(node) {
    return &problem.orders[o].pickup_location;
  }
  if let Some(o) = ix.is_delivery(node) {
    return &problem.orders[o].delivery_location;
  }
  panic!("node {node} not relevant to vehicle {v}");
}

#[cfg(test)]
mod tests {
  use super::*;
  use vrppd_core::{Location, Order, Vehicle};

  fn loc(lat: f64, lon: f64) -> Location {
    Location {
      hash: format!("{lat},{lon}"),
      latitude: lat,
      longitude: lon,
    }
  }

  fn vehicle(id: u32, price_km: f64, lat: f64, lon: f64) -> Vehicle {
    Vehicle {
      id,
      start_location: loc(lat, lon),
      price_km,
    }
  }

  fn order(id: u32, p: (f64, f64), d: (f64, f64)) -> Order {
    Order {
      id,
      pickup_location: loc(p.0, p.1),
      delivery_location: loc(d.0, d.1),
      load_factor: 1.0,
    }
  }

  #[test]
  fn empty_problem_yields_zero() {
    let problem = Problem {
      vehicles: vec![],
      orders: vec![],
    };
    let r = solve_milp(&problem, Objective::Distance, Duration::from_secs(10), 1).unwrap();
    assert_eq!(r.objective_value, 0.0);
    assert_eq!(r.status, MilpStatus::Optimal);
  }

  #[test]
  fn empty_objective_unsupported() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 1.0, 0.0, 0.0)],
      orders: vec![order(1, (0.0, 0.0), (0.0, 1.0))],
    };
    assert!(matches!(
      solve_milp(&problem, Objective::Empty, Duration::from_secs(10), 1),
      Err(MilpError::UnsupportedObjective(Objective::Empty))
    ));
  }

  /// A BF-derived warm start on a tiny instance must produce the same optimum
  /// as a cold solve. Acts as a smoke test for the `warm_start::decode` path
  /// until PSA/CEA crates land and can exercise it more thoroughly.
  #[test]
  fn warm_start_n1_matches_cold() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 1.0, 0.0, 0.0)],
      orders: vec![order(1, (0.0, 0.0), (0.0, 0.5))],
    };

    let cold = solve_milp(&problem, Objective::Distance, Duration::from_secs(10), 1).unwrap();

    let bf = vrppd_brute_force::solve(&problem);
    let warm = solve_milp_with_warm_start(
      &problem,
      Objective::Distance,
      Duration::from_secs(10),
      1,
      &bf.best_distance_solution,
    )
    .unwrap();

    assert_eq!(warm.status, MilpStatus::Optimal);
    assert!(
      (warm.objective_value - cold.objective_value).abs() < 1e-6,
      "warm-start objective {} differs from cold {}",
      warm.objective_value,
      cold.objective_value
    );
  }
}
