//! Internal mutable solution representation shared by the metaheuristic solvers.

use std::collections::HashMap;

use crate::{haversine_km, Location};
use crate::{Order, Problem, ProblemSolution, RouteStop, StopKind, Vehicle, VehicleRoute};

const LOAD_TOLERANCE: f64 = 1e-6;
const EMPTY_LEG_THRESHOLD: f64 = 1e-3;
const MAX_LOAD: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct OrderMatrix {
  pub data: Vec<f64>,
  pub dim: usize,
}

impl OrderMatrix {
  #[inline(always)]
  pub fn get(&self, from: usize, to: usize) -> f64 {
    self.data[from * self.dim + to]
  }

  pub fn build(orders: &[Order]) -> Self {
    let dim = orders.len() * 2;
    let mut data = vec![0.0; dim * dim];

    for i in 0..dim {
      for j in 0..dim {
        if i == j {
          continue;
        }
        let from = stop_location(orders, i);
        let to = stop_location(orders, j);
        data[i * dim + j] = haversine_km(from, to);
      }
    }

    Self { data, dim }
  }
}

#[derive(Clone, Debug)]
pub struct VehicleStartMatrix {
  pub data: Vec<f64>,
  pub n_orders: usize,
}

impl VehicleStartMatrix {
  #[inline(always)]
  pub fn get(&self, vehicle_idx: usize, order_idx: usize) -> f64 {
    self.data[vehicle_idx * self.n_orders + order_idx]
  }

  pub fn build(vehicles: &[Vehicle], orders: &[Order]) -> Self {
    let n_orders = orders.len();
    let mut data = vec![0.0; vehicles.len() * n_orders.max(1)];

    for (v_idx, vehicle) in vehicles.iter().enumerate() {
      for (o_idx, order) in orders.iter().enumerate() {
        data[v_idx * n_orders + o_idx] =
          haversine_km(&vehicle.start_location, &order.pickup_location);
      }
    }

    Self { data, n_orders }
  }
}

#[inline(always)]
fn stop_location(orders: &[Order], stop_idx: usize) -> &Location {
  let order_idx = stop_idx / 2;
  if stop_idx % 2 == 0 {
    &orders[order_idx].pickup_location
  } else {
    &orders[order_idx].delivery_location
  }
}

#[inline(always)]
pub fn stop_node(order_idx: usize, kind: StopKind) -> usize {
  match kind {
    StopKind::Pickup => order_idx * 2,
    StopKind::Delivery => order_idx * 2 + 1,
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkingStop {
  pub order_idx: usize,
  pub kind: StopKind,
}

#[derive(Clone, Debug, Default)]
pub struct WorkingRoute {
  pub stops: Vec<WorkingStop>,
  pub total_distance: f64,
  pub empty_distance: f64,
  pub total_price: f64,
}

#[derive(Clone, Debug)]
pub struct WorkingSolution {
  pub routes: Vec<WorkingRoute>,
  pub total_distance: f64,
  pub empty_distance: f64,
  pub total_price: f64,
}

impl WorkingSolution {
  pub fn empty(num_vehicles: usize) -> Self {
    Self {
      routes: vec![WorkingRoute::default(); num_vehicles],
      total_distance: 0.0,
      empty_distance: 0.0,
      total_price: 0.0,
    }
  }

  pub fn is_valid(&self, problem: &Problem) -> bool {
    let mut picked_up = vec![false; problem.orders.len()];
    let mut delivered = vec![false; problem.orders.len()];

    for route in &self.routes {
      if !route.is_capacity_feasible(problem) {
        return false;
      }
      for stop in &route.stops {
        match stop.kind {
          StopKind::Pickup => {
            if picked_up[stop.order_idx] {
              return false;
            }
            picked_up[stop.order_idx] = true;
          }
          StopKind::Delivery => {
            if !picked_up[stop.order_idx] || delivered[stop.order_idx] {
              return false;
            }
            delivered[stop.order_idx] = true;
          }
        }
      }
    }

    picked_up == delivered
  }

  pub fn recalculate_all(
    &mut self,
    problem: &Problem,
    order_mat: &OrderMatrix,
    vstart_mat: &VehicleStartMatrix,
  ) {
    self.total_distance = 0.0;
    self.empty_distance = 0.0;
    self.total_price = 0.0;

    for (v_idx, route) in self.routes.iter_mut().enumerate() {
      route.recalculate(v_idx, problem, order_mat, vstart_mat);
      self.total_distance += route.total_distance;
      self.empty_distance += route.empty_distance;
      self.total_price += route.total_price;
    }
  }

  pub fn into_problem_solution(self, problem: &Problem) -> ProblemSolution {
    let mut routes_out: HashMap<String, VehicleRoute> = HashMap::new();
    for (v_idx, route) in self.routes.into_iter().enumerate() {
      let stops = route
        .stops
        .into_iter()
        .map(|s| RouteStop {
          order_id: problem.orders[s.order_idx].id,
          kind: s.kind,
        })
        .collect();
      routes_out.insert(
        problem.vehicles[v_idx].id.to_string(),
        VehicleRoute {
          stops,
          total_distance: route.total_distance,
          empty_distance: route.empty_distance,
          total_price: route.total_price,
        },
      );
    }
    ProblemSolution {
      routes: routes_out,
      total_distance: self.total_distance,
      empty_distance: self.empty_distance,
      total_price: self.total_price,
    }
  }
}

impl WorkingRoute {
  pub fn is_capacity_feasible(&self, problem: &Problem) -> bool {
    if self.stops.is_empty() {
      return true;
    }
    let mut load = 0.0;
    let mut picked = vec![false; problem.orders.len()];

    for stop in &self.stops {
      let load_delta = 1.0 / problem.orders[stop.order_idx].load_factor;
      match stop.kind {
        StopKind::Pickup => {
          if picked[stop.order_idx] {
            return false;
          }
          picked[stop.order_idx] = true;
          load += load_delta;
        }
        StopKind::Delivery => {
          if !picked[stop.order_idx] {
            return false;
          }
          load -= load_delta;
        }
      }
      if load > MAX_LOAD + LOAD_TOLERANCE {
        return false;
      }
    }

    load.abs() < LOAD_TOLERANCE
  }

  pub fn is_capacity_feasible_for_partial(&self, problem: &Problem) -> bool {
    let mut load = 0.0;
    let mut picked = vec![false; problem.orders.len()];
    for stop in &self.stops {
      let load_delta = 1.0 / problem.orders[stop.order_idx].load_factor;
      match stop.kind {
        StopKind::Pickup => {
          if picked[stop.order_idx] {
            return false;
          }
          picked[stop.order_idx] = true;
          load += load_delta;
        }
        StopKind::Delivery => {
          if !picked[stop.order_idx] {
            return false;
          }
          load -= load_delta;
        }
      }
      if load > MAX_LOAD + LOAD_TOLERANCE {
        return false;
      }
    }
    true
  }

  pub fn recalculate(
    &mut self,
    v_idx: usize,
    problem: &Problem,
    order_mat: &OrderMatrix,
    vstart_mat: &VehicleStartMatrix,
  ) {
    self.total_distance = 0.0;
    self.empty_distance = 0.0;
    self.total_price = 0.0;

    if self.stops.is_empty() {
      return;
    }

    let first = self.stops[0];
    let leg_to_first = vstart_mat.get(v_idx, first.order_idx);
    self.total_distance += leg_to_first;
    self.empty_distance += leg_to_first;

    let mut load = match first.kind {
      StopKind::Pickup => 1.0 / problem.orders[first.order_idx].load_factor,
      StopKind::Delivery => -1.0 / problem.orders[first.order_idx].load_factor,
    };

    for window in self.stops.windows(2) {
      let from = window[0];
      let to = window[1];
      let from_node = stop_node(from.order_idx, from.kind);
      let to_node = stop_node(to.order_idx, to.kind);
      let leg = order_mat.get(from_node, to_node);

      self.total_distance += leg;
      if load.abs() < EMPTY_LEG_THRESHOLD {
        self.empty_distance += leg;
      }

      let delta = 1.0 / problem.orders[to.order_idx].load_factor;
      load += match to.kind {
        StopKind::Pickup => delta,
        StopKind::Delivery => -delta,
      };
    }

    self.total_price = self.total_distance * problem.vehicles[v_idx].price_km;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Location, Order, Problem, Vehicle};

  fn loc(lat: f64, lon: f64) -> Location {
    Location {
      hash: format!("{lat},{lon}"),
      latitude: lat,
      longitude: lon,
    }
  }

  fn vehicle(id: u32) -> Vehicle {
    Vehicle {
      id,
      start_location: loc(0.0, 0.0),
      price_km: 2.0,
    }
  }

  fn order(id: u32, lf: f64) -> Order {
    Order {
      id,
      pickup_location: loc(0.0, 0.0),
      delivery_location: loc(0.0, 0.0),
      load_factor: lf,
    }
  }

  #[test]
  fn empty_solution_is_valid() {
    let problem = Problem {
      vehicles: vec![vehicle(1)],
      orders: vec![order(1, 1.0)],
    };
    let sol = WorkingSolution::empty(1);
    assert!(sol.is_valid(&problem));
  }

  #[test]
  fn delivery_before_pickup_is_invalid() {
    let problem = Problem {
      vehicles: vec![vehicle(1)],
      orders: vec![order(1, 1.0)],
    };
    let mut sol = WorkingSolution::empty(1);
    sol.routes[0].stops = vec![
      WorkingStop {
        order_idx: 0,
        kind: StopKind::Delivery,
      },
      WorkingStop {
        order_idx: 0,
        kind: StopKind::Pickup,
      },
    ];
    assert!(!sol.is_valid(&problem));
  }

  #[test]
  fn capacity_overflow_is_invalid() {
    let problem = Problem {
      vehicles: vec![vehicle(1)],
      orders: vec![order(1, 2.0), order(2, 2.0), order(3, 0.5)],
    };
    let mut sol = WorkingSolution::empty(1);
    sol.routes[0].stops = vec![
      WorkingStop {
        order_idx: 0,
        kind: StopKind::Pickup,
      },
      WorkingStop {
        order_idx: 1,
        kind: StopKind::Pickup,
      },
      WorkingStop {
        order_idx: 2,
        kind: StopKind::Pickup,
      },
      WorkingStop {
        order_idx: 0,
        kind: StopKind::Delivery,
      },
      WorkingStop {
        order_idx: 1,
        kind: StopKind::Delivery,
      },
      WorkingStop {
        order_idx: 2,
        kind: StopKind::Delivery,
      },
    ];
    assert!(!sol.is_valid(&problem));
  }

  #[test]
  fn order_matrix_zero_on_diagonal() {
    let orders = vec![Order {
      id: 1,
      pickup_location: loc(0.0, 0.0),
      delivery_location: loc(1.0, 0.0),
      load_factor: 1.0,
    }];
    let m = OrderMatrix::build(&orders);
    assert_eq!(m.dim, 2);
    assert_eq!(m.get(0, 0), 0.0);
    assert_eq!(m.get(1, 1), 0.0);
  }

  #[test]
  fn vehicle_start_matrix_layout() {
    let vehicles = vec![
      Vehicle {
        id: 7,
        start_location: loc(0.0, 0.0),
        price_km: 1.0,
      },
      Vehicle {
        id: 8,
        start_location: loc(10.0, 0.0),
        price_km: 1.0,
      },
    ];
    let orders = vec![
      Order {
        id: 1,
        pickup_location: loc(0.0, 0.0),
        delivery_location: loc(50.0, 50.0),
        load_factor: 1.0,
      },
      Order {
        id: 2,
        pickup_location: loc(10.0, 0.0),
        delivery_location: loc(50.0, 50.0),
        load_factor: 1.0,
      },
    ];
    let m = VehicleStartMatrix::build(&vehicles, &orders);
    assert!(m.get(0, 0) < 1e-3);
    assert!(m.get(1, 1) < 1e-3);
    assert!(m.get(0, 1) > 100.0);
    assert!(m.get(1, 0) > 100.0);
  }
}
