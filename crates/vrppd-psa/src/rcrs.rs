//! RCRS (Residual Capacity and Radial Surcharge) initial-solution heuristic.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

use vrppd_core::{Objective, Problem, StopKind};
use vrppd_core::{OrderMatrix, VehicleStartMatrix, WorkingRoute, WorkingSolution, WorkingStop};

pub fn generate_rcrs<R: Rng + ?Sized>(
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  rng: &mut R,
) -> WorkingSolution {
  let mut sol = WorkingSolution::empty(problem.vehicles.len());

  let mut order_indices: Vec<usize> = (0..problem.orders.len()).collect();
  order_indices.shuffle(rng);

  for o_idx in order_indices {
    let mut best: Option<Insertion> = None;

    for v_idx in 0..problem.vehicles.len() {
      let route_len = sol.routes[v_idx].stops.len();
      for pickup_pos in 0..=route_len {
        for delivery_pos in (pickup_pos + 1)..=(route_len + 1) {
          let metrics = match estimate_insertion(
            &sol.routes[v_idx],
            v_idx,
            o_idx,
            pickup_pos,
            delivery_pos,
            problem,
            order_mat,
            vstart_mat,
          ) {
            Some(m) => m,
            None => continue,
          };

          let cost = match target {
            Objective::Price => metrics.delta_total * problem.vehicles[v_idx].price_km,
            Objective::Distance => metrics.delta_total,
            Objective::Empty => metrics.delta_empty + 0.4 * vstart_mat.get(v_idx, o_idx),
          };

          if best.as_ref().is_none_or(|b| cost < b.cost) {
            best = Some(Insertion {
              v_idx,
              pickup_pos,
              delivery_pos,
              cost,
            });
          }
        }
      }
    }

    if let Some(ins) = best {
      let route = &mut sol.routes[ins.v_idx];
      route.stops.insert(
        ins.pickup_pos,
        WorkingStop {
          order_idx: o_idx,
          kind: StopKind::Pickup,
        },
      );
      route.stops.insert(
        ins.delivery_pos,
        WorkingStop {
          order_idx: o_idx,
          kind: StopKind::Delivery,
        },
      );
      route.recalculate(ins.v_idx, problem, order_mat, vstart_mat);
    }
  }

  sol.recalculate_all(problem, order_mat, vstart_mat);
  sol
}

pub fn generate_rcrs_seeded(
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  seed: u64,
) -> WorkingSolution {
  let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
  generate_rcrs(problem, order_mat, vstart_mat, target, &mut rng)
}

#[derive(Clone, Copy, Debug)]
struct Insertion {
  v_idx: usize,
  pickup_pos: usize,
  delivery_pos: usize,
  cost: f64,
}

#[derive(Clone, Copy, Debug)]
struct InsertionMetrics {
  delta_total: f64,
  delta_empty: f64,
}

#[allow(clippy::too_many_arguments)]
fn estimate_insertion(
  route: &WorkingRoute,
  v_idx: usize,
  order_idx: usize,
  pickup_pos: usize,
  delivery_pos: usize,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
) -> Option<InsertionMetrics> {
  let mut trial = WorkingRoute {
    stops: Vec::with_capacity(route.stops.len() + 2),
    ..WorkingRoute::default()
  };
  trial.stops.extend_from_slice(&route.stops);
  trial.stops.insert(
    pickup_pos,
    WorkingStop {
      order_idx,
      kind: StopKind::Pickup,
    },
  );
  trial.stops.insert(
    delivery_pos,
    WorkingStop {
      order_idx,
      kind: StopKind::Delivery,
    },
  );

  if !trial.is_capacity_feasible_for_partial(problem) {
    return None;
  }

  trial.recalculate(v_idx, problem, order_mat, vstart_mat);

  Some(InsertionMetrics {
    delta_total: trial.total_distance - route.total_distance,
    delta_empty: trial.empty_distance - route.empty_distance,
  })
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

  fn vehicle(id: u32, lat: f64, lon: f64) -> Vehicle {
    Vehicle {
      id,
      start_location: loc(lat, lon),
      price_km: 1.0,
    }
  }

  fn order(id: u32, p: (f64, f64), d: (f64, f64), lf: f64) -> Order {
    Order {
      id,
      pickup_location: loc(p.0, p.1),
      delivery_location: loc(d.0, d.1),
      load_factor: lf,
    }
  }

  #[test]
  fn rcrs_produces_valid_solution_on_single_order() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 0.0, 0.0)],
      orders: vec![order(7, (0.5, 0.5), (1.0, 1.0), 1.0)],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let sol = generate_rcrs_seeded(&problem, &order_mat, &vstart_mat, Objective::Distance, 42);

    assert!(sol.is_valid(&problem));
    assert_eq!(sol.routes[0].stops.len(), 2);
    assert_eq!(sol.routes[0].stops[0].kind, StopKind::Pickup);
    assert_eq!(sol.routes[0].stops[1].kind, StopKind::Delivery);
  }

  #[test]
  fn rcrs_rejects_orders_that_violate_capacity_when_alone() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 0.0, 0.0)],
      orders: vec![
        order(1, (0.5, 0.5), (1.0, 1.0), 2.0),
        order(2, (0.5, 0.5), (1.0, 1.0), 0.5),
      ],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let sol = generate_rcrs_seeded(&problem, &order_mat, &vstart_mat, Objective::Distance, 0);

    let placed: Vec<u32> = sol.routes[0]
      .stops
      .iter()
      .filter(|s| s.kind == StopKind::Pickup)
      .map(|s| problem.orders[s.order_idx].id)
      .collect();
    assert_eq!(placed, vec![1]);
  }

  #[test]
  fn rcrs_uses_multiple_vehicles_when_geographically_split() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 0.0, 0.0), vehicle(2, 10.0, 10.0)],
      orders: vec![
        order(1, (0.0, 0.0), (0.5, 0.5), 1.0),
        order(2, (10.0, 10.0), (10.5, 10.5), 1.0),
      ],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let sol = generate_rcrs_seeded(&problem, &order_mat, &vstart_mat, Objective::Distance, 1);

    assert!(sol.is_valid(&problem));
    let v1_orders: Vec<u32> = sol.routes[0]
      .stops
      .iter()
      .filter(|s| s.kind == StopKind::Pickup)
      .map(|s| problem.orders[s.order_idx].id)
      .collect();
    let v2_orders: Vec<u32> = sol.routes[1]
      .stops
      .iter()
      .filter(|s| s.kind == StopKind::Pickup)
      .map(|s| problem.orders[s.order_idx].id)
      .collect();
    assert_eq!(v1_orders, vec![1]);
    assert_eq!(v2_orders, vec![2]);
  }
}
