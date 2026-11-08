//! RSCIM (Random Seeds Cheapest Insertion Method) — initial-solution generator.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

use vrppd_core::{
  Objective, OrderMatrix, Problem, StopKind, VehicleStartMatrix, WorkingRoute, WorkingSolution,
  WorkingStop,
};

fn choose_seed_count(problem: &Problem) -> usize {
  if problem.orders.is_empty() {
    return 0;
  }
  let total_demand: f64 = problem.orders.iter().map(|o| 1.0 / o.load_factor).sum();
  let mean_capacity = 1.0_f64;
  let raw = total_demand / mean_capacity;
  let k = raw.ceil() as usize;
  k.clamp(1, problem.vehicles.len().min(problem.orders.len()))
}

pub fn generate_rscim<R: Rng + ?Sized>(
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  rng: &mut R,
) -> WorkingSolution {
  if problem.orders.is_empty() {
    return WorkingSolution::empty(problem.vehicles.len());
  }

  let mut sol = WorkingSolution::empty(problem.vehicles.len());

  let mut order_perm: Vec<usize> = (0..problem.orders.len()).collect();
  order_perm.shuffle(rng);

  let k = choose_seed_count(problem);
  let mut used_vehicle = vec![false; problem.vehicles.len()];

  for &o_idx in order_perm.iter().take(k) {
    let mut best_v: Option<usize> = None;
    let mut best_dist = f64::INFINITY;
    for (v_idx, used) in used_vehicle.iter().enumerate() {
      if *used {
        continue;
      }
      let d = vstart_mat.get(v_idx, o_idx);
      if d < best_dist {
        best_dist = d;
        best_v = Some(v_idx);
      }
    }
    let v_idx = match best_v {
      Some(v) => v,
      None => break,
    };
    used_vehicle[v_idx] = true;

    let route = &mut sol.routes[v_idx];
    route.stops.push(WorkingStop {
      order_idx: o_idx,
      kind: StopKind::Pickup,
    });
    route.stops.push(WorkingStop {
      order_idx: o_idx,
      kind: StopKind::Delivery,
    });
    if route.is_capacity_feasible(problem) {
      route.recalculate(v_idx, problem, order_mat, vstart_mat);
    } else {
      route.stops.clear();
      used_vehicle[v_idx] = false;
    }
  }

  for &o_idx in order_perm.iter().skip(k) {
    insert_cheapest(&mut sol, o_idx, problem, order_mat, vstart_mat, target);
  }

  sol.recalculate_all(problem, order_mat, vstart_mat);
  sol
}

pub fn generate_rscim_seeded(
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  seed: u64,
) -> WorkingSolution {
  let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
  generate_rscim(problem, order_mat, vstart_mat, target, &mut rng)
}

#[derive(Clone, Copy)]
struct Insertion {
  v_idx: usize,
  pickup_pos: usize,
  delivery_pos: usize,
  cost: f64,
}

pub fn insert_cheapest(
  sol: &mut WorkingSolution,
  o_idx: usize,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
) -> bool {
  let mut best: Option<Insertion> = None;

  for v_idx in 0..problem.vehicles.len() {
    let route_len = sol.routes[v_idx].stops.len();
    let route_dist = sol.routes[v_idx].total_distance;
    let route_empty = sol.routes[v_idx].empty_distance;

    for pickup_pos in 0..=route_len {
      for delivery_pos in (pickup_pos + 1)..=(route_len + 1) {
        let cost = match trial_insertion_cost(
          &sol.routes[v_idx],
          v_idx,
          o_idx,
          pickup_pos,
          delivery_pos,
          problem,
          order_mat,
          vstart_mat,
          target,
          route_dist,
          route_empty,
        ) {
          Some(c) => c,
          None => continue,
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
    true
  } else {
    false
  }
}

#[allow(clippy::too_many_arguments)]
fn trial_insertion_cost(
  route: &WorkingRoute,
  v_idx: usize,
  order_idx: usize,
  pickup_pos: usize,
  delivery_pos: usize,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  route_dist_before: f64,
  route_empty_before: f64,
) -> Option<f64> {
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

  let cost = match target {
    Objective::Distance => trial.total_distance - route_dist_before,
    Objective::Empty => {
      let delta_empty = trial.empty_distance - route_empty_before;
      let start_to_pickup = vstart_mat.get(v_idx, order_idx);
      delta_empty + 0.4 * start_to_pickup
    }
    Objective::Price => {
      (trial.total_distance - route_dist_before) * problem.vehicles[v_idx].price_km
    }
  };
  Some(cost)
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
  fn rscim_produces_valid_solution_on_single_order() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 0.0, 0.0)],
      orders: vec![order(7, (0.5, 0.5), (1.0, 1.0), 1.0)],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let sol = generate_rscim_seeded(&problem, &order_mat, &vstart_mat, Objective::Distance, 1);

    assert!(sol.is_valid(&problem));
    assert_eq!(sol.routes[0].stops.len(), 2);
    assert_eq!(sol.routes[0].stops[0].kind, StopKind::Pickup);
    assert_eq!(sol.routes[0].stops[1].kind, StopKind::Delivery);
  }

  #[test]
  fn rscim_uses_multiple_vehicles_when_geographically_split() {
    let problem = Problem {
      vehicles: vec![vehicle(1, 0.0, 0.0), vehicle(2, 10.0, 10.0)],
      orders: vec![
        order(1, (0.0, 0.0), (0.5, 0.5), 1.0),
        order(2, (10.0, 10.0), (10.5, 10.5), 1.0),
      ],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let sol = generate_rscim_seeded(&problem, &order_mat, &vstart_mat, Objective::Distance, 2);
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

  #[test]
  fn rscim_seed_count_matches_paper_formula() {
    let problem = Problem {
      vehicles: (1..=4).map(|i| vehicle(i, i as f64, 0.0)).collect(),
      orders: (1..=4)
        .map(|i| order(i, (i as f64, 0.0), (i as f64, 1.0), 1.0))
        .collect(),
    };
    assert_eq!(choose_seed_count(&problem), 4);

    let problem = Problem {
      vehicles: (1..=4).map(|i| vehicle(i, i as f64, 0.0)).collect(),
      orders: (1..=4)
        .map(|i| order(i, (i as f64, 0.0), (i as f64, 1.0), 2.0))
        .collect(),
    };
    assert_eq!(choose_seed_count(&problem), 2);
  }
}
