//! FSCIM crossover operator used by Population II.

use rand::seq::SliceRandom;
use rand::Rng;

use vrppd_core::{Objective, OrderMatrix, Problem, StopKind, VehicleStartMatrix, WorkingSolution};

use crate::rscim::insert_cheapest;

pub fn crossover<R: Rng + ?Sized>(
  parent1: &WorkingSolution,
  parent2: &WorkingSolution,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  target: Objective,
  rng: &mut R,
) -> WorkingSolution {
  let num_v = problem.vehicles.len();
  let num_o = problem.orders.len();

  let mut offspring = WorkingSolution::empty(num_v);
  let mut covered_orders = vec![false; num_o];
  let mut used_vehicles = vec![false; num_v];

  let mut p1_candidates: Vec<usize> = parent1
    .routes
    .iter()
    .enumerate()
    .filter(|(_, r)| !r.stops.is_empty())
    .map(|(i, _)| i)
    .collect();
  let mut p2_candidates: Vec<usize> = parent2
    .routes
    .iter()
    .enumerate()
    .filter(|(_, r)| !r.stops.is_empty())
    .map(|(i, _)| i)
    .collect();

  loop {
    let inherited_p1 = try_inherit(
      &mut offspring,
      parent1,
      &mut p1_candidates,
      &mut covered_orders,
      &mut used_vehicles,
      rng,
    );
    let inherited_p2 = try_inherit(
      &mut offspring,
      parent2,
      &mut p2_candidates,
      &mut covered_orders,
      &mut used_vehicles,
      rng,
    );
    if !inherited_p1 && !inherited_p2 {
      break;
    }
  }

  offspring.recalculate_all(problem, order_mat, vstart_mat);

  let leftovers: Vec<usize> = (0..num_o).filter(|i| !covered_orders[*i]).collect();
  for o_idx in leftovers {
    insert_cheapest(
      &mut offspring,
      o_idx,
      problem,
      order_mat,
      vstart_mat,
      target,
    );
  }

  offspring.recalculate_all(problem, order_mat, vstart_mat);
  offspring
}

fn try_inherit<R: Rng + ?Sized>(
  offspring: &mut WorkingSolution,
  parent: &WorkingSolution,
  candidates: &mut Vec<usize>,
  covered_orders: &mut [bool],
  used_vehicles: &mut [bool],
  rng: &mut R,
) -> bool {
  candidates.shuffle(rng);
  let mut feasible_idx: Option<usize> = None;
  for (i, &v_idx) in candidates.iter().enumerate() {
    if used_vehicles[v_idx] {
      continue;
    }
    let route = &parent.routes[v_idx];
    let conflict = route
      .stops
      .iter()
      .any(|s| s.kind == StopKind::Pickup && covered_orders[s.order_idx]);
    if !conflict {
      feasible_idx = Some(i);
      break;
    }
  }
  let i = match feasible_idx {
    Some(i) => i,
    None => return false,
  };
  let v_idx = candidates.swap_remove(i);

  offspring.routes[v_idx].stops = parent.routes[v_idx].stops.clone();
  used_vehicles[v_idx] = true;
  for stop in &parent.routes[v_idx].stops {
    if stop.kind == StopKind::Pickup {
      covered_orders[stop.order_idx] = true;
    }
  }
  true
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;
  use vrppd_core::{Location, Order, Vehicle};

  use crate::rscim::generate_rscim;

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

  fn order(id: u32, p: (f64, f64), d: (f64, f64)) -> Order {
    Order {
      id,
      pickup_location: loc(p.0, p.1),
      delivery_location: loc(d.0, d.1),
      load_factor: 1.0,
    }
  }

  #[test]
  fn crossover_produces_valid_offspring_with_full_coverage() {
    let problem = Problem {
      vehicles: vec![
        vehicle(1, 0.0, 0.0),
        vehicle(2, 5.0, 5.0),
        vehicle(3, 10.0, 10.0),
      ],
      orders: vec![
        order(1, (0.0, 0.0), (1.0, 1.0)),
        order(2, (5.0, 5.0), (6.0, 6.0)),
        order(3, (10.0, 10.0), (11.0, 11.0)),
        order(4, (2.5, 2.5), (3.0, 3.0)),
      ],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let mut rng = Xoshiro256StarStar::seed_from_u64(99);
    let parent1 = generate_rscim(
      &problem,
      &order_mat,
      &vstart_mat,
      Objective::Distance,
      &mut rng,
    );
    let parent2 = generate_rscim(
      &problem,
      &order_mat,
      &vstart_mat,
      Objective::Distance,
      &mut rng,
    );

    for _ in 0..30 {
      let child = crossover(
        &parent1,
        &parent2,
        &problem,
        &order_mat,
        &vstart_mat,
        Objective::Distance,
        &mut rng,
      );
      assert!(child.is_valid(&problem));
      let pickups: usize = child
        .routes
        .iter()
        .flat_map(|r| r.stops.iter())
        .filter(|s| s.kind == StopKind::Pickup)
        .count();
      assert_eq!(pickups, problem.orders.len());
    }
  }
}
