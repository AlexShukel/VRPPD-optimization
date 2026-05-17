//! Neighbour-generating operators (Shift, Swap, Intra-Shuffle) for the SA loop.

use rand::Rng;

use vrppd_core::{Problem, StopKind};
use vrppd_core::{OrderMatrix, VehicleStartMatrix, WorkingSolution, WorkingStop};

use crate::config::OperatorWeights;

pub fn generate_neighbor<R: Rng + ?Sized>(
  current: &WorkingSolution,
  problem: &Problem,
  order_mat: &OrderMatrix,
  vstart_mat: &VehicleStartMatrix,
  weights: OperatorWeights,
  rng: &mut R,
) -> Option<WorkingSolution> {
  let non_empty: Vec<usize> = current
    .routes
    .iter()
    .enumerate()
    .filter(|(_, r)| !r.stops.is_empty())
    .map(|(i, _)| i)
    .collect();

  if non_empty.is_empty() {
    return None;
  }

  let mut next = current.clone();
  let r: f64 = rng.r#gen();

  if r < weights.shift {
    apply_shift(&mut next, &non_empty, rng);
  } else if r < weights.shift + weights.swap && non_empty.len() >= 2 {
    apply_swap(&mut next, &non_empty, rng);
  } else {
    apply_intra_shuffle(&mut next, &non_empty, rng);
  }

  if !next.is_valid(problem) {
    return None;
  }
  next.recalculate_all(problem, order_mat, vstart_mat);
  Some(next)
}

fn apply_shift<R: Rng + ?Sized>(sol: &mut WorkingSolution, non_empty: &[usize], rng: &mut R) {
  let v1 = non_empty[rng.gen_range(0..non_empty.len())];
  let r1_len = sol.routes[v1].stops.len();
  if r1_len == 0 {
    return;
  }

  let stop_idx = rng.gen_range(0..r1_len);
  let order_idx = sol.routes[v1].stops[stop_idx].order_idx;
  sol.routes[v1].stops.retain(|s| s.order_idx != order_idx);

  let v2 = rng.gen_range(0..sol.routes.len());
  let r2_len = sol.routes[v2].stops.len();

  let pickup_pos = rng.gen_range(0..=r2_len);
  sol.routes[v2].stops.insert(
    pickup_pos,
    WorkingStop {
      order_idx,
      kind: StopKind::Pickup,
    },
  );

  let len_after = sol.routes[v2].stops.len();
  let delivery_pos = rng.gen_range((pickup_pos + 1)..=len_after);
  sol.routes[v2].stops.insert(
    delivery_pos,
    WorkingStop {
      order_idx,
      kind: StopKind::Delivery,
    },
  );
}

fn apply_swap<R: Rng + ?Sized>(sol: &mut WorkingSolution, non_empty: &[usize], rng: &mut R) {
  let v1 = non_empty[rng.gen_range(0..non_empty.len())];
  let mut v2 = non_empty[rng.gen_range(0..non_empty.len())];
  let mut tries = 0;
  while v1 == v2 && tries < 5 {
    v2 = non_empty[rng.gen_range(0..non_empty.len())];
    tries += 1;
  }
  if v1 == v2 {
    return;
  }

  let r1_len = sol.routes[v1].stops.len();
  let r2_len = sol.routes[v2].stops.len();
  let o1 = sol.routes[v1].stops[rng.gen_range(0..r1_len)].order_idx;
  let o2 = sol.routes[v2].stops[rng.gen_range(0..r2_len)].order_idx;
  if o1 == o2 {
    return;
  }

  sol.routes[v1].stops.retain(|s| s.order_idx != o1);
  sol.routes[v2].stops.retain(|s| s.order_idx != o2);

  sol.routes[v1].stops.push(WorkingStop {
    order_idx: o2,
    kind: StopKind::Pickup,
  });
  sol.routes[v1].stops.push(WorkingStop {
    order_idx: o2,
    kind: StopKind::Delivery,
  });
  sol.routes[v2].stops.push(WorkingStop {
    order_idx: o1,
    kind: StopKind::Pickup,
  });
  sol.routes[v2].stops.push(WorkingStop {
    order_idx: o1,
    kind: StopKind::Delivery,
  });
}

fn apply_intra_shuffle<R: Rng + ?Sized>(
  sol: &mut WorkingSolution,
  non_empty: &[usize],
  rng: &mut R,
) {
  let v = non_empty[rng.gen_range(0..non_empty.len())];
  let stops = &mut sol.routes[v].stops;
  if stops.len() < 4 {
    return;
  }

  let mut orders: Vec<usize> = Vec::with_capacity(stops.len() / 2);
  for s in stops.iter() {
    if !orders.contains(&s.order_idx) {
      orders.push(s.order_idx);
    }
  }

  for i in (1..orders.len()).rev() {
    let j = rng.gen_range(0..=i);
    orders.swap(i, j);
  }

  stops.clear();
  for o in orders {
    stops.push(WorkingStop {
      order_idx: o,
      kind: StopKind::Pickup,
    });
    stops.push(WorkingStop {
      order_idx: o,
      kind: StopKind::Delivery,
    });
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256StarStar;
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

  fn order(id: u32, p: (f64, f64), d: (f64, f64)) -> Order {
    Order {
      id,
      pickup_location: loc(p.0, p.1),
      delivery_location: loc(d.0, d.1),
      load_factor: 1.0,
    }
  }

  fn fixture() -> (Problem, OrderMatrix, VehicleStartMatrix, WorkingSolution) {
    let problem = Problem {
      vehicles: vec![vehicle(1, 0.0, 0.0), vehicle(2, 10.0, 10.0)],
      orders: vec![
        order(1, (0.0, 0.0), (1.0, 1.0)),
        order(2, (10.0, 10.0), (11.0, 11.0)),
        order(3, (5.0, 5.0), (6.0, 6.0)),
      ],
    };
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);

    let mut sol = WorkingSolution::empty(2);
    sol.routes[0].stops = vec![
      WorkingStop {
        order_idx: 0,
        kind: StopKind::Pickup,
      },
      WorkingStop {
        order_idx: 0,
        kind: StopKind::Delivery,
      },
      WorkingStop {
        order_idx: 2,
        kind: StopKind::Pickup,
      },
      WorkingStop {
        order_idx: 2,
        kind: StopKind::Delivery,
      },
    ];
    sol.routes[1].stops = vec![
      WorkingStop {
        order_idx: 1,
        kind: StopKind::Pickup,
      },
      WorkingStop {
        order_idx: 1,
        kind: StopKind::Delivery,
      },
    ];
    sol.recalculate_all(&problem, &order_mat, &vstart_mat);
    (problem, order_mat, vstart_mat, sol)
  }

  #[test]
  fn neighbor_post_condition_is_valid_or_none() {
    let (problem, order_mat, vstart_mat, sol) = fixture();
    let weights = OperatorWeights::default();
    let mut rng = Xoshiro256StarStar::seed_from_u64(2026);

    for _ in 0..200 {
      if let Some(n) = generate_neighbor(&sol, &problem, &order_mat, &vstart_mat, weights, &mut rng)
      {
        assert!(n.is_valid(&problem), "operator produced an invalid solution");
        let summed: f64 = n.routes.iter().map(|r| r.total_distance).sum();
        assert!((summed - n.total_distance).abs() < 1e-9);
      }
    }
  }

  #[test]
  fn neighbors_preserve_order_count() {
    let (problem, order_mat, vstart_mat, sol) = fixture();
    let weights = OperatorWeights::default();
    let mut rng = Xoshiro256StarStar::seed_from_u64(7);

    for _ in 0..200 {
      if let Some(n) = generate_neighbor(&sol, &problem, &order_mat, &vstart_mat, weights, &mut rng)
      {
        let total_pickups: usize = n
          .routes
          .iter()
          .map(|r| r.stops.iter().filter(|s| s.kind == StopKind::Pickup).count())
          .sum();
        assert_eq!(total_pickups, problem.orders.len());
      }
    }
  }
}
