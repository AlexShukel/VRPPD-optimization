//! Fast solution helpers for LNS: order lookup, removal, delta-cost cheapest
//! insertion (pickup before delivery, capacity feasible) and route savings.
//!
//! Only the DISTANCE objective is supported by the delta evaluation; the
//! `WorkingRoute` totals (distance, empty distance, price) are still kept
//! consistent by recalculating a route after every structural change.

use vrppd_core::{
  OrderMatrix, Problem, StopKind, VehicleStartMatrix, WorkingRoute, WorkingSolution, WorkingStop,
  stop_node,
};

const MAX_LOAD: f64 = 1.0;
const LOAD_TOLERANCE: f64 = 1e-6;

/// Static per-instance data shared by all LNS components.
pub struct Context<'a> {
  pub problem: &'a Problem,
  pub order_mat: OrderMatrix,
  pub vstart_mat: VehicleStartMatrix,
  /// 1 / load_factor per order.
  pub demand: Vec<f64>,
  /// Feature 7: distance from the order's pickup or delivery to the nearest
  /// pickup or delivery of any other order (static).
  pub nearest_other: Vec<f64>,
  /// Shaw relatedness: pickup-pickup + delivery-delivery distance.
  pub relatedness: Vec<f64>,
  /// Largest pairwise stop distance (used for normalisation).
  pub max_dist: f64,
}

impl<'a> Context<'a> {
  pub fn new(problem: &'a Problem) -> Self {
    let n = problem.orders.len();
    let order_mat = OrderMatrix::build(&problem.orders);
    let vstart_mat = VehicleStartMatrix::build(&problem.vehicles, &problem.orders);
    let demand: Vec<f64> = problem.orders.iter().map(|o| 1.0 / o.load_factor).collect();

    let mut nearest_other = vec![f64::INFINITY; n];
    let mut relatedness = vec![0.0; n * n];
    let mut max_dist = 0.0_f64;
    for i in 0..n {
      let pi = stop_node(i, StopKind::Pickup);
      let di = stop_node(i, StopKind::Delivery);
      for j in 0..n {
        if i == j {
          continue;
        }
        let pj = stop_node(j, StopKind::Pickup);
        let dj = stop_node(j, StopKind::Delivery);
        let dpp = order_mat.get(pi, pj);
        let dpd = order_mat.get(pi, dj);
        let ddp = order_mat.get(di, pj);
        let ddd = order_mat.get(di, dj);
        let m = dpp.min(dpd).min(ddp).min(ddd);
        if m < nearest_other[i] {
          nearest_other[i] = m;
        }
        relatedness[i * n + j] = dpp + ddd;
        max_dist = max_dist.max(dpp).max(dpd).max(ddp).max(ddd);
      }
      max_dist = max_dist.max(order_mat.get(pi, di));
    }
    for v in nearest_other.iter_mut() {
      if !v.is_finite() {
        *v = 0.0;
      }
    }
    if max_dist <= 0.0 {
      max_dist = 1.0;
    }

    Self {
      problem,
      order_mat,
      vstart_mat,
      demand,
      nearest_other,
      relatedness,
      max_dist,
    }
  }

  #[inline(always)]
  pub fn n_orders(&self) -> usize {
    self.problem.orders.len()
  }

  #[inline(always)]
  pub fn n_vehicles(&self) -> usize {
    self.problem.vehicles.len()
  }

  #[inline(always)]
  pub fn related(&self, i: usize, j: usize) -> f64 {
    self.relatedness[i * self.n_orders() + j]
  }

  /// Distance between two stops, where `from == None` means the vehicle start
  /// (only defined for a pickup `to`, which is the only stop that can be first).
  #[inline(always)]
  fn leg(&self, v_idx: usize, from: Option<WorkingStop>, to: WorkingStop) -> f64 {
    match from {
      None => self.vstart_mat.get(v_idx, to.order_idx),
      Some(f) => self.order_mat.get(
        stop_node(f.order_idx, f.kind),
        stop_node(to.order_idx, to.kind),
      ),
    }
  }

  #[inline(always)]
  fn stop_dist(&self, a: WorkingStop, b: WorkingStop) -> f64 {
    self.order_mat.get(
      stop_node(a.order_idx, a.kind),
      stop_node(b.order_idx, b.kind),
    )
  }
}

/// Where each order currently is: vehicle index and positions of its two stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
  pub v_idx: usize,
  pub pickup_pos: usize,
  pub delivery_pos: usize,
}

/// Map order index -> placement (None if the order is currently unrouted).
pub fn placements(sol: &WorkingSolution, n_orders: usize) -> Vec<Option<Placement>> {
  let mut out = vec![None; n_orders];
  for (v_idx, route) in sol.routes.iter().enumerate() {
    for (pos, stop) in route.stops.iter().enumerate() {
      let entry = out[stop.order_idx].get_or_insert(Placement {
        v_idx,
        pickup_pos: 0,
        delivery_pos: 0,
      });
      entry.v_idx = v_idx;
      match stop.kind {
        StopKind::Pickup => entry.pickup_pos = pos,
        StopKind::Delivery => entry.delivery_pos = pos,
      }
    }
  }
  out
}

/// Distance of a route if the given order were removed (closed form, no clone).
pub fn route_distance_without(
  ctx: &Context,
  route: &WorkingRoute,
  v_idx: usize,
  order_idx: usize,
) -> f64 {
  let stops = &route.stops;
  let mut total = 0.0;
  let mut prev: Option<WorkingStop> = None;
  for &s in stops {
    if s.order_idx == order_idx {
      continue;
    }
    total += ctx.leg(v_idx, prev, s);
    prev = Some(s);
  }
  total
}

/// Saving in route distance if the order is removed from its route (>= 0 unless
/// triangle inequality is violated by the data).
pub fn removal_saving(ctx: &Context, sol: &WorkingSolution, p: Placement, order_idx: usize) -> f64 {
  let route = &sol.routes[p.v_idx];
  route.total_distance - route_distance_without(ctx, route, p.v_idx, order_idx)
}

/// Remove the order's two stops from the solution and recalculate the route.
pub fn remove_order(ctx: &Context, sol: &mut WorkingSolution, v_idx: usize, order_idx: usize) {
  let route = &mut sol.routes[v_idx];
  let before = (
    route.total_distance,
    route.empty_distance,
    route.total_price,
  );
  route.stops.retain(|s| s.order_idx != order_idx);
  route.recalculate(v_idx, ctx.problem, &ctx.order_mat, &ctx.vstart_mat);
  sol.total_distance += route.total_distance - before.0;
  sol.empty_distance += route.empty_distance - before.1;
  sol.total_price += route.total_price - before.2;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Insertion {
  pub v_idx: usize,
  /// Pickup is inserted before original index `pickup_pos` (0..=len).
  pub pickup_pos: usize,
  /// Delivery is inserted before original index `delivery_pos` (pickup_pos..=len).
  /// After insertion the delivery sits at index `delivery_pos + 1`.
  pub delivery_pos: usize,
  pub delta: f64,
}

/// Evaluate every feasible insertion position for `order_idx` in `route` and
/// return the cheapest (by distance delta).
pub fn best_insertion_in_route(
  ctx: &Context,
  route: &WorkingRoute,
  v_idx: usize,
  order_idx: usize,
) -> Option<Insertion> {
  let stops = &route.stops;
  let len = stops.len();
  let d = ctx.demand[order_idx];
  let pickup = WorkingStop {
    order_idx,
    kind: StopKind::Pickup,
  };
  let delivery = WorkingStop {
    order_idx,
    kind: StopKind::Delivery,
  };
  let p_d = ctx.stop_dist(pickup, delivery);

  // load_after[t] = load after visiting stops[t]
  let mut load_after = Vec::with_capacity(len);
  let mut load = 0.0;
  for s in stops {
    let dd = ctx.demand[s.order_idx];
    match s.kind {
      StopKind::Pickup => load += dd,
      StopKind::Delivery => load -= dd,
    }
    load_after.push(load);
  }

  let mut best: Option<Insertion> = None;

  for i in 0..=len {
    let load_before_pickup = if i == 0 { 0.0 } else { load_after[i - 1] };
    if load_before_pickup + d > MAX_LOAD + LOAD_TOLERANCE {
      continue;
    }
    let prev_p: Option<WorkingStop> = if i == 0 { None } else { Some(stops[i - 1]) };
    let prev_to_pickup = ctx.leg(v_idx, prev_p, pickup);

    // Case j == i: pickup and delivery adjacent, between prev_p and stops[i].
    {
      let (removed, added) = if i < len {
        let next = stops[i];
        (
          ctx.leg(v_idx, prev_p, next),
          prev_to_pickup + p_d + ctx.stop_dist(delivery, next),
        )
      } else {
        (0.0, prev_to_pickup + p_d)
      };
      let delta = added - removed;
      if best.as_ref().is_none_or(|b| delta < b.delta) {
        best = Some(Insertion {
          v_idx,
          pickup_pos: i,
          delivery_pos: i,
          delta,
        });
      }
    }

    if i >= len {
      continue;
    }

    // Case j > i: pickup between prev_p and stops[i]; delivery between stops[j-1] and stops[j].
    let next_p = stops[i];
    let pickup_part =
      prev_to_pickup + ctx.stop_dist(pickup, next_p) - ctx.leg(v_idx, prev_p, next_p);
    let mut max_load = load_before_pickup;
    for j in (i + 1)..=len {
      // stops[j-1] now carries the extra demand d.
      max_load = max_load.max(load_after[j - 1]);
      if max_load + d > MAX_LOAD + LOAD_TOLERANCE {
        break;
      }
      let prev_d = stops[j - 1];
      let delivery_part = if j < len {
        let next_d = stops[j];
        ctx.stop_dist(prev_d, delivery) + ctx.stop_dist(delivery, next_d)
          - ctx.stop_dist(prev_d, next_d)
      } else {
        ctx.stop_dist(prev_d, delivery)
      };
      let delta = pickup_part + delivery_part;
      if best.as_ref().is_none_or(|b| delta < b.delta) {
        best = Some(Insertion {
          v_idx,
          pickup_pos: i,
          delivery_pos: j,
          delta,
        });
      }
    }
  }

  best
}

/// Cheapest insertion over all vehicles.
pub fn best_insertion(ctx: &Context, sol: &WorkingSolution, order_idx: usize) -> Option<Insertion> {
  let mut best: Option<Insertion> = None;
  for (v_idx, route) in sol.routes.iter().enumerate() {
    if let Some(ins) = best_insertion_in_route(ctx, route, v_idx, order_idx)
      && best.as_ref().is_none_or(|b| ins.delta < b.delta)
    {
      best = Some(ins);
    }
  }
  best
}

/// Apply an insertion and recalculate the touched route.
pub fn apply_insertion(ctx: &Context, sol: &mut WorkingSolution, order_idx: usize, ins: Insertion) {
  let route = &mut sol.routes[ins.v_idx];
  let before = (
    route.total_distance,
    route.empty_distance,
    route.total_price,
  );
  route.stops.insert(
    ins.pickup_pos,
    WorkingStop {
      order_idx,
      kind: StopKind::Pickup,
    },
  );
  route.stops.insert(
    ins.delivery_pos + 1,
    WorkingStop {
      order_idx,
      kind: StopKind::Delivery,
    },
  );
  route.recalculate(ins.v_idx, ctx.problem, &ctx.order_mat, &ctx.vstart_mat);
  sol.total_distance += route.total_distance - before.0;
  sol.empty_distance += route.empty_distance - before.1;
  sol.total_price += route.total_price - before.2;
}

/// Greedy cheapest insertion of all `orders` (in the given sequence).
/// Returns false if some order could not be inserted (should not happen with
/// load factors >= 1, since an empty vehicle always fits one order).
pub fn repair_greedy(ctx: &Context, sol: &mut WorkingSolution, orders: &[usize]) -> bool {
  let mut ok = true;
  for &o in orders {
    match best_insertion(ctx, sol, o) {
      Some(ins) => apply_insertion(ctx, sol, o, ins),
      None => ok = false,
    }
  }
  ok
}

/// Remove all `orders` from the solution.
pub fn destroy(ctx: &Context, sol: &mut WorkingSolution, orders: &[usize]) {
  let pl = placements(sol, ctx.n_orders());
  for &o in orders {
    if let Some(p) = pl[o] {
      remove_order(ctx, sol, p.v_idx, o);
    }
  }
}

/// Convert the solution into stop lists (order ids) per vehicle id.
pub fn stop_lists(problem: &Problem, sol: &WorkingSolution) -> Vec<Vec<(u32, StopKind)>> {
  sol
    .routes
    .iter()
    .map(|r| {
      r.stops
        .iter()
        .map(|s| (problem.orders[s.order_idx].id, s.kind))
        .collect()
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use rand::SeedableRng;
  use rand::seq::SliceRandom;
  use rand_xoshiro::Xoshiro256StarStar;
  use vrppd_core::Objective;

  fn load_problem(cell: &str, idx: usize) -> Problem {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("../../problem-gen/problems")
      .join(cell);
    let mut files: Vec<_> = std::fs::read_dir(dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .map(|e| e.path())
      .collect();
    files.sort();
    serde_json::from_str(&std::fs::read_to_string(&files[idx]).unwrap()).unwrap()
  }

  #[test]
  fn delta_insertion_matches_cea_cheapest_insertion() {
    for (cell, idx) in [("10_10", 0), ("10_20", 1), ("20_50", 2)] {
      let problem = load_problem(cell, idx);
      let ctx = Context::new(&problem);
      let mut rng = Xoshiro256StarStar::seed_from_u64(7);
      let mut sol = vrppd_cea::generate_rscim(
        &problem,
        &ctx.order_mat,
        &ctx.vstart_mat,
        Objective::Distance,
        &mut rng,
      );
      assert!(sol.is_valid(&problem));
      let n = problem.orders.len();
      for round in 0..20 {
        let mut orders: Vec<usize> = (0..n).collect();
        orders.shuffle(&mut rng);
        let removed: Vec<usize> = orders.into_iter().take(3 + round % 4).collect();
        destroy(&ctx, &mut sol, &removed);
        assert!(sol.is_valid(&problem));
        for &o in &removed {
          let mut cea_sol = sol.clone();
          let ins = best_insertion(&ctx, &sol, o).expect("insertion exists");
          let before = sol.total_distance;
          apply_insertion(&ctx, &mut sol, o, ins);
          assert!(sol.is_valid(&problem), "invalid after insertion");
          assert!(
            (sol.total_distance - before - ins.delta).abs() < 1e-6,
            "delta mismatch: {} vs {}",
            sol.total_distance - before,
            ins.delta
          );
          vrppd_cea::insert_cheapest(
            &mut cea_sol,
            o,
            &problem,
            &ctx.order_mat,
            &ctx.vstart_mat,
            Objective::Distance,
          );
          cea_sol.recalculate_all(&problem, &ctx.order_mat, &ctx.vstart_mat);
          assert!(
            (cea_sol.total_distance - sol.total_distance).abs() < 1e-6,
            "cea {} vs lns {}",
            cea_sol.total_distance,
            sol.total_distance
          );
        }
        let mut check = sol.clone();
        check.recalculate_all(&problem, &ctx.order_mat, &ctx.vstart_mat);
        assert!((check.total_distance - sol.total_distance).abs() < 1e-6);
      }
    }
  }

  #[test]
  fn removal_saving_matches_recalculation() {
    let problem = load_problem("20_50", 0);
    let ctx = Context::new(&problem);
    let sol = vrppd_cea::generate_rscim_seeded(
      &problem,
      &ctx.order_mat,
      &ctx.vstart_mat,
      Objective::Distance,
      3,
    );
    let pl = placements(&sol, ctx.n_orders());
    #[allow(clippy::needless_range_loop)]
    for o in 0..ctx.n_orders() {
      let p = pl[o].unwrap();
      let saving = removal_saving(&ctx, &sol, p, o);
      let mut trial = sol.clone();
      remove_order(&ctx, &mut trial, p.v_idx, o);
      assert!(((sol.total_distance - trial.total_distance) - saving).abs() < 1e-6);
      assert!(trial.is_valid(&problem));
    }
  }
}
