//! The eight hand-made B0 features per order (raw, unstandardised).
//!
//! 1. route saving if the order is removed (km)
//! 2. distance vehicle start -> pickup (km)
//! 3. pickup -> delivery distance (km)
//! 4. load factor
//! 5. vehicle price per km
//! 6. number of orders on the same route
//! 7. distance to the nearest pickup/delivery of another order (km)
//! 8. relative position in the route: mean of the two stop indices / (len - 1)

use vrppd_core::{StopKind, WorkingSolution, WorkingStop, stop_node};

use crate::solution::{Context, Placement, removal_saving};

pub const N_FEATURES: usize = 8;

pub const FEATURE_NAMES: [&str; N_FEATURES] = [
  "route_saving",
  "start_to_pickup",
  "pickup_to_delivery",
  "load_factor",
  "price_km",
  "orders_on_route",
  "nearest_other",
  "relative_position",
];

/// Fill the feature rows of every order on route `v_idx` (rows indexed by order).
fn fill_route_rows(ctx: &Context, sol: &WorkingSolution, v_idx: usize, out: &mut [f64]) {
  let route = &sol.routes[v_idx];
  let len = route.stops.len();
  let vehicle = &ctx.problem.vehicles[v_idx];
  // Positions of each order's stops on this route, in stop order.
  let mut places: Vec<(usize, Placement)> = Vec::with_capacity(len / 2);
  for (pos, stop) in route.stops.iter().enumerate() {
    let o = stop.order_idx;
    let slot = match places.iter().position(|(x, _)| *x == o) {
      Some(s) => s,
      None => {
        places.push((
          o,
          Placement {
            v_idx,
            pickup_pos: 0,
            delivery_pos: 0,
          },
        ));
        places.len() - 1
      }
    };
    match stop.kind {
      StopKind::Pickup => places[slot].1.pickup_pos = pos,
      StopKind::Delivery => places[slot].1.delivery_pos = pos,
    }
  }
  for (o, p) in places {
    let order = &ctx.problem.orders[o];
    let row = &mut out[o * N_FEATURES..(o + 1) * N_FEATURES];
    row[0] = removal_saving(ctx, sol, p, o);
    row[1] = ctx.vstart_mat.get(v_idx, o);
    row[2] = ctx.order_mat.get(
      stop_node(o, StopKind::Pickup),
      stop_node(o, StopKind::Delivery),
    );
    row[3] = order.load_factor;
    row[4] = vehicle.price_km;
    row[5] = (len / 2) as f64;
    row[6] = ctx.nearest_other[o];
    row[7] = if len > 1 {
      (p.pickup_pos + p.delivery_pos) as f64 / 2.0 / (len - 1) as f64
    } else {
      0.0
    };
  }
}

/// Compute the feature matrix (row-major, n_orders x N_FEATURES).
/// Unrouted orders (should not happen in a complete solution) get zeros.
pub fn compute_features(ctx: &Context, sol: &WorkingSolution) -> Vec<f64> {
  let mut out = vec![0.0; ctx.n_orders() * N_FEATURES];
  for v_idx in 0..sol.routes.len() {
    fill_route_rows(ctx, sol, v_idx, &mut out);
  }
  out
}

/// Feature matrix kept between LNS iterations: a destroy-repair step only
/// changes a few routes, so only the rows of orders on routes whose stop list
/// changed since the previous call are recomputed. A row depends on its own
/// route only (and on the static context), so the result equals
/// [`compute_features`] exactly. Use one cache per context and search run.
pub struct FeatureCache {
  routes: Vec<Vec<WorkingStop>>,
  feats: Vec<f64>,
  changed: Vec<usize>,
}

impl FeatureCache {
  /// Placeholder with no rows; replace with [`FeatureCache::new`] before use.
  pub(crate) fn empty() -> Self {
    Self {
      routes: Vec::new(),
      feats: Vec::new(),
      changed: Vec::new(),
    }
  }

  pub fn new(ctx: &Context) -> Self {
    Self {
      routes: vec![Vec::new(); ctx.n_vehicles()],
      feats: vec![0.0; ctx.n_orders() * N_FEATURES],
      changed: Vec::new(),
    }
  }

  /// Bring the cache up to `sol` and return the matrix (n_orders x N_FEATURES).
  pub fn update(&mut self, ctx: &Context, sol: &WorkingSolution) -> &[f64] {
    self.changed.clear();
    let dirty: Vec<usize> = (0..sol.routes.len())
      .filter(|&v| self.routes[v] != sol.routes[v].stops)
      .collect();
    // Zero first so that an order that left a route and is now unrouted (or
    // moved to another dirty route) is rewritten by the fill below.
    for &v in &dirty {
      for s in &self.routes[v] {
        let o = s.order_idx;
        self.feats[o * N_FEATURES..(o + 1) * N_FEATURES].fill(0.0);
        self.changed.push(o);
      }
    }
    for &v in &dirty {
      fill_route_rows(ctx, sol, v, &mut self.feats);
      for s in &sol.routes[v].stops {
        self.changed.push(s.order_idx);
      }
      self.routes[v].clone_from(&sol.routes[v].stops);
    }
    self.changed.sort_unstable();
    self.changed.dedup();
    &self.feats
  }

  /// The matrix as of the last [`update`](Self::update).
  pub fn features(&self) -> &[f64] {
    &self.feats
  }

  /// Orders whose row may have changed in the last [`update`](Self::update)
  /// (all orders on the first call that saw them).
  pub fn changed(&self) -> &[usize] {
    &self.changed
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::destroy::RandomDestroy;
  use crate::lns::{LnsConfig, initial_solution, run_lns};

  #[test]
  fn feature_cache_matches_full_recompute() {
    let dir =
      std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../problem-gen/problems/10_20");
    let mut files: Vec<_> = std::fs::read_dir(dir)
      .unwrap()
      .filter_map(|e| e.ok())
      .map(|e| e.path())
      .collect();
    files.sort();
    let problem: vrppd_core::Problem =
      serde_json::from_str(&std::fs::read_to_string(&files[0]).unwrap()).unwrap();
    let ctx = Context::new(&problem);
    let mut cache = FeatureCache::new(&ctx);
    let mut calls = 0;
    let mut hook = |_: u64, sol: &WorkingSolution, _: f64| {
      let cached = cache.update(&ctx, sol).to_vec();
      assert_eq!(cached, compute_features(&ctx, sol));
      calls += 1;
    };
    let cfg = LnsConfig {
      max_iters: Some(300),
      seed: 2,
      ..Default::default()
    };
    run_lns(
      &ctx,
      initial_solution(&ctx, 2),
      &mut RandomDestroy,
      &cfg,
      Some(&mut hook),
    );
    assert_eq!(calls, 300);
  }
}
