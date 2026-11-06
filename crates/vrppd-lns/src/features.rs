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

use vrppd_core::{StopKind, WorkingSolution, stop_node};

use crate::solution::{Context, placements, removal_saving};

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

/// Compute the feature matrix (row-major, n_orders x N_FEATURES).
/// Unrouted orders (should not happen in a complete solution) get zeros.
pub fn compute_features(ctx: &Context, sol: &WorkingSolution) -> Vec<f64> {
  let n = ctx.n_orders();
  let pl = placements(sol, n);
  let mut out = vec![0.0; n * N_FEATURES];
  for (o, place) in pl.iter().enumerate() {
    let Some(p) = place else { continue };
    let route = &sol.routes[p.v_idx];
    let len = route.stops.len();
    let order = &ctx.problem.orders[o];
    let vehicle = &ctx.problem.vehicles[p.v_idx];
    let row = &mut out[o * N_FEATURES..(o + 1) * N_FEATURES];
    row[0] = removal_saving(ctx, sol, *p, o);
    row[1] = ctx.vstart_mat.get(p.v_idx, o);
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
  out
}
