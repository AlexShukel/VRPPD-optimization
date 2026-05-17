//! Warm-start decoder for the MILP solver.
//!
//! Converts a [`vrppd_core::ProblemSolution`] into a flat column-value vector compatible with
//! `highs::Model::set_solution`. The vector is indexed exactly as the
//! columns were added in `build_milp`

use vrppd_core::{Problem, ProblemSolution, StopKind};

use crate::ModelLayout;

pub(crate) fn decode(
  problem: &Problem,
  warm_start: &ProblemSolution,
  layout: &ModelLayout,
) -> Vec<f64> {
  let mut vals = vec![0.0_f64; layout.num_cols];

  let vehicle_index_for: std::collections::HashMap<u32, usize> = problem
    .vehicles
    .iter()
    .enumerate()
    .map(|(i, v)| (v.id, i))
    .collect();

  let order_index_for: std::collections::HashMap<u32, usize> = problem
    .orders
    .iter()
    .enumerate()
    .map(|(i, o)| (o.id, i))
    .collect();

  let ix = &layout.ix;

  for (vehicle_id_str, route) in &warm_start.routes {
    let vehicle_id: u32 = match vehicle_id_str.parse() {
      Ok(id) => id,
      Err(_) => continue,
    };
    let v = match vehicle_index_for.get(&vehicle_id) {
      Some(&vi) => vi,
      None => continue,
    };

    if route.stops.is_empty() {
      continue;
    }

    let stop_nodes: Vec<(usize, usize, StopKind)> = route
      .stops
      .iter()
      .filter_map(|s| {
        let o = *order_index_for.get(&s.order_id)?;
        let node = match s.kind {
          StopKind::Pickup => ix.pickup(o),
          StopKind::Delivery => ix.delivery(o),
        };
        Some((node, o, s.kind))
      })
      .collect();

    // y_ov
    for s in &route.stops {
      if s.kind == StopKind::Pickup {
        if let Some(&o) = order_index_for.get(&s.order_id) {
          if let Some(&col) = layout.y.get(&(o, v)) {
            vals[col] = 1.0;
          }
        }
      }
    }

    // x_ijv.
    if !stop_nodes.is_empty() {
      let s_node = ix.start(v);
      let first_node = stop_nodes[0].0;

      if let Some(&col) = layout.x.get(&(s_node, first_node, v)) {
        vals[col] = 1.0;
      }

      for w in stop_nodes.windows(2) {
        let i = w[0].0;
        let j = w[1].0;
        if let Some(&col) = layout.x.get(&(i, j, v)) {
          vals[col] = 1.0;
        }
      }

      let last_node = stop_nodes.last().unwrap().0;
      if let Some(&col) = layout.x.get(&(last_node, s_node, v)) {
        vals[col] = 1.0;
      }
    }

    // q_iv
    {
      let mut load = 0.0_f64;
      for &(node, o, kind) in &stop_nodes {
        let delta = 1.0 / problem.orders[o].load_factor;
        load += match kind {
          StopKind::Pickup => delta,
          StopKind::Delivery => -delta,
        };
        if let Some(&col) = layout.q.get(&(node, v)) {
          debug_assert!(
            load >= -f64::EPSILON,
            "warm-start route walk produced negative load {load} at node {node}; \
             upstream solver returned a route with delivery before pickup or wrong load_factor"
          );
          let clamped_load = load.max(0.0);
          vals[col] = clamped_load;
        }
      }
    }

    // u_iv
    for (pos, &(node, _o, _kind)) in stop_nodes.iter().enumerate() {
      let u_val = (pos + 1) as f64;
      if let Some(&col) = layout.u.get(&(node, v)) {
        vals[col] = u_val;
      }
    }
  }

  vals
}
