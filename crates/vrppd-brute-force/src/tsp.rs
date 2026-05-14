use vrppd_core::Order;

use crate::context::SolverContext;
use crate::types::{InternalBestResults, InternalTspResult, PathBuffer};

pub fn solve_tsp(
  ctx: &mut SolverContext,
  vehicle_idx: usize,
  target_mask: u32,
) -> InternalBestResults {
  let cache_idx = vehicle_idx * (1 << ctx.n_orders) + target_mask as usize;

  // Memoised lookup. Bounds are guaranteed by SolverContext::new.
  let cached_opt = unsafe { ctx.memo.get_unchecked(cache_idx) };
  if let Some(cached) = cached_opt {
    return *cached;
  }

  let vehicle_price = ctx.vehicles[vehicle_idx].price_km;

  // (val, path, secondary, tertiary)
  let mut best_dist = (f64::INFINITY, PathBuffer::default(), 0.0, 0.0);
  let mut best_empty = (f64::INFINITY, PathBuffer::default(), 0.0, 0.0);
  let mut best_price = (f64::INFINITY, PathBuffer::default(), 0.0, 0.0);

  let mut path_stack = PathBuffer::default();

  // Many parameters are intentional: the inner DFS is the solver hot path and
  // we explicitly pass each piece of state by reference rather than capturing
  // through a closure for predictable performance.
  #[allow(clippy::too_many_arguments)]
  fn dfs(
    n_orders: usize,
    num_nodes: usize,
    veh_start: &[f64],
    dist_mat: &[f64],
    orders: &[Order],
    v_idx: usize,
    v_price: f64,
    target_mask: u32,
    last_node: Option<usize>,
    cur: (f64, f64, f64, f64), // (dist, empty, price, load)
    path: &mut PathBuffer,
    pickup_mask: u32,
    deliver_mask: u32,
    b_dist: &mut (f64, PathBuffer, f64, f64),
    b_empty: &mut (f64, PathBuffer, f64, f64),
    b_price: &mut (f64, PathBuffer, f64, f64),
  ) {
    let (c_dist, c_empty, c_price, c_load) = cur;

    if c_dist >= b_dist.0 && c_empty >= b_empty.0 && c_price >= b_price.0 {
      return;
    }

    if deliver_mask == target_mask {
      if c_dist < b_dist.0 {
        *b_dist = (c_dist, *path, c_empty, c_price);
      }
      if c_empty < b_empty.0 {
        *b_empty = (c_empty, *path, c_dist, c_price);
      }
      if c_price < b_price.0 {
        *b_price = (c_price, *path, c_dist, c_empty);
      }
      return;
    }

    for o_idx in 0..n_orders {
      let order_bit = 1 << o_idx;
      if (target_mask & order_bit) == 0 {
        continue;
      }

      let order = &orders[o_idx];
      let load_val = 1.0 / order.load_factor;

      // PICKUP
      if (pickup_mask & order_bit) == 0 {
        if c_load + load_val > 1.000_001 {
          continue;
        }

        let leg_dist = match last_node {
          None => veh_start[v_idx * n_orders + o_idx],
          Some(prev) => dist_mat[prev * num_nodes + (2 * o_idx)],
        };

        let is_empty = pickup_mask == deliver_mask;
        let add_empty = if is_empty { leg_dist } else { 0.0 };

        path.nodes[path.len as usize] = (2 * o_idx) as u8;
        path.len += 1;

        dfs(
          n_orders,
          num_nodes,
          veh_start,
          dist_mat,
          orders,
          v_idx,
          v_price,
          target_mask,
          Some(2 * o_idx),
          (
            c_dist + leg_dist,
            c_empty + add_empty,
            c_price + leg_dist * v_price,
            c_load + load_val,
          ),
          path,
          pickup_mask | order_bit,
          deliver_mask,
          b_dist,
          b_empty,
          b_price,
        );

        path.len -= 1;
      }
      // DELIVERY
      else if (pickup_mask & order_bit) != 0 && (deliver_mask & order_bit) == 0 {
        let prev = last_node.unwrap_or(0);
        let leg_dist = dist_mat[prev * num_nodes + (2 * o_idx + 1)];

        path.nodes[path.len as usize] = (2 * o_idx + 1) as u8;
        path.len += 1;

        dfs(
          n_orders,
          num_nodes,
          veh_start,
          dist_mat,
          orders,
          v_idx,
          v_price,
          target_mask,
          Some(2 * o_idx + 1),
          (
            c_dist + leg_dist,
            c_empty,
            c_price + leg_dist * v_price,
            c_load - load_val,
          ),
          path,
          pickup_mask,
          deliver_mask | order_bit,
          b_dist,
          b_empty,
          b_price,
        );

        path.len -= 1;
      }
    }
  }

  dfs(
    ctx.n_orders,
    ctx.num_nodes,
    &ctx.veh_start_mat,
    &ctx.dist_mat,
    ctx.orders,
    vehicle_idx,
    vehicle_price,
    target_mask,
    None,
    (0.0, 0.0, 0.0, 0.0),
    &mut path_stack,
    0,
    0,
    &mut best_dist,
    &mut best_empty,
    &mut best_price,
  );

  let result = if best_dist.0 < f64::INFINITY {
    InternalBestResults {
      min_dist: InternalTspResult {
        path: best_dist.1,
        total_dist: best_dist.0,
        total_empty: best_dist.2,
        total_price: best_dist.3,
      },
      min_empty: InternalTspResult {
        path: best_empty.1,
        total_dist: best_empty.2,
        total_empty: best_empty.0,
        total_price: best_empty.3,
      },
      min_price: InternalTspResult {
        path: best_price.1,
        total_dist: best_price.2,
        total_empty: best_price.3,
        total_price: best_price.0,
      },
      valid: true,
    }
  } else {
    let dummy = InternalTspResult {
      path: PathBuffer::default(),
      total_dist: 0.0,
      total_empty: 0.0,
      total_price: 0.0,
    };
    InternalBestResults {
      min_dist: dummy,
      min_empty: dummy,
      min_price: dummy,
      valid: false,
    }
  };

  unsafe {
    *ctx.memo.get_unchecked_mut(cache_idx) = Some(result);
  }

  result
}
