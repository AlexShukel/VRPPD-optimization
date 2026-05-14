use vrppd_core::{Location, Order, Vehicle, haversine_km};

use crate::types::InternalBestResults;

pub struct SolverContext<'a> {
  pub orders: &'a Vec<Order>,
  pub vehicles: &'a Vec<Vehicle>,

  // Flattened matrices for cache locality.
  pub dist_mat: Vec<f64>,
  pub num_nodes: usize,
  pub veh_start_mat: Vec<f64>,

  // Memoisation table for the inner TSP.
  pub memo: Vec<Option<InternalBestResults>>,
  pub n_orders: usize,

  pub best_dist: f64,
  pub best_dist_assignments: Vec<u32>,

  pub best_price: f64,
  pub best_price_assignments: Vec<u32>,

  pub best_empty: f64,
  pub best_empty_assignments: Vec<u32>,

  pub full_mask: u32,
}

impl<'a> SolverContext<'a> {
  pub fn new(orders: &'a Vec<Order>, vehicles: &'a Vec<Vehicle>) -> Self {
    let n_orders = orders.len();
    let num_nodes = n_orders * 2;

    // Order-order distance matrix (pickup-pickup, pickup-delivery, etc.).
    let mut dist_mat = vec![0.0; num_nodes * num_nodes];

    let get_loc = |idx: usize| -> &Location {
      let order_idx = idx / 2;
      if idx % 2 == 0 {
        &orders[order_idx].pickup_location
      } else {
        &orders[order_idx].delivery_location
      }
    };

    for i in 0..num_nodes {
      for j in 0..num_nodes {
        if i != j {
          dist_mat[i * num_nodes + j] = haversine_km(get_loc(i), get_loc(j));
        }
      }
    }

    // Vehicle-start to order-pickup distances.
    let mut veh_start_mat = vec![0.0; vehicles.len() * n_orders.max(1)];
    for (v_idx, vehicle) in vehicles.iter().enumerate() {
      for (o_idx, order) in orders.iter().enumerate() {
        veh_start_mat[v_idx * n_orders + o_idx] =
          haversine_km(&vehicle.start_location, &order.pickup_location);
      }
    }

    let cache_size = vehicles.len() * (1 << n_orders);

    SolverContext {
      orders,
      vehicles,
      dist_mat,
      num_nodes,
      veh_start_mat,
      memo: vec![None; cache_size],
      n_orders,

      best_dist: f64::INFINITY,
      best_dist_assignments: vec![0; vehicles.len()],

      best_price: f64::INFINITY,
      best_price_assignments: vec![0; vehicles.len()],

      best_empty: f64::INFINITY,
      best_empty_assignments: vec![0; vehicles.len()],

      full_mask: if n_orders == 0 {
        0
      } else {
        (1 << n_orders) - 1
      },
    }
  }
}
