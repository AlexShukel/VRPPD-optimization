//! JSON wire format shared between `lib.rs` and `python/solver.py`.

use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub(crate) struct SolverRequest<'a> {
  pub solver: &'a str,
  pub objective: &'a str,
  pub timeout_ms: u64,
  pub threads: usize,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub first_solution_strategy: Option<&'a str>,
  pub problem: WireProblem,
}

#[derive(Serialize)]
pub(crate) struct WireProblem {
  pub vehicles: Vec<WireVehicle>,
  pub orders: Vec<WireOrder>,
}

#[derive(Serialize)]
pub(crate) struct WireVehicle {
  pub id: u32,
  pub start_lat: f64,
  pub start_lon: f64,
  pub price_km: f64,
}

impl WireVehicle {
  pub fn from_core(v: &vrppd_core::Vehicle) -> Self {
    Self {
      id: v.id,
      start_lat: v.start_location.latitude,
      start_lon: v.start_location.longitude,
      price_km: v.price_km,
    }
  }
}

#[derive(Serialize)]
pub(crate) struct WireOrder {
  pub id: u32,
  pub pickup_lat: f64,
  pub pickup_lon: f64,
  pub delivery_lat: f64,
  pub delivery_lon: f64,
  pub load_factor: f64,
}

impl WireOrder {
  pub fn from_core(o: &vrppd_core::Order) -> Self {
    Self {
      id: o.id,
      pickup_lat: o.pickup_location.latitude,
      pickup_lon: o.pickup_location.longitude,
      delivery_lat: o.delivery_location.latitude,
      delivery_lon: o.delivery_location.longitude,
      load_factor: o.load_factor,
    }
  }
}

#[derive(Deserialize, Debug)]
pub(crate) struct SolverResponse {
  pub ok: bool,
  pub objective_value: Option<f64>,
  pub status: Option<String>,
  pub solver_runtime_ms: Option<u64>,
  pub error_kind: Option<String>,
  pub error_msg: Option<String>,
}
