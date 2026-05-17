use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Location {
  pub hash: String,
  pub latitude: f64,
  pub longitude: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Vehicle {
  pub id: u32,
  pub start_location: Location,
  pub price_km: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Order {
  pub id: u32,
  pub pickup_location: Location,
  pub delivery_location: Location,
  pub load_factor: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Problem {
  pub vehicles: Vec<Vehicle>,
  pub orders: Vec<Order>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StopKind {
  Pickup,
  Delivery,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Objective {
  #[serde(rename = "EMPTY")]
  Empty,
  #[serde(rename = "DISTANCE")]
  Distance,
  #[serde(rename = "PRICE")]
  Price,
}

impl Objective {
  pub fn as_str(self) -> &'static str {
    match self {
      Objective::Empty => "EMPTY",
      Objective::Distance => "DISTANCE",
      Objective::Price => "PRICE",
    }
  }

  pub fn energy(self, sol: &ProblemSolution) -> f64 {
    match self {
      Objective::Empty => sol.empty_distance,
      Objective::Distance => sol.total_distance,
      Objective::Price => sol.total_price,
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RouteStop {
  pub order_id: u32,
  #[serde(rename = "type")]
  pub kind: StopKind,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VehicleRoute {
  pub stops: Vec<RouteStop>,
  pub total_distance: f64,
  pub empty_distance: f64,
  pub total_price: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProblemSolution {
  // vehicle id => route
  pub routes: HashMap<String, VehicleRoute>,
  pub total_distance: f64,
  pub empty_distance: f64,
  pub total_price: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AlgorithmSolution {
  pub best_distance_solution: ProblemSolution,
  pub best_price_solution: ProblemSolution,
  pub best_empty_solution: ProblemSolution,
}
