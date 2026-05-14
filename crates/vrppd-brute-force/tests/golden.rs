//! Golden / property-based regression tests for the brute-force solver.
//!
//! Fixtures live under `tests/fixtures/`. Tests assert algebraic invariants
//! that any correct solution must satisfy, plus structural shape — so they
//! catch regressions without baking in hand-computed kilometre figures.

use std::collections::HashSet;
use std::path::PathBuf;

use vrppd_brute_force::solve;
use vrppd_core::{Problem, ProblemSolution, StopKind, haversine_km};

fn load_fixture(name: &str) -> Problem {
  let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  path.push("tests/fixtures");
  path.push(name);
  let raw =
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// A solution is internally consistent when:
/// - every order id appears exactly once as pickup and exactly once as delivery
///   across all routes,
/// - within each route, every order's pickup precedes its delivery,
/// - aggregate totals equal the sum of per-route totals,
/// - per-route `total_price` equals `total_distance * vehicle.price_km`.
fn assert_solution_consistent(problem: &Problem, sol: &ProblemSolution) {
  let mut seen_pickup: HashSet<u32> = HashSet::new();
  let mut seen_delivery: HashSet<u32> = HashSet::new();

  let mut sum_total = 0.0;
  let mut sum_empty = 0.0;
  let mut sum_price = 0.0;

  for (vehicle_id_str, route) in &sol.routes {
    let vehicle_id: u32 = vehicle_id_str
      .parse()
      .expect("route key should parse as vehicle id");
    let vehicle = problem
      .vehicles
      .iter()
      .find(|v| v.id == vehicle_id)
      .expect("route key references a known vehicle");

    let mut route_pickups: HashSet<u32> = HashSet::new();
    for stop in &route.stops {
      match stop.kind {
        StopKind::Pickup => {
          assert!(
            route_pickups.insert(stop.order_id),
            "order {} picked up twice in vehicle {}",
            stop.order_id,
            vehicle_id
          );
          assert!(
            seen_pickup.insert(stop.order_id),
            "order {} picked up across multiple vehicles",
            stop.order_id
          );
        }
        StopKind::Delivery => {
          assert!(
            route_pickups.contains(&stop.order_id),
            "order {} delivered before pickup in vehicle {}",
            stop.order_id,
            vehicle_id
          );
          assert!(
            seen_delivery.insert(stop.order_id),
            "order {} delivered across multiple vehicles",
            stop.order_id
          );
        }
      }
    }

    let approx = (route.total_distance * vehicle.price_km - route.total_price).abs();
    assert!(
      approx < 1e-9,
      "route {vehicle_id}: total_price {} != total_distance {} * price_km {}",
      route.total_price,
      route.total_distance,
      vehicle.price_km
    );

    sum_total += route.total_distance;
    sum_empty += route.empty_distance;
    sum_price += route.total_price;
  }

  assert!((sum_total - sol.total_distance).abs() < 1e-9);
  assert!((sum_empty - sol.empty_distance).abs() < 1e-9);
  assert!((sum_price - sol.total_price).abs() < 1e-9);
}

#[test]
fn single_vehicle_single_order_is_solved_optimally() {
  let problem = load_fixture("single_vehicle_single_order.json");
  let sol = solve(&problem);

  for variant_sol in [
    &sol.best_distance_solution,
    &sol.best_empty_solution,
    &sol.best_price_solution,
  ] {
    assert_eq!(
      variant_sol.routes.len(),
      1,
      "exactly one vehicle should be used"
    );
    assert_solution_consistent(&problem, variant_sol);

    let route = variant_sol.routes.values().next().unwrap();
    assert_eq!(route.stops.len(), 2, "stops = pickup + delivery");
    assert_eq!(route.stops[0].kind, StopKind::Pickup);
    assert_eq!(route.stops[1].kind, StopKind::Delivery);

    let order = &problem.orders[0];
    let vehicle = &problem.vehicles[0];
    let leg_to_pickup = haversine_km(&vehicle.start_location, &order.pickup_location);
    let leg_loaded = haversine_km(&order.pickup_location, &order.delivery_location);

    assert!((route.empty_distance - leg_to_pickup).abs() < 1e-9);
    assert!((route.total_distance - (leg_to_pickup + leg_loaded)).abs() < 1e-9);
  }

  // For an instance with no choice, all three objectives must match exactly.
  let d = sol.best_distance_solution.total_distance;
  let e = sol.best_empty_solution.total_distance;
  let p = sol.best_price_solution.total_distance;
  assert!((d - e).abs() < 1e-9 && (d - p).abs() < 1e-9);
}

#[test]
fn two_vehicles_three_orders_is_internally_consistent() {
  let problem = load_fixture("two_vehicles_three_orders.json");
  let sol = solve(&problem);

  for variant_sol in [
    &sol.best_distance_solution,
    &sol.best_empty_solution,
    &sol.best_price_solution,
  ] {
    assert!(
      !variant_sol.routes.is_empty(),
      "feasible instance must produce routes"
    );
    assert_solution_consistent(&problem, variant_sol);

    // Every order must be served exactly once across all routes.
    let mut delivered: HashSet<u32> = HashSet::new();
    for route in variant_sol.routes.values() {
      for stop in &route.stops {
        if stop.kind == StopKind::Delivery {
          assert!(delivered.insert(stop.order_id));
        }
      }
    }
    assert_eq!(delivered.len(), problem.orders.len());
  }

  // Each per-objective solution must be at least as good as the others on its
  // own metric.
  assert!(
    sol.best_distance_solution.total_distance <= sol.best_price_solution.total_distance + 1e-9
  );
  assert!(
    sol.best_distance_solution.total_distance <= sol.best_empty_solution.total_distance + 1e-9
  );
  assert!(
    sol.best_empty_solution.empty_distance <= sol.best_distance_solution.empty_distance + 1e-9
  );
  assert!(sol.best_empty_solution.empty_distance <= sol.best_price_solution.empty_distance + 1e-9);
  assert!(sol.best_price_solution.total_price <= sol.best_distance_solution.total_price + 1e-9);
  assert!(sol.best_price_solution.total_price <= sol.best_empty_solution.total_price + 1e-9);
}

#[test]
fn empty_problem_returns_default_solution() {
  let problem = Problem {
    vehicles: vec![],
    orders: vec![],
  };
  let sol = solve(&problem);
  // No orders => infeasibility flag is set, default solutions are returned.
  assert_eq!(sol.best_distance_solution.routes.len(), 0);
  assert_eq!(sol.best_distance_solution.total_distance, 0.0);
}
