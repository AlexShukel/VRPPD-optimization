//! Exact brute-force solver for the specific VRPPD variant.
//!
//! Two-level search: outer level enumerates which subset of orders each vehicle
//! receives (with branch-and-bound pruning across all three objectives); inner
//! level solves a precedence-constrained TSP for each (vehicle, order-subset)
//! pair via memoised DFS over pickup/delivery state.

mod context;
mod tsp;
mod types;

use std::collections::HashMap;

use vrppd_core::{AlgorithmSolution, Problem, ProblemSolution, RouteStop, StopKind, VehicleRoute};

use context::SolverContext;
use tsp::solve_tsp;

/// Produces the best solution per objective (distance, price, empty distance)
/// for the given problem.
///
/// Returns an [`AlgorithmSolution`] with default sub-solutions if no feasible
/// assignment exists (e.g. when capacity constraints make the instance
/// infeasible).
pub fn solve(problem: &Problem) -> AlgorithmSolution {
  let mut ctx = SolverContext::new(&problem.orders, &problem.vehicles);
  let mut assignments = vec![0_u32; problem.vehicles.len()];

  solve_recursive(&mut ctx, 0, 0, 0.0, 0.0, 0.0, &mut assignments);

  let best_dist_vec = ctx.best_dist_assignments.clone();
  let best_price_vec = ctx.best_price_assignments.clone();
  let best_empty_vec = ctx.best_empty_assignments.clone();

  let dist_sol = if ctx.best_dist < f64::INFINITY {
    reconstruct_solution(&mut ctx, &best_dist_vec, Objective::Distance)
  } else {
    ProblemSolution::default()
  };
  let price_sol = if ctx.best_price < f64::INFINITY {
    reconstruct_solution(&mut ctx, &best_price_vec, Objective::Price)
  } else {
    ProblemSolution::default()
  };
  let empty_sol = if ctx.best_empty < f64::INFINITY {
    reconstruct_solution(&mut ctx, &best_empty_vec, Objective::Empty)
  } else {
    ProblemSolution::default()
  };

  AlgorithmSolution {
    best_distance_solution: dist_sol,
    best_price_solution: price_sol,
    best_empty_solution: empty_sol,
  }
}

#[derive(Clone, Copy)]
enum Objective {
  Distance,
  Price,
  Empty,
}

fn solve_recursive(
  ctx: &mut SolverContext,
  vehicle_idx: usize,
  assignment_mask: u32,
  current_dist: f64,
  current_price: f64,
  current_empty: f64,
  assignments: &mut Vec<u32>,
) {
  if current_dist >= ctx.best_dist
    && current_price >= ctx.best_price
    && current_empty >= ctx.best_empty
  {
    return;
  }

  if assignment_mask == ctx.full_mask {
    if current_dist < ctx.best_dist {
      ctx.best_dist = current_dist;
      ctx.best_dist_assignments.copy_from_slice(assignments);
    }
    if current_price < ctx.best_price {
      ctx.best_price = current_price;
      ctx.best_price_assignments.copy_from_slice(assignments);
    }
    if current_empty < ctx.best_empty {
      ctx.best_empty = current_empty;
      ctx.best_empty_assignments.copy_from_slice(assignments);
    }
    return;
  }

  if vehicle_idx >= ctx.vehicles.len() {
    return;
  }

  let remaining_mask = ctx.full_mask ^ assignment_mask;
  let mut submask = remaining_mask;

  loop {
    if submask == 0 {
      break;
    }

    let res = solve_tsp(ctx, vehicle_idx, submask);

    if res.valid {
      assignments[vehicle_idx] = submask;

      solve_recursive(
        ctx,
        vehicle_idx + 1,
        assignment_mask | submask,
        current_dist + res.min_dist.total_dist,
        current_price + res.min_price.total_price,
        current_empty + res.min_empty.total_empty,
        assignments,
      );

      assignments[vehicle_idx] = 0;
    }

    submask = (submask - 1) & remaining_mask;
    if submask == 0 {
      break;
    }
  }

  // Skip this vehicle entirely.
  solve_recursive(
    ctx,
    vehicle_idx + 1,
    assignment_mask,
    current_dist,
    current_price,
    current_empty,
    assignments,
  );
}

fn reconstruct_solution(
  ctx: &mut SolverContext,
  assignments: &[u32],
  objective: Objective,
) -> ProblemSolution {
  let mut solution = ProblemSolution {
    routes: HashMap::new(),
    total_distance: 0.0,
    empty_distance: 0.0,
    total_price: 0.0,
  };

  for (v_idx, &mask) in assignments.iter().enumerate() {
    if mask == 0 {
      continue;
    }
    let res = solve_tsp(ctx, v_idx, mask);
    if !res.valid {
      continue;
    }

    let internal_res = match objective {
      Objective::Distance => res.min_dist,
      Objective::Price => res.min_price,
      Objective::Empty => res.min_empty,
    };

    let mut stops = Vec::with_capacity(internal_res.path.len as usize);
    for i in 0..internal_res.path.len {
      let node = internal_res.path.nodes[i as usize];
      let order_id = ctx.orders[(node / 2) as usize].id;
      let kind = if node % 2 == 0 {
        StopKind::Pickup
      } else {
        StopKind::Delivery
      };
      stops.push(RouteStop { order_id, kind });
    }

    let route = VehicleRoute {
      stops,
      total_distance: internal_res.total_dist,
      empty_distance: internal_res.total_empty,
      total_price: internal_res.total_price,
    };

    solution.total_distance += route.total_distance;
    solution.total_price += route.total_price;
    solution.empty_distance += route.empty_distance;
    solution
      .routes
      .insert(ctx.vehicles[v_idx].id.to_string(), route);
  }

  solution
}
