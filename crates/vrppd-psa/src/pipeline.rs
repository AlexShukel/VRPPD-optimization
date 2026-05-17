//! Multi-thread pipeline SA driver with worker-to-worker influence forwarding.

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use rand::{thread_rng, Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

use vrppd_core::{Objective, Problem};
use vrppd_core::{OrderMatrix, VehicleStartMatrix, WorkingSolution};

use crate::config::SaConfig;
use crate::operators::generate_neighbor;
use crate::rcrs::generate_rcrs;
use crate::sa::{ConvergencePoint, Solved};

pub fn solve_pipeline(problem: &Problem, target: Objective, config: SaConfig) -> Solved {
  let mut rng = thread_rng();
  let seed: u64 = rng.r#gen();
  solve_pipeline_seeded(problem, target, config, seed)
}

pub fn solve_pipeline_seeded(
  problem: &Problem,
  target: Objective,
  config: SaConfig,
  seed: u64,
) -> Solved {
  let threads = config.threads.max(1);

  let order_mat = Arc::new(OrderMatrix::build(&problem.orders));
  let vstart_mat = Arc::new(VehicleStartMatrix::build(
    &problem.vehicles,
    &problem.orders,
  ));
  let problem = Arc::new(problem.clone());

  let mut master_rng = Xoshiro256StarStar::seed_from_u64(seed);
  let initial = generate_rcrs(&problem, &order_mat, &vstart_mat, target, &mut master_rng);

  let started = Instant::now();
  let mut history = vec![ConvergencePoint::from_solution(0.0, 0, &initial)];

  let (report_tx, report_rx): (Sender<Report>, Receiver<Report>) = unbounded();
  let mut influence_txs: Vec<Sender<Influence>> = Vec::with_capacity(threads);
  let mut influence_rxs: Vec<Receiver<Influence>> = Vec::with_capacity(threads);
  for _ in 0..threads {
    let (tx, rx) = bounded::<Influence>(1);
    influence_txs.push(tx);
    influence_rxs.push(rx);
  }

  let mut handles = Vec::with_capacity(threads);
  for (worker_idx, inbox) in influence_rxs.into_iter().enumerate() {
    let problem = Arc::clone(&problem);
    let order_mat = Arc::clone(&order_mat);
    let vstart_mat = Arc::clone(&vstart_mat);
    let initial = initial.clone();
    let report_tx = report_tx.clone();
    let worker_seed = seed
      .wrapping_add(worker_idx as u64)
      .wrapping_add(0xDEAD_BEEF);

    let handle = thread::spawn(move || {
      run_worker(
        worker_idx,
        problem,
        order_mat,
        vstart_mat,
        initial,
        target,
        config,
        worker_seed,
        inbox,
        report_tx,
      );
    });
    handles.push(handle);
  }
  drop(report_tx);

  let mut global_best: Option<WorkingSolution> = None;
  let mut global_best_energy = f64::INFINITY;
  let mut alive = threads;

  while let Ok(report) = report_rx.recv() {
    if report.energy < global_best_energy {
      global_best_energy = report.energy;
      global_best = Some(report.solution.clone());
      history.push(ConvergencePoint::from_solution(
        started.elapsed().as_secs_f64() * 1_000.0,
        report.iteration,
        &report.solution,
      ));
    }

    if matches!(report.kind, ReportKind::Sync) {
      let next = report.worker_idx + 1;
      if next < threads {
        let _ = influence_txs[next].try_send(Influence {
          solution: report.solution,
          energy: report.energy,
        });
      }
    } else {
      alive -= 1;
      if alive == 0 {
        break;
      }
    }
  }

  for h in handles {
    let _ = h.join();
  }

  let best = global_best.unwrap_or(initial);
  Solved {
    solution: best.into_problem_solution(&problem),
    history,
  }
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
  worker_idx: usize,
  problem: Arc<Problem>,
  order_mat: Arc<OrderMatrix>,
  vstart_mat: Arc<VehicleStartMatrix>,
  initial: WorkingSolution,
  target: Objective,
  config: SaConfig,
  seed: u64,
  inbox: Receiver<Influence>,
  outbox: Sender<Report>,
) {
  let mut rng = Xoshiro256StarStar::seed_from_u64(seed);

  let mut current = initial;
  let mut current_energy = energy(&current, target);
  let mut best = current.clone();
  let mut best_energy = current_energy;

  let temp_jitter: f64 = rng.r#gen_range(0.9..1.1);
  let mut temperature = config.initial_temp * temp_jitter;

  let mut iteration: u64 = 0;
  let mut batch_count: u64 = 0;

  while iteration < config.max_iterations && temperature >= config.min_temp {
    while let Ok(inf) = inbox.try_recv() {
      if inf.energy < current_energy {
        current = inf.solution;
        current_energy = inf.energy;
        if current_energy < best_energy {
          best_energy = current_energy;
          best = current.clone();
        }
        temperature = temperature.max(config.reheat_floor);
      }
    }

    for _ in 0..config.batch_size {
      iteration += 1;

      if let Some(neighbor) = generate_neighbor(
        &current,
        &problem,
        &order_mat,
        &vstart_mat,
        config.weights,
        &mut rng,
      ) {
        let neighbor_energy = energy(&neighbor, target);
        let delta = neighbor_energy - current_energy;
        let accept = delta < 0.0 || {
          let p: f64 = rng.r#gen();
          p < (-delta / temperature).exp()
        };
        if accept {
          current = neighbor;
          current_energy = neighbor_energy;
          if current_energy < best_energy {
            best_energy = current_energy;
            best = current.clone();
          }
        }
      }

      temperature *= config.cooling_rate;

      if iteration >= config.max_iterations || temperature < config.min_temp {
        break;
      }
    }

    batch_count += 1;
    if batch_count % config.sync_interval == 0 {
      let _ = outbox.send(Report {
        worker_idx,
        kind: ReportKind::Sync,
        energy: best_energy,
        iteration,
        solution: best.clone(),
      });
    }
  }

  let _ = outbox.send(Report {
    worker_idx,
    kind: ReportKind::Done,
    energy: best_energy,
    iteration,
    solution: best,
  });
}

#[inline(always)]
fn energy(sol: &WorkingSolution, target: Objective) -> f64 {
  match target {
    Objective::Empty => sol.empty_distance,
    Objective::Distance => sol.total_distance,
    Objective::Price => sol.total_price,
  }
}

#[derive(Clone, Debug)]
struct Influence {
  solution: WorkingSolution,
  energy: f64,
}

#[derive(Clone, Debug)]
struct Report {
  worker_idx: usize,
  kind: ReportKind,
  energy: f64,
  iteration: u64,
  solution: WorkingSolution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReportKind {
  Sync,
  Done,
}
