//! Coevolutionary Algorithm (CEA) for the heterogeneous-fleet point-to-point VRPPD variant.

pub mod coevolve;
pub mod config;
pub mod crossover;
pub mod fitness;
pub mod local_improvement;
pub mod population;
pub mod recombination;
pub mod reproduction;
pub mod rscim;

pub use coevolve::{solve_cea, solve_cea_seeded, ConvergencePoint, ExitReason, Solved};
pub use config::CeaConfig;
pub use crossover::crossover;
pub use fitness::{fitness_values, roulette_select, Fitness};
pub use local_improvement::local_improve;
pub use population::{Individual, Population};
pub use recombination::recombine;
pub use reproduction::reproduce_elite;
pub use rscim::{generate_rscim, generate_rscim_seeded};
