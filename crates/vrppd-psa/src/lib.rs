//! Parallel Simulated Annealing solver for the adapted VRPPD variant.

pub mod config;
pub mod operators;
pub mod pipeline;
pub mod rcrs;
pub mod sa;

pub use config::{default_config_for, OperatorWeights, SaConfig};
pub use pipeline::{solve_pipeline, solve_pipeline_seeded};
pub use sa::{solve, solve_seeded, ConvergencePoint, Solved};
