//! Large Neighbourhood Search for the VRPPD (distance objective) with
//! pluggable destroy operators (random, worst, related, adaptive, neural),
//! supervised label generation and a pure-Rust MLP forward pass.

pub mod destroy;
pub mod features;
pub mod io;
pub mod label;
pub mod lns;
pub mod mlp;
pub mod solution;

pub use destroy::{DestroyOperator, OPERATOR_NAMES, build_operator};
pub use features::{FEATURE_NAMES, N_FEATURES, compute_features};
pub use label::{LabelConfig, LabelledState, label_instance};
pub use lns::{LnsConfig, LnsResult, default_k, initial_solution, run_lns};
pub use mlp::Mlp;
pub use solution::Context;
