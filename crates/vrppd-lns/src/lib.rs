//! Large Neighbourhood Search for the VRPPD (distance, price or empty-distance
//! objective) with pluggable destroy operators (random, worst, related,
//! adaptive, neural, adaptive_n), supervised label generation and pure-Rust
//! forward passes of the B0 MLP and the V1 encoder.

pub mod destroy;
pub mod encoder;
pub mod features;
pub mod io;
pub mod label;
pub mod lns;
pub mod mlp;
pub mod scorer;
pub mod solution;

pub use destroy::{DestroyOperator, OPERATOR_NAMES, build_operator};
pub use encoder::Encoder;
pub use features::{FEATURE_NAMES, FeatureCache, N_FEATURES, compute_features};
pub use label::{LabelConfig, LabelledState, label_instance};
pub use lns::{LnsConfig, LnsResult, default_k, distance_name, initial_solution, k_for, run_lns};
pub use mlp::Mlp;
pub use scorer::{EncoderScorer, MlpScorer, Scorer, load_scorer};
pub use solution::{Context, objective_name, parse_objective};
