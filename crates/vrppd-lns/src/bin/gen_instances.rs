//! Seedable instance generator with the same distribution as
//! `problem-gen/generate-problems.ts`: sample `V` vehicles and `O` orders
//! without replacement from the master pools in `problem-gen/data/`.

use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar;
use vrppd_core::{Order, Problem, Vehicle};
use vrppd_lns::io::workspace_root;

#[derive(Parser, Debug)]
#[command(
  name = "gen_instances",
  about = "Seeded VRPPD instance generator (same pools as the thesis generator)."
)]
struct Args {
  #[arg(long)]
  vehicles: usize,
  #[arg(long)]
  orders: usize,
  #[arg(long)]
  count: usize,
  #[arg(long)]
  seed: u64,
  /// Output directory; files are <i>.json.
  #[arg(long)]
  out: PathBuf,
  /// Pool directory (default problem-gen/data).
  #[arg(long)]
  data_dir: Option<PathBuf>,
}

fn latest(dir: &std::path::Path, prefix: &str) -> PathBuf {
  let mut best: Option<(u64, PathBuf)> = None;
  for e in std::fs::read_dir(dir).expect("read data dir").flatten() {
    let p = e.path();
    let stem = p
      .file_stem()
      .and_then(|s| s.to_str())
      .unwrap_or("")
      .to_string();
    if let Some(rest) = stem.strip_prefix(&format!("{prefix}_"))
      && let Ok(ts) = rest.parse::<u64>()
      && best.as_ref().is_none_or(|b| ts > b.0)
    {
      best = Some((ts, p.clone()));
    }
  }
  best
    .map(|b| b.1)
    .unwrap_or_else(|| panic!("no {prefix}_<ts>.json in {}", dir.display()))
}

fn main() {
  let args = Args::parse();
  let data_dir = args
    .data_dir
    .unwrap_or_else(|| workspace_root().join("problem-gen/data"));
  let orders: Vec<Order> =
    serde_json::from_str(&std::fs::read_to_string(latest(&data_dir, "orders")).unwrap()).unwrap();
  let vehicles: Vec<Vehicle> =
    serde_json::from_str(&std::fs::read_to_string(latest(&data_dir, "vehicles")).unwrap()).unwrap();
  assert!(args.vehicles <= vehicles.len() && args.orders <= orders.len());
  std::fs::create_dir_all(&args.out).expect("create out dir");
  let mut rng = Xoshiro256StarStar::seed_from_u64(args.seed);
  for i in 0..args.count {
    let v: Vec<Vehicle> = vehicles
      .choose_multiple(&mut rng, args.vehicles)
      .cloned()
      .collect();
    let o: Vec<Order> = orders
      .choose_multiple(&mut rng, args.orders)
      .cloned()
      .collect();
    let problem = Problem {
      vehicles: v,
      orders: o,
    };
    std::fs::write(
      args.out.join(format!("{i}.json")),
      serde_json::to_string_pretty(&problem).unwrap(),
    )
    .expect("write instance");
  }
  eprintln!(
    "gen_instances: {} instances {}x{} seed={} -> {}",
    args.count,
    args.vehicles,
    args.orders,
    args.seed,
    args.out.display()
  );
}
