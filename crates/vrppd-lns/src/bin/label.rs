//! Label generation CLI: random-removal LNS states + M random removal tries
//! per state, written as JSONL (one state per line).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;
use vrppd_lns::io::{cell_name, init_threads, load_instances, resolve_cell_dir};
use vrppd_lns::{Context, LabelConfig, label_instance};

#[derive(Parser, Debug)]
#[command(name = "label", about = "Generate supervised destroy labels (JSONL).")]
struct Args {
  /// Directory of instance JSON files (or thesis cell name).
  #[arg(long)]
  cell: String,
  /// Split label stored in every line (train | val).
  #[arg(long, default_value = "train")]
  split: String,
  /// Iterations of the random-removal LNS producing the states.
  #[arg(long, default_value_t = 2000)]
  lns_iters: u64,
  /// States per instance (evenly spread over the run, last = final state).
  #[arg(long, default_value_t = 10)]
  states: usize,
  /// Random removal sets per state.
  #[arg(long, default_value_t = 100)]
  m: usize,
  /// Destroy size (0 = 15 %, min 3).
  #[arg(long, default_value_t = 0)]
  k: usize,
  #[arg(long, default_value_t = 1)]
  seed: u64,
  /// Snapshot schedule: log | even.
  #[arg(long, default_value = "log")]
  schedule: String,
  /// Output JSONL path.
  #[arg(long)]
  out: PathBuf,
  #[arg(long)]
  max_instances: Option<usize>,
  /// Worker threads (hard-capped at 6).
  #[arg(long, default_value_t = 6)]
  threads: usize,
}

fn main() {
  let args = Args::parse();
  let threads = init_threads(args.threads, 6);
  let dir = resolve_cell_dir(&args.cell);
  let cell = cell_name(&dir);
  let instances = load_instances(&dir, args.max_instances).unwrap_or_else(|e| {
    eprintln!("{e}");
    std::process::exit(1)
  });
  if let Some(parent) = args.out.parent() {
    fs::create_dir_all(parent).expect("create out dir");
  }
  let tmp_path = args.out.with_extension("jsonl.tmp");
  let writer = Mutex::new(BufWriter::new(File::create(&tmp_path).expect("create out")));
  eprintln!(
    "label: cell={cell} split={} instances={} lns_iters={} states={} m={} threads={threads}",
    args.split,
    instances.len(),
    args.lns_iters,
    args.states,
    args.m
  );
  let started = Instant::now();
  let n_states: usize = instances
    .par_iter()
    .enumerate()
    .map(|(i, inst)| {
      let ctx = Context::new(&inst.problem);
      let cfg = LabelConfig {
        lns_iters: args.lns_iters,
        states_per_instance: args.states,
        m: args.m,
        k: if args.k > 0 { Some(args.k) } else { None },
        seed: args.seed + i as u64,
        schedule: args.schedule.clone(),
      };
      let name = format!("{}_{}_{}", args.split, cell, inst.id);
      let states = label_instance(&ctx, &name, &cell, &args.split, &cfg);
      let mut text = String::new();
      for s in &states {
        text.push_str(&serde_json::to_string(s).unwrap());
        text.push('\n');
      }
      let mut w = writer.lock().unwrap();
      w.write_all(text.as_bytes()).expect("write");
      states.len()
    })
    .sum();
  writer.lock().unwrap().flush().expect("flush");
  drop(writer);
  fs::rename(&tmp_path, &args.out).expect("rename");
  eprintln!(
    "label: {} states from {} instances in {:.1}s -> {}",
    n_states,
    instances.len(),
    started.elapsed().as_secs_f64(),
    args.out.display()
  );
}
