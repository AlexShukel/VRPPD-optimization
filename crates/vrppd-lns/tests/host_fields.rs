//! The pipeline's bookkeeping fields (`host`, `threads`, `cpu_list`, the weights
//! of `lns` runs, the pool size and seeds of `cea_ref` runs) reach every file the
//! `lns` and `cea_ref` binaries write, and old reference files without them still
//! load. Needs the untracked instances in `problem-gen/problems/10_10`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fresh_dir(name: &str) -> PathBuf {
  let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
  let _ = std::fs::remove_dir_all(&dir);
  std::fs::create_dir_all(&dir).unwrap();
  dir
}

fn read_json(path: &PathBuf) -> serde_json::Value {
  serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn lns_run_files_record_host_threads_and_cpu_list() {
  let out = fresh_dir("lns_host_fields");
  let status = Command::new(env!("CARGO_BIN_EXE_lns"))
    .args(["--cell", "10_10", "--max-instances", "1", "--iters", "5"])
    .args([
      "--threads",
      "2",
      "--host-tag",
      "test-host",
      "--cpu-list",
      "0,2",
    ])
    .arg("--out")
    .arg(&out)
    .status()
    .unwrap();
  assert!(status.success());
  let run = read_json(&out.join("0_s1.json"));
  assert_eq!(run["host"], "test-host");
  assert_eq!(run["threads"], 2);
  assert_eq!(run["cpu_list"], "0,2");
}

#[test]
fn cea_ref_records_host_fields_and_keeps_old_records() {
  let dir = fresh_dir("cea_ref_host_fields");
  let out = dir.join("cea_ref_10_10.json");
  let old = r#"[{"cell":"10_10","instance":"0","n_orders":10,"n_vehicles":10,"runs":[1.0],
    "runtimes_ms":[1.0],"generations":[1],"exit_reasons":["Stagnation"],"best":1.0}]"#;
  std::fs::write(&out, old).unwrap();
  let status = Command::new(env!("CARGO_BIN_EXE_cea_ref"))
    .args([
      "--cell",
      "10_10",
      "--max-instances",
      "2",
      "--runs",
      "1",
      "--seconds",
      "1",
    ])
    .args([
      "--threads",
      "1",
      "--host-tag",
      "test-host",
      "--cpu-list",
      "0",
    ])
    .arg("--out")
    .arg(&out)
    .status()
    .unwrap();
  assert!(status.success());
  let records = read_json(&out);
  let records = records.as_array().unwrap();
  assert_eq!(records.len(), 2);
  assert_eq!(records[0]["best"], 1.0);
  assert_eq!(records[0]["host"], "");
  assert_eq!(records[0]["threads"], 0);
  assert_eq!(records[0]["cpu_list"], "");
  assert!(records[0].get("rayon_threads").is_none() && records[0].get("seeds").is_none());
  assert_eq!(records[1]["instance"], "1");
  assert_eq!(records[1]["host"], "test-host");
  assert_eq!(records[1]["threads"], 1);
  assert_eq!(records[1]["cpu_list"], "0");
  assert_eq!(records[1]["rayon_threads"], 1);
  // seed 7 (default), run 0, instance id "1" of length 1: (7 * 1_000_003 + 0) ^ 1
  assert_eq!(records[1]["seeds"], serde_json::json!([7_000_020u64]));
}

#[test]
fn cea_ref_pool_has_the_recorded_threads_whatever_rayon_num_threads_says() {
  let dir = fresh_dir("cea_ref_pool_size");
  let out = dir.join("cea_ref_10_10.json");
  let status = Command::new(env!("CARGO_BIN_EXE_cea_ref"))
    .env("RAYON_NUM_THREADS", "5") // run_all.sh exported max(ITER_THREADS, TIMED_THREADS)
    .args(["--cell", "10_10", "--max-instances", "1", "--runs", "2"])
    .args(["--seconds", "1", "--threads", "2"])
    .arg("--out")
    .arg(&out)
    .status()
    .unwrap();
  assert!(status.success());
  let records = read_json(&out);
  let record = &records.as_array().unwrap()[0];
  assert_eq!(record["threads"], 2);
  assert_eq!(record["rayon_threads"], 2);
  assert_eq!(record["seeds"].as_array().unwrap().len(), 2);
}

fn run_lns(out: &Path, weights: &Path, force: bool) -> bool {
  let mut cmd = Command::new(env!("CARGO_BIN_EXE_lns"));
  cmd
    .args(["--cell", "10_10", "--max-instances", "1", "--iters", "5"])
    .args(["--threads", "1", "--operator", "neural", "--weights"])
    .arg(weights)
    .arg("--out")
    .arg(out);
  if force {
    cmd.arg("--force");
  }
  cmd.status().unwrap().success()
}

#[test]
fn lns_records_weights_and_refuses_runs_of_other_weights() {
  let dir = fresh_dir("lns_weights_hash");
  let weights = dir.join("w.json");
  let fixture =
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlp_fixture_weights.json");
  std::fs::copy(&fixture, &weights).unwrap();
  let out = dir.join("runs");
  assert!(run_lns(&out, &weights, false));
  let run_path = out.join("0_s1.json");
  let run = read_json(&run_path);
  assert_eq!(run["weights"], weights.to_str().unwrap());
  // `shasum -a 256` of the fixture starts with 9c595e53d61c1f6d
  assert_eq!(run["weights_hash"], "9c595e53d61c1f6d");
  let written = std::fs::read(&run_path).unwrap();

  // the same weights: the run is skipped
  assert!(run_lns(&out, &weights, false));
  // other contents at the same path (the same network, reformatted): refused, file kept
  let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&fixture).unwrap()).unwrap();
  std::fs::write(&weights, serde_json::to_string_pretty(&value).unwrap()).unwrap();
  assert!(!run_lns(&out, &weights, false));
  assert_eq!(std::fs::read(&run_path).unwrap(), written);
  // --force reruns it with the new hash
  assert!(run_lns(&out, &weights, true));
  assert_ne!(read_json(&run_path)["weights_hash"], "9c595e53d61c1f6d");
}

#[test]
fn lns_refuses_a_neural_run_file_without_weights_hash() {
  let dir = fresh_dir("lns_weights_hash_missing");
  let weights =
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlp_fixture_weights.json");
  let out = dir.join("runs");
  assert!(run_lns(&out, &weights, false));
  let run_path = out.join("0_s1.json");
  let mut run = read_json(&run_path);
  run.as_object_mut().unwrap().remove("weights_hash");
  std::fs::write(&run_path, run.to_string()).unwrap();
  assert!(!run_lns(&out, &weights, false));
}
