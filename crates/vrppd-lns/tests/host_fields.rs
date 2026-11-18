//! The pipeline's bookkeeping fields (`host`, `threads`, `cpu_list`) reach every
//! file the `lns` and `cea_ref` binaries write, and old reference files without
//! them still load. Needs the untracked instances in `problem-gen/problems/10_10`.

use std::path::PathBuf;
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
  assert_eq!(records[1]["instance"], "1");
  assert_eq!(records[1]["host"], "test-host");
  assert_eq!(records[1]["threads"], 1);
  assert_eq!(records[1]["cpu_list"], "0");
}
