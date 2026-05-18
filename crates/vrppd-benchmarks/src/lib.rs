//! Shared helpers for the benchmark binaries: workspace path resolution, instance loading, run-id construction, results-directory writes, and CLI parsing of objective + cell-list flags.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use vrppd_core::{Objective, Problem};

pub fn workspace_root() -> PathBuf {
  let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  match manifest_dir.parent().and_then(|p| p.parent()) {
    Some(p) => p.to_path_buf(),
    None => manifest_dir,
  }
}

pub fn problem_dir() -> PathBuf {
  workspace_root().join("problem-gen").join("problems")
}

pub fn results_root() -> PathBuf {
  workspace_root().join("results")
}

#[derive(Clone, Debug, Serialize)]
pub struct Instance {
  pub cell: String,
  pub id: String,
  pub path: PathBuf,
  #[serde(skip)]
  pub problem: Problem,
}

pub fn load_cell_instances(cell: &str) -> io::Result<Vec<Instance>> {
  let dir = problem_dir().join(cell);
  if !dir.exists() {
    return Err(io::Error::new(
      io::ErrorKind::NotFound,
      format!("cell directory not found: {}", dir.display()),
    ));
  }
  let mut entries: Vec<_> = fs::read_dir(&dir)?
    .filter_map(|e| e.ok())
    .filter(|e| {
      e.path()
        .extension()
        .map(|ext| ext == "json")
        .unwrap_or(false)
    })
    .collect();
  entries.sort_by_key(|e| e.path());

  let mut out = Vec::with_capacity(entries.len());
  for entry in entries {
    let path = entry.path();
    let stem = path
      .file_stem()
      .and_then(|s| s.to_str())
      .unwrap_or("")
      .to_string();
    let id = stem.split('_').next().unwrap_or(&stem).to_string();
    let text = fs::read_to_string(&path)?;
    let problem: Problem = serde_json::from_str(&text).map_err(|e| {
      io::Error::new(
        io::ErrorKind::InvalidData,
        format!("parse {}: {}", path.display(), e),
      )
    })?;
    out.push(Instance {
      cell: cell.to_string(),
      id,
      path,
      problem,
    });
  }
  Ok(out)
}

pub fn parse_cells(s: &str) -> Vec<String> {
  s.split(',')
    .map(|c| c.trim().to_string())
    .filter(|c| !c.is_empty())
    .collect()
}

pub fn parse_objective(s: &str) -> Result<Objective, String> {
  match s.to_ascii_lowercase().as_str() {
    "distance" | "dist" | "d" => Ok(Objective::Distance),
    "price" | "p" => Ok(Objective::Price),
    "empty" | "e" => Ok(Objective::Empty),
    other => Err(format!(
      "unknown objective {other:?} (expected distance|price|empty)"
    )),
  }
}

pub fn make_run_id(prefix: &str) -> String {
  let ts = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs())
    .unwrap_or(0);
  format!("{prefix}-{ts}")
}

pub fn make_run_dir(prefix: &str) -> io::Result<PathBuf> {
  let dir = results_root().join(make_run_id(prefix));
  fs::create_dir_all(&dir)?;
  Ok(dir)
}

pub fn write_run_outputs(
  run_dir: &Path,
  results_json: &serde_json::Value,
  summary_md: &str,
) -> io::Result<()> {
  fs::write(
    run_dir.join("results.json"),
    serde_json::to_vec_pretty(results_json)?,
  )?;
  fs::write(run_dir.join("summary.md"), summary_md)?;
  Ok(())
}

pub fn mean(xs: &[f64]) -> f64 {
  if xs.is_empty() {
    return 0.0;
  }
  xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn stdev(xs: &[f64]) -> f64 {
  if xs.len() < 2 {
    return 0.0;
  }
  let m = mean(xs);
  let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
  var.sqrt()
}

pub fn min_f(xs: &[f64]) -> f64 {
  xs.iter().cloned().fold(f64::INFINITY, f64::min)
}

pub fn max_f(xs: &[f64]) -> f64 {
  xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}
