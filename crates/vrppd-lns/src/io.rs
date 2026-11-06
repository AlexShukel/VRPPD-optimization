//! Instance loading helpers shared by the CLI binaries.

use std::path::{Path, PathBuf};

use vrppd_core::Problem;

pub struct LoadedInstance {
  pub id: String,
  pub path: PathBuf,
  pub problem: Problem,
}

/// Workspace root (two levels above this crate's manifest dir).
pub fn workspace_root() -> PathBuf {
  let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
  manifest_dir
    .parent()
    .and_then(|p| p.parent())
    .map(|p| p.to_path_buf())
    .unwrap_or(manifest_dir)
}

/// Resolve `--cell`: either a directory path or a thesis cell name under
/// `problem-gen/problems/`.
pub fn resolve_cell_dir(cell: &str) -> PathBuf {
  let p = Path::new(cell);
  if p.is_dir() {
    return p.to_path_buf();
  }
  workspace_root().join("problem-gen/problems").join(cell)
}

/// Load every `*.json` instance in a directory, sorted by file name; the id is
/// the file stem up to the first `_` (thesis naming `<i>_<timestamp>.json`) or
/// the whole stem.
pub fn load_instances(dir: &Path, limit: Option<usize>) -> Result<Vec<LoadedInstance>, String> {
  let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
    .map_err(|e| format!("read dir {}: {e}", dir.display()))?
    .filter_map(|e| e.ok())
    .map(|e| e.path())
    .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
    .collect();
  files.sort_by_key(|p| {
    let stem = p
      .file_stem()
      .and_then(|s| s.to_str())
      .unwrap_or("")
      .to_string();
    let num: u64 = stem
      .split('_')
      .next()
      .and_then(|s| s.parse().ok())
      .unwrap_or(u64::MAX);
    (num, stem)
  });
  if let Some(l) = limit {
    files.truncate(l);
  }
  files
    .into_iter()
    .map(|path| {
      let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
      let id = stem.split('_').next().unwrap_or(&stem).to_string();
      let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
      let problem: Problem =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
      Ok(LoadedInstance { id, path, problem })
    })
    .collect()
}

/// Cell name from a directory path (last component).
pub fn cell_name(dir: &Path) -> String {
  dir
    .file_name()
    .and_then(|s| s.to_str())
    .unwrap_or("cell")
    .to_string()
}

/// Cap rayon's global pool at `max` threads unless RAYON_NUM_THREADS is set lower.
pub fn init_threads(requested: usize, max: usize) -> usize {
  let env_cap = std::env::var("RAYON_NUM_THREADS")
    .ok()
    .and_then(|s| s.parse::<usize>().ok());
  let mut n = requested.max(1).min(max);
  if let Some(c) = env_cap {
    n = n.min(c.max(1));
  }
  let _ = rayon::ThreadPoolBuilder::new()
    .num_threads(n)
    .build_global();
  n
}
