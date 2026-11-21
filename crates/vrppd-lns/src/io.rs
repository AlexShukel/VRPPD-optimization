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

/// Content hash of a file as the binaries record it (`weights_hash`): the
/// first 16 hex digits of its SHA-256.
pub fn file_hash(path: &Path) -> Result<String, String> {
  let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
  Ok(
    sha256(&data)[..8]
      .iter()
      .map(|b| format!("{b:02x}"))
      .collect(),
  )
}

/// SHA-256 digest (FIPS 180-4) of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
  const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  let mut h: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  // Padding: a 1 bit, zeros up to 56 bytes mod 64, then the bit length.
  let mut msg = data.to_vec();
  msg.push(0x80);
  while msg.len() % 64 != 56 {
    msg.push(0);
  }
  msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_be_bytes());
  for block in msg.chunks_exact(64) {
    let mut w = [0u32; 64];
    for (t, word) in block.chunks_exact(4).enumerate() {
      w[t] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
    }
    for t in 16..64 {
      let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
      let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16]
        .wrapping_add(s0)
        .wrapping_add(w[t - 7])
        .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
    for (k, wt) in K.iter().zip(w) {
      let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
      let ch = (e & f) ^ (!e & g);
      let t1 = hh
        .wrapping_add(s1)
        .wrapping_add(ch)
        .wrapping_add(*k)
        .wrapping_add(wt);
      let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
      let maj = (a & b) ^ (a & c) ^ (b & c);
      hh = g;
      g = f;
      f = e;
      e = d.wrapping_add(t1);
      d = c;
      c = b;
      b = a;
      a = t1.wrapping_add(s0.wrapping_add(maj));
    }
    for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
      *x = x.wrapping_add(y);
    }
  }
  let mut out = [0u8; 32];
  for (chunk, x) in out.chunks_exact_mut(4).zip(h) {
    chunk.copy_from_slice(&x.to_be_bytes());
  }
  out
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

#[cfg(test)]
mod tests {
  use super::*;

  fn hex(d: [u8; 32]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
  }

  #[test]
  fn sha256_matches_known_digests() {
    let cases: [(&[u8], &str); 4] = [
      (
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      ),
      (
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      ),
      // 56 bytes: the length no longer fits the first block
      (
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
      ),
      (
        &[b'a'; 1000],
        "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
      ),
    ];
    for (data, want) in cases {
      assert_eq!(hex(sha256(data)), want, "{} bytes", data.len());
    }
  }

  #[test]
  fn file_hash_is_first_16_hex_digits() {
    // `shasum -a 256` of the fixture: 9c595e53d61c1f6d1b1c3633f345ab88...
    let path =
      Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlp_fixture_weights.json");
    assert_eq!(file_hash(&path).unwrap(), "9c595e53d61c1f6d");
    assert!(file_hash(Path::new("/no/such/file")).is_err());
  }
}
