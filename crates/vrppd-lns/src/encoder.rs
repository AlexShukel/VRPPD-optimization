//! V1 forward pass in pure Rust (f32), loading weights exported by
//! `python/vrppd_nn/encoder.py` (`export_json`). That module is the reference
//! for the arithmetic; this file follows it layer by layer.
//!
//! Tokens (V vehicles, N orders, T = V + 2N): the start of vehicle v is token v,
//! the pickup of order o is V + o, its delivery V + N + o. Features: type one-hot
//! (start, pickup, delivery), latitude, longitude, demand (1 / loadFactor, 0 for a
//! start), price per km (start only); the last four are standardised with the
//! file's `tok_mean` / `tok_std`.
//!
//! Layers (Linear weights are (out, in), y = x W^T + b; LayerNorm eps 1e-5):
//! `embed`, attention blocks `att0..att{n_att-1}`, the neighbour layer `nb`, the
//! route summary `rs`, blocks `att{n_att}..att{2 n_att - 1}`, and the score head
//! `head` on [h_pickup ; h_delivery]. Nothing before `nb` sees the routes, so
//! that part depends only on the instance ([`Encoder::encode_instance`]).

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use vrppd_core::{Problem, StopKind, WorkingSolution};

use crate::solution::Context;

const F_TOK: usize = 7;
const LN_EPS: f32 = 1e-5;
/// No token (no previous / next stop, or no route).
const NONE: usize = usize::MAX;

#[derive(Deserialize)]
struct Config {
  d: usize,
  heads: usize,
  n_att: usize,
  ff_mult: usize,
}

#[derive(Deserialize)]
struct Param {
  shape: Vec<usize>,
  data: Vec<f32>,
}

#[derive(Deserialize)]
struct WeightsFile {
  model: String,
  config: Config,
  tok_mean: Vec<f64>,
  tok_std: Vec<f64>,
  params: BTreeMap<String, Param>,
}

/// Exported parameters, taken one by one so that missing, misshapen and
/// unexpected names are all errors (like PyTorch's strict `load_state_dict`).
struct Params(BTreeMap<String, Param>);

impl Params {
  fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
    let p = self
      .0
      .remove(name)
      .ok_or_else(|| format!("missing parameter {name}"))?;
    if p.shape != shape {
      return Err(format!(
        "parameter {name} has shape {:?}, expected {shape:?}",
        p.shape
      ));
    }
    let len: usize = shape.iter().product();
    if p.data.len() != len {
      return Err(format!(
        "parameter {name} has {} values, its shape needs {len}",
        p.data.len()
      ));
    }
    Ok(p.data)
  }

  fn finish(self) -> Result<(), String> {
    match self.0.keys().next() {
      Some(name) => Err(format!("unexpected parameter {name}")),
      None => Ok(()),
    }
  }
}

/// c += a b for row-major a (m x k), b (k x n) and c (m x n). The inner loop
/// runs along a row of b and c, so it vectorises.
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
  debug_assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
  for (a_row, c_row) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)).take(m) {
    for (&x, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
      for (cv, &bv) in c_row.iter_mut().zip(b_row) {
        *cv += x * bv;
      }
    }
  }
}

fn add(x: &mut [f32], y: &[f32]) {
  for (a, b) in x.iter_mut().zip(y) {
    *a += b;
  }
}

fn relu(x: &mut [f32]) {
  for v in x.iter_mut() {
    *v = v.max(0.0);
  }
}

fn softmax(x: &mut [f32]) {
  let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
  let mut sum = 0.0;
  for v in x.iter_mut() {
    *v = (*v - max).exp();
    sum += *v;
  }
  let inv = 1.0 / sum;
  for v in x.iter_mut() {
    *v *= inv;
  }
}

struct Linear {
  n_in: usize,
  n_out: usize,
  /// The PyTorch weight transposed: n_in x n_out, row-major.
  wt: Vec<f32>,
  /// Empty for a layer without bias.
  bias: Vec<f32>,
}

impl Linear {
  fn load(
    p: &mut Params,
    name: &str,
    n_in: usize,
    n_out: usize,
    with_bias: bool,
  ) -> Result<Self, String> {
    let w = p.take(&format!("{name}.weight"), &[n_out, n_in])?;
    let mut wt = vec![0.0; n_in * n_out];
    for o in 0..n_out {
      for i in 0..n_in {
        wt[i * n_out + o] = w[o * n_in + i];
      }
    }
    let bias = if with_bias {
      p.take(&format!("{name}.bias"), &[n_out])?
    } else {
      Vec::new()
    };
    Ok(Self {
      n_in,
      n_out,
      wt,
      bias,
    })
  }

  /// y = bias in every one of `rows` rows (zeros without bias).
  fn init(&self, rows: usize, y: &mut Vec<f32>) {
    y.clear();
    if self.bias.is_empty() {
      y.resize(rows * self.n_out, 0.0);
    } else {
      for _ in 0..rows {
        y.extend_from_slice(&self.bias);
      }
    }
  }

  /// y = x W^T + b for x (rows x n_in); y becomes rows x n_out.
  fn forward(&self, x: &[f32], rows: usize, y: &mut Vec<f32>) {
    self.init(rows, y);
    matmul(x, &self.wt, y, rows, self.n_in, self.n_out);
  }

  /// y = [a ; b] W^T + bias for a, b (rows x n_in/2 each), without building the concatenation.
  fn forward_concat(&self, a: &[f32], b: &[f32], rows: usize, y: &mut Vec<f32>) {
    let half = self.n_in / 2;
    self.init(rows, y);
    let (wa, wb) = self.wt.split_at(half * self.n_out);
    matmul(a, wa, y, rows, half, self.n_out);
    matmul(b, wb, y, rows, half, self.n_out);
  }
}

struct LayerNorm {
  weight: Vec<f32>,
  bias: Vec<f32>,
}

impl LayerNorm {
  fn load(p: &mut Params, name: &str, d: usize) -> Result<Self, String> {
    Ok(Self {
      weight: p.take(&format!("{name}.weight"), &[d])?,
      bias: p.take(&format!("{name}.bias"), &[d])?,
    })
  }

  /// Normalise every row of x (rows of width d) in place; population variance.
  fn apply(&self, x: &mut [f32]) {
    let d = self.weight.len();
    for row in x.chunks_exact_mut(d) {
      let mean = row.iter().sum::<f32>() / d as f32;
      let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
      let inv = 1.0 / (var + LN_EPS).sqrt();
      for ((v, w), b) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
        *v = (*v - mean) * inv * w + b;
      }
    }
  }
}

struct AttentionBlock {
  q: Linear,
  k: Linear,
  v: Linear,
  o: Linear,
  ln1: LayerNorm,
  ff1: Linear,
  ff2: Linear,
  ln2: LayerNorm,
}

impl AttentionBlock {
  fn load(p: &mut Params, name: &str, d: usize, ff_mult: usize) -> Result<Self, String> {
    let lin = |p: &mut Params, part: &str, n_in, n_out| {
      Linear::load(p, &format!("{name}.{part}"), n_in, n_out, true)
    };
    Ok(Self {
      q: lin(p, "q", d, d)?,
      k: lin(p, "k", d, d)?,
      v: lin(p, "v", d, d)?,
      o: lin(p, "o", d, d)?,
      ln1: LayerNorm::load(p, &format!("{name}.ln1"), d)?,
      ff1: lin(p, "ff1", d, ff_mult * d)?,
      ff2: lin(p, "ff2", ff_mult * d, d)?,
      ln2: LayerNorm::load(p, &format!("{name}.ln2"), d)?,
    })
  }

  /// h = ln1(h + o(MHA(h))); h = ln2(h + ff2(relu(ff1(h)))) over t tokens, all of them keys.
  fn apply(&self, heads: usize, h: &mut [f32], t: usize, b: &mut Buffers) {
    let d = self.q.n_out;
    let dh = d / heads;
    let scale = 1.0 / (dh as f32).sqrt();
    self.q.forward(h, t, &mut b.q);
    self.k.forward(h, t, &mut b.k);
    self.v.forward(h, t, &mut b.v);
    b.att.clear();
    b.att.resize(t * d, 0.0);
    b.qh.resize(t * dh, 0.0);
    b.kt.resize(dh * t, 0.0);
    b.vh.resize(t * dh, 0.0);
    b.row.resize(t, 0.0);
    for j in 0..heads {
      // Head j's slices of q (pre-scaled), k (transposed: dh x t) and v.
      let off = j * dh;
      for s in 0..t {
        for c in 0..dh {
          b.qh[s * dh + c] = b.q[s * d + off + c] * scale;
          b.kt[c * t + s] = b.k[s * d + off + c];
          b.vh[s * dh + c] = b.v[s * d + off + c];
        }
      }
      for r in 0..t {
        b.row.fill(0.0);
        matmul(&b.qh[r * dh..(r + 1) * dh], &b.kt, &mut b.row, 1, dh, t);
        softmax(&mut b.row);
        let out = &mut b.att[r * d + off..r * d + off + dh];
        matmul(&b.row, &b.vh, out, 1, t, dh);
      }
    }
    self.o.forward(&b.att, t, &mut b.tmp);
    add(h, &b.tmp);
    self.ln1.apply(h);
    self.ff1.forward(h, t, &mut b.hid);
    relu(&mut b.hid);
    self.ff2.forward(&b.hid, t, &mut b.tmp);
    add(h, &b.tmp);
    self.ln2.apply(h);
  }
}

/// Scratch buffers of the forward pass, kept between calls to avoid allocation.
#[derive(Default)]
struct Buffers {
  q: Vec<f32>,
  k: Vec<f32>,
  v: Vec<f32>,
  att: Vec<f32>,
  tmp: Vec<f32>,
  hid: Vec<f32>,
  qh: Vec<f32>,
  kt: Vec<f32>,
  vh: Vec<f32>,
  row: Vec<f32>,
  counts: Vec<f32>,
}

/// Route links of every token: previous and next token on its route and the
/// route (vehicle) it is on; [`NONE`] where there is none.
#[derive(Default)]
struct Links {
  prev: Vec<usize>,
  next: Vec<usize>,
  route: Vec<usize>,
}

impl Links {
  fn build(&mut self, sol: &WorkingSolution, n_vehicles: usize, n_orders: usize) {
    let t = n_vehicles + 2 * n_orders;
    for v in [&mut self.prev, &mut self.next, &mut self.route] {
      v.clear();
      v.resize(t, NONE);
    }
    for (v, r) in sol.routes.iter().enumerate() {
      self.route[v] = v;
      let mut last = v;
      for stop in &r.stops {
        let tok = match stop.kind {
          StopKind::Pickup => n_vehicles + stop.order_idx,
          StopKind::Delivery => n_vehicles + n_orders + stop.order_idx,
        };
        self.route[tok] = v;
        self.prev[tok] = last;
        self.next[last] = tok;
        last = tok;
      }
    }
  }
}

/// Reusable state of [`Encoder::score_encoded`].
#[derive(Default)]
pub(crate) struct Workspace {
  h: Vec<f32>,
  links: Links,
  buf: Buffers,
}

/// The route-independent part of the forward pass for one instance: token
/// states after the embedding and the first `n_att` attention blocks.
pub(crate) struct InstanceEncoding {
  n_vehicles: usize,
  n_orders: usize,
  h: Vec<f32>,
}

/// V1 model: transformer-style encoder over stop tokens with route-aware layers.
pub struct Encoder {
  d: usize,
  heads: usize,
  n_att: usize,
  tok_mean: [f64; 4],
  tok_std: [f64; 4],
  embed: Linear,
  /// 2 n_att blocks; the first n_att run before the route layers.
  att: Vec<AttentionBlock>,
  nb_w1: Linear,
  nb_w2: Linear,
  nb_w3: Linear,
  nb_ff: Linear,
  nb_ln: LayerNorm,
  rs_w4: Linear,
  rs_ln: LayerNorm,
  head_l1: Linear,
  head_l2: Linear,
}

impl Encoder {
  /// Load a V1 weights file; every parameter the config implies must be present
  /// with its shape, and no other.
  pub fn load(path: &Path) -> Result<Encoder, String> {
    let text =
      std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let file: WeightsFile =
      serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    Self::from_file(file).map_err(|e| format!("{}: {e}", path.display()))
  }

  fn from_file(file: WeightsFile) -> Result<Encoder, String> {
    if file.model != "v1" {
      return Err(format!("not a V1 weights file (model '{}')", file.model));
    }
    let Config {
      d,
      heads,
      n_att,
      ff_mult,
    } = file.config;
    if d == 0 || heads == 0 || ff_mult == 0 || d % heads != 0 {
      return Err(format!(
        "bad config d={d} heads={heads} ff_mult={ff_mult} (d must be a positive multiple of heads)"
      ));
    }
    let stats = |name: &str, v: &[f64]| -> Result<[f64; 4], String> {
      <[f64; 4]>::try_from(v).map_err(|_| format!("{name} needs 4 values, has {}", v.len()))
    };
    let tok_mean = stats("tok_mean", &file.tok_mean)?;
    let tok_std = stats("tok_std", &file.tok_std)?;
    if tok_std.iter().any(|s| *s == 0.0 || !s.is_finite()) {
      return Err(format!(
        "tok_std {tok_std:?} has a zero or non-finite entry"
      ));
    }
    let mut p = Params(file.params);
    let encoder = Encoder {
      d,
      heads,
      n_att,
      tok_mean,
      tok_std,
      embed: Linear::load(&mut p, "embed", F_TOK, d, true)?,
      att: (0..2 * n_att)
        .map(|i| AttentionBlock::load(&mut p, &format!("att{i}"), d, ff_mult))
        .collect::<Result<_, _>>()?,
      nb_w1: Linear::load(&mut p, "nb.w1", d, d, false)?,
      nb_w2: Linear::load(&mut p, "nb.w2", d, d, false)?,
      nb_w3: Linear::load(&mut p, "nb.w3", 2 * d, d, true)?,
      nb_ff: Linear::load(&mut p, "nb.ff", d, d, true)?,
      nb_ln: LayerNorm::load(&mut p, "nb.ln", d)?,
      rs_w4: Linear::load(&mut p, "rs.w4", d, d, true)?,
      rs_ln: LayerNorm::load(&mut p, "rs.ln", d)?,
      head_l1: Linear::load(&mut p, "head.l1", 2 * d, d, true)?,
      head_l2: Linear::load(&mut p, "head.l2", d, 1, true)?,
    };
    p.finish()?;
    Ok(encoder)
  }

  /// One score per order (problem order) for the routes of `sol`. Runs the
  /// whole forward pass; [`crate::scorer::EncoderScorer`] caches the
  /// route-independent part per instance.
  pub fn score(&self, ctx: &Context, sol: &WorkingSolution) -> Vec<f64> {
    let mut ws = Workspace::default();
    let encoded = self.encode_instance(ctx.problem, &mut ws);
    self.score_encoded(&encoded, sol, &mut ws)
  }

  /// Standardised token features (T x 7). As in Python, raw features are
  /// rounded to f32 first and standardised in f64.
  fn tokens(&self, problem: &Problem) -> Vec<f32> {
    let (nv, n) = (problem.vehicles.len(), problem.orders.len());
    let mut tok = vec![0.0_f32; (nv + 2 * n) * F_TOK];
    let mut set = |row: usize, kind: usize, values: [f64; 4]| {
      let r = &mut tok[row * F_TOK..(row + 1) * F_TOK];
      r[kind] = 1.0;
      for (c, x) in values.into_iter().enumerate() {
        r[3 + c] = ((x as f32 as f64 - self.tok_mean[c]) / self.tok_std[c]) as f32;
      }
    };
    for (v, veh) in problem.vehicles.iter().enumerate() {
      let loc = &veh.start_location;
      set(v, 0, [loc.latitude, loc.longitude, 0.0, veh.price_km]);
    }
    for (o, order) in problem.orders.iter().enumerate() {
      let demand = 1.0 / order.load_factor;
      let (p, q) = (&order.pickup_location, &order.delivery_location);
      set(nv + o, 1, [p.latitude, p.longitude, demand, 0.0]);
      set(nv + n + o, 2, [q.latitude, q.longitude, demand, 0.0]);
    }
    tok
  }

  /// Embedding and the first `n_att` attention blocks (they do not see the routes).
  pub(crate) fn encode_instance(&self, problem: &Problem, ws: &mut Workspace) -> InstanceEncoding {
    let (nv, n) = (problem.vehicles.len(), problem.orders.len());
    let t = nv + 2 * n;
    let mut h = Vec::new();
    self.embed.forward(&self.tokens(problem), t, &mut h);
    for blk in &self.att[..self.n_att] {
      blk.apply(self.heads, &mut h, t, &mut ws.buf);
    }
    InstanceEncoding {
      n_vehicles: nv,
      n_orders: n,
      h,
    }
  }

  /// The route-dependent rest of the forward pass: neighbour layer, route
  /// summary, the last `n_att` attention blocks and the score head. An order
  /// missing from every route (Python rejects such states) gets no neighbours
  /// and a zero route mean.
  pub(crate) fn score_encoded(
    &self,
    enc: &InstanceEncoding,
    sol: &WorkingSolution,
    ws: &mut Workspace,
  ) -> Vec<f64> {
    let (nv, n) = (enc.n_vehicles, enc.n_orders);
    let t = nv + 2 * n;
    assert_eq!(sol.routes.len(), nv, "one route per vehicle");
    ws.links.build(sol, nv, n);
    ws.h.clear();
    ws.h.extend_from_slice(&enc.h);
    self.neighbour(&mut ws.h, &ws.links, t, &mut ws.buf);
    self.route_summary(&mut ws.h, &ws.links.route, nv, t, &mut ws.buf);
    for blk in &self.att[self.n_att..] {
      blk.apply(self.heads, &mut ws.h, t, &mut ws.buf);
    }
    self.head(&ws.h, nv, n, &mut ws.buf)
  }

  /// h = ln(h + ff(relu(w3([h ; w1 h_prev + w2 h_next])))), a missing neighbour
  /// contributing a zero vector (w1, w2 have no bias).
  fn neighbour(&self, h: &mut [f32], links: &Links, t: usize, b: &mut Buffers) {
    let d = self.d;
    self.nb_w1.forward(h, t, &mut b.q); // w1 h of every token
    self.nb_w2.forward(h, t, &mut b.k); // w2 h of every token
    b.att.clear(); // the neighbour sum of every token
    b.att.resize(t * d, 0.0);
    for s in 0..t {
      let around = &mut b.att[s * d..(s + 1) * d];
      let (p, x) = (links.prev[s], links.next[s]);
      if p != NONE {
        add(around, &b.q[p * d..(p + 1) * d]);
      }
      if x != NONE {
        add(around, &b.k[x * d..(x + 1) * d]);
      }
    }
    self.nb_w3.forward_concat(h, &b.att, t, &mut b.tmp);
    relu(&mut b.tmp);
    self.nb_ff.forward(&b.tmp, t, &mut b.hid);
    add(h, &b.hid);
    self.nb_ln.apply(h);
  }

  /// h = ln(h + w4 m_route(t)), m_r the mean of h over the tokens of route r
  /// (start token included). w4 is applied to the V means, not the T copies.
  fn route_summary(&self, h: &mut [f32], route: &[usize], nv: usize, t: usize, b: &mut Buffers) {
    let d = self.d;
    b.v.clear(); // route means, nv x d
    b.v.resize(nv * d, 0.0);
    b.counts.clear();
    b.counts.resize(nv, 0.0);
    for s in (0..t).filter(|&s| route[s] != NONE) {
      let r = route[s];
      add(&mut b.v[r * d..(r + 1) * d], &h[s * d..(s + 1) * d]);
      b.counts[r] += 1.0;
    }
    for (sums, &count) in b.v.chunks_exact_mut(d).zip(&b.counts) {
      for x in sums.iter_mut() {
        *x /= count;
      }
    }
    self.rs_w4.forward(&b.v, nv, &mut b.tmp);
    for s in 0..t {
      let summary = match route[s] {
        NONE => &self.rs_w4.bias[..],
        r => &b.tmp[r * d..(r + 1) * d],
      };
      add(&mut h[s * d..(s + 1) * d], summary);
    }
    self.rs_ln.apply(h);
  }

  /// score_o = l2(relu(l1([h_pickup(o) ; h_delivery(o)]))).
  fn head(&self, h: &[f32], nv: usize, n: usize, b: &mut Buffers) -> Vec<f64> {
    let d = self.d;
    let pickups = &h[nv * d..(nv + n) * d];
    let deliveries = &h[(nv + n) * d..(nv + 2 * n) * d];
    self
      .head_l1
      .forward_concat(pickups, deliveries, n, &mut b.tmp);
    relu(&mut b.tmp);
    self.head_l2.forward(&b.tmp, n, &mut b.row);
    b.row.iter().map(|&s| s as f64).collect()
  }
}

#[cfg(test)]
mod tests {
  use std::path::PathBuf;

  use vrppd_core::{StopKind, WorkingStop};

  use super::*;

  fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1_fixture.json")
  }

  fn fixture() -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixture_path()).unwrap()).unwrap()
  }

  /// Working solution from the fixture's routes: per vehicle, [order index, "pickup" | "delivery"].
  pub(crate) fn fixture_solution(routes: &serde_json::Value) -> WorkingSolution {
    let routes = routes.as_array().unwrap();
    let mut sol = WorkingSolution::empty(routes.len());
    for (route, stops) in sol.routes.iter_mut().zip(routes) {
      for stop in stops.as_array().unwrap() {
        let kind = match stop[1].as_str().unwrap() {
          "pickup" => StopKind::Pickup,
          "delivery" => StopKind::Delivery,
          other => panic!("bad stop kind {other}"),
        };
        route.stops.push(WorkingStop {
          order_idx: stop[0].as_u64().unwrap() as usize,
          kind,
        });
      }
    }
    sol
  }

  #[test]
  fn v1_matches_pytorch_on_fixture() {
    let encoder = Encoder::load(&fixture_path()).unwrap();
    let fx = fixture();
    let problem: Problem = serde_json::from_value(fx["fixture"]["instance"].clone()).unwrap();
    let ctx = Context::new(&problem);
    let states = fx["fixture"]["states"].as_array().unwrap();
    assert_eq!(states.len(), 3);
    for state in states {
      let expected: Vec<f64> = state["scores"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
      let got = encoder.score(&ctx, &fixture_solution(&state["routes"]));
      assert_eq!(got.len(), expected.len(), "{}", state["name"]);
      let diff = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
      assert!(diff < 1e-4, "{}: max abs difference {diff}", state["name"]);
    }
  }

  #[test]
  fn v1_load_rejects_missing_parameter() {
    let dir = std::env::temp_dir().join(format!("vrppd-encoder-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, value: &serde_json::Value| {
      let path = dir.join(name);
      std::fs::write(&path, value.to_string()).unwrap();
      path
    };
    let mut missing = fixture();
    missing["params"]
      .as_object_mut()
      .unwrap()
      .remove("att0.k.bias")
      .unwrap();
    let err = Encoder::load(&write("missing.json", &missing))
      .err()
      .unwrap();
    assert!(err.contains("missing parameter att0.k.bias"), "{err}");

    let mut misshapen = fixture();
    misshapen["params"]["nb.w3.weight"]["shape"] = serde_json::json!([32, 16]);
    let err = Encoder::load(&write("misshapen.json", &misshapen))
      .err()
      .unwrap();
    assert!(err.contains("nb.w3.weight"), "{err}");

    let mut extra = fixture();
    extra["params"]["att2.q.bias"] = extra["params"]["att0.q.bias"].clone();
    let err = Encoder::load(&write("extra.json", &extra)).err().unwrap();
    assert!(err.contains("unexpected parameter att2.q.bias"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
  }

  #[test]
  fn v1_scores_instance_larger_than_training() {
    use crate::io::{load_instances, resolve_cell_dir};
    use crate::lns::initial_solution;
    let instance = load_instances(&resolve_cell_dir("100_500"), Some(1))
      .unwrap()
      .remove(0);
    assert_eq!(instance.id, "0");
    let ctx = Context::new(&instance.problem);
    let sol = initial_solution(&ctx, 1);
    let scores = Encoder::load(&fixture_path()).unwrap().score(&ctx, &sol);
    assert_eq!(scores.len(), 500);
    assert!(scores.iter().all(|s| s.is_finite()));
    let spread = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
      - scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert!(spread > 0.0, "all 500 scores are equal");
  }
}
