#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../.."

for secs in 60 120 300; do
  echo "==== MILP sensitivity: ${secs}s cap ===="
  cargo run --release -p vrppd-benchmarks --bin bench_large -- \
    --cells 10_20,20_50 \
    --reps 1 \
    --objectives distance,price \
    --algos milp \
    --milp-timeout-secs "${secs}" \
    --max-instances-per-cell 20
done
