#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../.."

cargo run --release -p vrppd-benchmarks --bin bench_large -- \
  --cells 10_10,10_20,20_50,30_100,50_200,100_500 \
  --reps 10 \
  --objectives distance,price,empty \
  --algos milp,psa,cea,routing-pca,routing-pci \
  --milp-timeout-secs 60 \
  --cea-timeout-secs 60 \
  --routing-timeout-secs 60 \
  --max-instances-per-cell 20 \
  "$@"
