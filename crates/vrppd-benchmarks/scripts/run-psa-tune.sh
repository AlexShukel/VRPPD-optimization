#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../.."

cargo run --release -p vrppd-benchmarks --bin tune_psa -- \
  --cells 10_20,20_50 \
  --reps 3 \
  --objective distance \
  "$@"
