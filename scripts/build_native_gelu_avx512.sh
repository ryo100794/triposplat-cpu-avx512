#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/artifacts/backends"
gcc -O3 -fPIC -shared -fopenmp -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma \
  "$ROOT/scripts/native_gelu_avx512.c" \
  -o "$ROOT/artifacts/backends/libtriposplat_gelu_avx512.so"
