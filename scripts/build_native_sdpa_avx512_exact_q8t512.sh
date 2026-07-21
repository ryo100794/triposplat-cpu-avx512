#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="${SRC:-scripts/native_sdpa_avx512_exact_q8t512.c}"
OUT="${OUT:-artifacts/backends/libtriposplat_sdpa_avx512_exact_q8t512.so}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUT}")"
gcc -O3 -fPIC -shared -fopenmp -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -march=native \
  -o "${OUT}" "${SRC}" -lm
ls -lh "${OUT}"
