#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="${SRC:-scripts/native_sdpa_avx512_exact_q8t512.c}"
OUT="${OUT:-artifacts/backends/libtriposplat_sdpa_avx512_exact_q8t512.so}"
CC="${CC:-gcc}"
KEY_TILE="${KEY_TILE:-512}"
QUERY_BLOCK="${QUERY_BLOCK:-8}"
read -r -a CFLAGS_ARRAY <<< "${CFLAGS:--O3 -fPIC}"
read -r -a ARCH_FLAGS_ARRAY <<< "${ARCH_FLAGS:--mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -march=native}"
read -r -a OPENMP_FLAGS_ARRAY <<< "${OPENMP_FLAGS:--fopenmp}"
read -r -a LDFLAGS_ARRAY <<< "${LDFLAGS:-}"

[[ "$KEY_TILE" == "64" || "$KEY_TILE" == "128" || "$KEY_TILE" == "256" || "$KEY_TILE" == "512" ]] || { echo "KEY_TILE must be 64, 128, 256, or 512" >&2; exit 2; }
[[ "$QUERY_BLOCK" == "4" || "$QUERY_BLOCK" == "8" ]] || { echo "QUERY_BLOCK must be 4 or 8" >&2; exit 2; }

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUT}")"
"${CC}" "${CFLAGS_ARRAY[@]}" "-DKEY_TILE=${KEY_TILE}" "-DQUERY_BLOCK=${QUERY_BLOCK}" "${ARCH_FLAGS_ARRAY[@]}" "${OPENMP_FLAGS_ARRAY[@]}" -shared \
  -o "${OUT}" "${SRC}" "${LDFLAGS_ARRAY[@]}" -lm
ls -lh "${OUT}"
