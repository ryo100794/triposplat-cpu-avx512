#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${CC:-gcc}"
OUT="${OUT:-$ROOT/artifacts/backends/libtriposplat_nf24_mlp_avx512.so}"
read -r -a CFLAGS_ARRAY <<< "${CFLAGS:--O3 -fPIC}"
read -r -a ARCH_FLAGS_ARRAY <<< "${ARCH_FLAGS:--mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma}"
read -r -a OPENMP_FLAGS_ARRAY <<< "${OPENMP_FLAGS:--fopenmp}"
read -r -a LDFLAGS_ARRAY <<< "${LDFLAGS:-}"

mkdir -p "$(dirname "$OUT")"
"$CC" "${CFLAGS_ARRAY[@]}" "${ARCH_FLAGS_ARRAY[@]}" "${OPENMP_FLAGS_ARRAY[@]}" -shared \
  "$ROOT/scripts/native_nf24_mlp_avx512.c" "${LDFLAGS_ARRAY[@]}" -lm -o "$OUT"
ls -lh "$OUT"
