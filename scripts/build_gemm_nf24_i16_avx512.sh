#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${CC:-gcc}"
OUT="${OUT:-$ROOT/artifacts/backends/libtriposplat_gemm_nf24_i16_avx512.so}"
ROW_TILE="${ROW_TILE:-8}"
read -r -a CFLAGS_ARRAY <<< "${CFLAGS:--O3 -fPIC}"
read -r -a ARCH_FLAGS_ARRAY <<< "${ARCH_FLAGS:--mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma}"
read -r -a OPENMP_FLAGS_ARRAY <<< "${OPENMP_FLAGS:--fopenmp}"
read -r -a LDFLAGS_ARRAY <<< "${LDFLAGS:-}"

mkdir -p "$(dirname "$OUT")"
[[ "$ROW_TILE" == "8" || "$ROW_TILE" == "16" || "$ROW_TILE" == "24" ]] || { echo "ROW_TILE must be 8, 16, or 24" >&2; exit 2; }
"$CC" "${CFLAGS_ARRAY[@]}" -DROW_TILE="$ROW_TILE" "${ARCH_FLAGS_ARRAY[@]}" "${OPENMP_FLAGS_ARRAY[@]}" -shared \
  "$ROOT/scripts/gemm_nf24_i16_avx512.c" \
  "$ROOT/scripts/gemm_nf24_i16_avx512_tail.c" \
  "${LDFLAGS_ARRAY[@]}" -o "$OUT"
ls -lh "$OUT"
