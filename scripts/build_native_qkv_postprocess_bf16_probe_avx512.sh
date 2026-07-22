#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${CC:-gcc}"
MODE="${BF16_TARGET:-v}"
OUT="${OUT:-$ROOT/artifacts/backends/libtriposplat_qkv_postprocess_bf16_${MODE}_probe_avx512.so}"
read -r -a CFLAGS_ARRAY <<< "${CFLAGS:--O3 -fPIC}"
read -r -a ARCH_FLAGS_ARRAY <<< "${ARCH_FLAGS:--mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma}"
read -r -a OPENMP_FLAGS_ARRAY <<< "${OPENMP_FLAGS:--fopenmp}"
read -r -a LDFLAGS_ARRAY <<< "${LDFLAGS:-}"

case "$MODE" in
  k) DEFINES=(-DTRIPOSPLAT_BF16_ROUND_K) ;;
  v) DEFINES=(-DTRIPOSPLAT_BF16_ROUND_V) ;;
  kv) DEFINES=(-DTRIPOSPLAT_BF16_ROUND_K -DTRIPOSPLAT_BF16_ROUND_V) ;;
  *) echo "unsupported BF16_TARGET: $MODE" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$OUT")"
"$CC" "${CFLAGS_ARRAY[@]}" "${ARCH_FLAGS_ARRAY[@]}" "${OPENMP_FLAGS_ARRAY[@]}" \
  "${DEFINES[@]}" -shared "$ROOT/scripts/native_qkv_postprocess_bf16_probe_avx512.c" \
  "${LDFLAGS_ARRAY[@]}" -lm -o "$OUT"
ls -lh "$OUT"
