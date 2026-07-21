#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT
export FLOW_RUNNER="scripts/run_rnf8_strict.sh"
export RUN_ID="${RUN_ID:-cpu_end_to_end_nf24_i16_q8t512_s20_g3_1024}"
export VIEWER_TITLE="${VIEWER_TITLE:-TripoSplat CPU NF24 int16 q8t512 s20}"
export STEPS=20
export RNF8_STAGES=3
export RNF8_RESIDUAL_MODE=nf24_i16
export RNF8_LIBRARY="${RNF8_LIBRARY:-${PROJECT_ROOT}/artifacts/backends/libtriposplat_gemm_nf24_i16_avx512.so}"
export SDPA_LIBRARY="${SDPA_LIBRARY:-${PROJECT_ROOT}/artifacts/backends/libtriposplat_sdpa_avx512_exact_q8t512.so}"
export SDPA_SYMBOL="${SDPA_SYMBOL:-triposplat_sdpa_f32_avx512_exact_q8t512}"

exec bash "${PROJECT_ROOT}/scripts/run_cpu_end_to_end_strict.sh"
