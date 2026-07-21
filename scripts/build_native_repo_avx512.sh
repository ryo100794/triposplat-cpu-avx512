#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CC="${CC:-gcc}"
mkdir -p "${PROJECT_ROOT}/artifacts/backends"
"${CC}" -O3 -fPIC -shared -fopenmp -ffp-contract=off \
  -mavx512f -mavx512dq -mavx512vl -mfma \
  "${PROJECT_ROOT}/scripts/native_repo_avx512.c" \
  -o "${PROJECT_ROOT}/artifacts/backends/libtriposplat_repo_avx512.so"
printf '%s\n' "${PROJECT_ROOT}/artifacts/backends/libtriposplat_repo_avx512.so"
