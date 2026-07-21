#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CC="${CC:-gcc}"

cd "${PROJECT_ROOT}"
mkdir -p artifacts/backends
"${CC}" -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl \
  -fopenmp -fPIC -shared -ffp-contract=off \
  scripts/native_sampler_avx512.c -o artifacts/backends/libtriposplat_sampler_avx512.so
objdump -d artifacts/backends/libtriposplat_sampler_avx512.so > artifacts/backends/libtriposplat_sampler_avx512.objdump.txt
printf '%s\n' artifacts/backends/libtriposplat_sampler_avx512.so
