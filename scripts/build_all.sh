#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash scripts/build_gemm_f32_avx512.sh
bash scripts/build_native_gelu_avx512.sh
bash scripts/build_native_activations_avx512.sh
bash scripts/build_native_norm_rope_avx512.sh
bash scripts/build_native_block_elementwise_avx512.sh
bash scripts/build_native_repo_avx512.sh
bash scripts/build_native_embeddings_avx512.sh
bash scripts/build_native_sampler_avx512.sh
bash scripts/build_native_sdpa_avx512_exact_q8.sh

printf 'Built strict AVX-512 backends in %s\n' "${ROOT}/artifacts/backends"
