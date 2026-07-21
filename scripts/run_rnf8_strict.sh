#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="${VENV_PY:-${PROJECT_ROOT}/.venv/bin/python}"
TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO:-${PROJECT_ROOT}/vendor/TripoSplat}"
TRIPOSPLAT_CKPTS="${TRIPOSPLAT_CKPTS:-${PROJECT_ROOT}/models/TripoSplat/ckpts}"
MODEL_THREADS="${MODEL_THREADS:-8}"
SDPA_THREADS="${SDPA_THREADS:-4}"
STEPS="${STEPS:-1}"
RNF8_STAGES="${RNF8_STAGES:-2}"
RUN_ID="${RUN_ID:-cpu_avx512_rnf8x${RNF8_STAGES}_strict_s${STEPS}_g3_1024}"
INPUT="${INPUT:-${PROJECT_ROOT}/inputs/prepared_rgb.webp}"
CONDITION_NPZ="${CONDITION_NPZ:-${PROJECT_ROOT}/inputs/condition_1024.npz}"
NOISE_NPZ="${NOISE_NPZ:-${PROJECT_ROOT}/inputs/noise_1024_seed0.npz}"
REFERENCE_NPZ="${REFERENCE_NPZ:-${PROJECT_ROOT}/inputs/reference_float32_s${STEPS}.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/quant_flow/${RUN_ID}}"
COMPARE_JSON="${COMPARE_JSON:-${PROJECT_ROOT}/artifacts/audits/${RUN_ID}_vs_float32_s${STEPS}.json}"
CAPACITY_CHECK_BYTES="${CAPACITY_CHECK_BYTES:-67108864}"

for path in "${VENV_PY}" "${TRIPOSPLAT_REPO}/triposplat.py" "${INPUT}" "${CONDITION_NPZ}" "${NOISE_NPZ}" "${REFERENCE_NPZ}"; do
  [[ -e "${path}" ]] || { printf 'Required path is missing: %s\n' "${path}" >&2; exit 2; }
done

required_libraries=(
  libtriposplat_gemm_rnf8_avx512.so
  libtriposplat_gelu_avx512.so
  libtriposplat_activations_avx512.so
  libtriposplat_norm_rope_avx512.so
  libtriposplat_block_elementwise_avx512.so
  libtriposplat_repo_avx512.so
  libtriposplat_embeddings_avx512.so
  libtriposplat_sampler_avx512.so
  libtriposplat_sdpa_avx512_exact_q8.so
)
for library in "${required_libraries[@]}"; do
  [[ -f "${PROJECT_ROOT}/artifacts/backends/${library}" ]] || {
    printf 'Native library is missing: %s\nRun scripts/build_all.sh first.\n' "${library}" >&2
    exit 2
  }
done

mkdir -p "${OUTPUT_DIR}" "$(dirname "${COMPARE_JSON}")"
cd "${PROJECT_ROOT}"
"${VENV_PY}" scripts/check_output_capacity.py --directory "${OUTPUT_DIR}" --bytes "${CAPACITY_CHECK_BYTES}"

env \
  GS_PROJECT_ROOT="${PROJECT_ROOT}" \
  TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO}" \
  TRIPOSPLAT_CKPTS="${TRIPOSPLAT_CKPTS}" \
  OMP_NUM_THREADS="${MODEL_THREADS}" \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  TORCH_NUM_THREADS="${MODEL_THREADS}" \
  TORCH_NUM_INTEROP_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 \
  TRIPOSPLAT_RNF8_STAGES="${RNF8_STAGES}" \
  TRIPOSPLAT_NATIVE_SDPA_THREADS="${SDPA_THREADS}" \
  TRIPOSPLAT_NATIVE_SDPA_LIBRARY=artifacts/backends/libtriposplat_sdpa_avx512_exact_q8.so \
  TRIPOSPLAT_NATIVE_SDPA_SYMBOL=triposplat_sdpa_f32_avx512_exact_q8 \
  "${VENV_PY}" scripts/run_triposplat_rnf8_param_batch.py \
    --output-dir "${OUTPUT_DIR}" \
    --input "${INPUT}" \
    --condition-npz "${CONDITION_NPZ}" \
    --noise-npz "${NOISE_NPZ}" \
    --canvas-size 1024 \
    --model-dtype float32 \
    --device cpu \
    --variants "base:${STEPS}:3.0:3.0" \
    --seed 0 \
    --static-condition-cache \
    --position-embed-cache \
    --cfg-deduplicate-state-forward \
    --cfg-deduplicate-state-assume-duplicated \
    --negative-condition-compression \
    --negative-condition-selective-final-block \
    --negative-condition-selective-final-positive-only \
    --negative-condition-inplace-elementwise \
    --negative-condition-internal-timing \
    --attention-backend native_avx512_exact \
    --attention-contiguous-qkv \
    --selective-final-block \
    --native-avx512-linear \
    --native-avx512-linear-include-regex '.*' \
    --native-avx512-linear-library artifacts/backends/libtriposplat_gemm_rnf8_avx512.so \
    --native-avx512-linear-threads "${MODEL_THREADS}" \
    --native-avx512-linear-strict \
    --native-avx512-gelu \
    --native-avx512-gelu-include-regex '.*' \
    --native-avx512-gelu-library artifacts/backends/libtriposplat_gelu_avx512.so \
    --native-avx512-gelu-threads "${MODEL_THREADS}" \
    --native-avx512-gelu-strict \
    --native-avx512-silu \
    --native-avx512-silu-library artifacts/backends/libtriposplat_activations_avx512.so \
    --native-avx512-silu-threads "${MODEL_THREADS}" \
    --native-avx512-silu-strict \
    --native-avx512-norm-rope \
    --native-avx512-norm-rope-library artifacts/backends/libtriposplat_norm_rope_avx512.so \
    --native-avx512-norm-rope-threads "${MODEL_THREADS}" \
    --native-avx512-norm-rope-strict \
    --native-avx512-block-elementwise \
    --native-avx512-block-elementwise-library artifacts/backends/libtriposplat_block_elementwise_avx512.so \
    --native-avx512-block-elementwise-threads "${MODEL_THREADS}" \
    --native-avx512-block-elementwise-strict \
    --native-avx512-repo \
    --native-avx512-repo-library artifacts/backends/libtriposplat_repo_avx512.so \
    --native-avx512-repo-threads "${MODEL_THREADS}" \
    --native-avx512-repo-strict \
    --native-avx512-embeddings \
    --native-avx512-embeddings-library artifacts/backends/libtriposplat_embeddings_avx512.so \
    --native-avx512-embeddings-threads "${MODEL_THREADS}" \
    --native-avx512-embeddings-strict \
    --native-avx512-sampler \
    --native-avx512-sampler-library artifacts/backends/libtriposplat_sampler_avx512.so \
    --native-avx512-sampler-threads "${MODEL_THREADS}" \
    --native-avx512-sampler-strict \
    --no-progress

"${VENV_PY}" scripts/compare_latent_npz.py \
  --reference "${REFERENCE_NPZ}" \
  --candidate "${OUTPUT_DIR}/base_latent.npz" \
  --output "${COMPARE_JSON}"

printf 'output_dir=%s\ncompare_json=%s\n' "${OUTPUT_DIR}" "${COMPARE_JSON}"
