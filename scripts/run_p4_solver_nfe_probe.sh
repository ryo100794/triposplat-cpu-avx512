#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="${VENV_PY:-${PROJECT_ROOT}/.venv/bin/python}"
TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO:-${PROJECT_ROOT}/vendor/TripoSplat}"
TRIPOSPLAT_CKPTS="${TRIPOSPLAT_CKPTS:-${PROJECT_ROOT}/models/TripoSplat/ckpts}"
INPUT="${INPUT:-${PROJECT_ROOT}/artifacts/prepared/cpu_rmbg_1024/prepared_rgb.webp}"
CONDITION_NPZ="${CONDITION_NPZ:-${PROJECT_ROOT}/artifacts/noise/condition_1024_bf16_cpu_rmbg.npz}"
NOISE_NPZ="${NOISE_NPZ:-${PROJECT_ROOT}/artifacts/noise/cpu_external_noise_1024_seed0_flow.npz}"
REFERENCE_NPZ="${REFERENCE_NPZ:?Set REFERENCE_NPZ to the exact Euler reference latent}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/quant_flow/p4_solver_nfe_probe}"
AUDIT_ROOT="${AUDIT_ROOT:-${PROJECT_ROOT}/artifacts/audits/p4_solver_nfe_probe}"
SOLVERS="${SOLVERS:-euler,ab2}"
STEPS="${STEPS:-3}"
REFERENCE_STEPS="${REFERENCE_STEPS:-4}"
MODEL_THREADS="${MODEL_THREADS:-4}"
GCC_RUNTIME_DIR="${GCC_RUNTIME_DIR:-}"

QKV_POSTPROCESS_LIBRARY="${QKV_POSTPROCESS_LIBRARY:-${PROJECT_ROOT}/artifacts/backends/libtriposplat_qkv_postprocess_v2_gcc13_znver4.so}"
PACKED_SDPA_LIBRARY="${PACKED_SDPA_LIBRARY:-${PROJECT_ROOT}/artifacts/backends/libtriposplat_sdpa_q8t512_packed_gcc13_znver4.so}"
NATIVE_SDPA_LIBRARY="${NATIVE_SDPA_LIBRARY:-${PROJECT_ROOT}/artifacts/backends/libtriposplat_sdpa_q8t512_gcc13_znver4_workspace.so}"
RNF8_LIBRARY="${RNF8_LIBRARY:-${PROJECT_ROOT}/artifacts/backends/libtriposplat_gemm_nf24_gcc13_znver4_mt8.so}"

for path in "${VENV_PY}" "${TRIPOSPLAT_REPO}/triposplat.py" "${INPUT}" "${CONDITION_NPZ}" "${NOISE_NPZ}" "${REFERENCE_NPZ}" "${QKV_POSTPROCESS_LIBRARY}" "${PACKED_SDPA_LIBRARY}" "${NATIVE_SDPA_LIBRARY}" "${RNF8_LIBRARY}"; do
  [[ -e "${path}" ]] || { printf 'Required path is missing: %s\n' "${path}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}" "${AUDIT_ROOT}"
cd "${PROJECT_ROOT}"
IFS=',' read -r -a requested_solvers <<< "${SOLVERS}"
for solver in "${requested_solvers[@]}"; do
  case "${solver}" in euler|ab2) ;; *) printf 'Unknown solver: %s\n' "${solver}" >&2; exit 2 ;; esac
  run_id="${solver}_s${STEPS}_vs_euler_s${REFERENCE_STEPS}"
  output_dir="${OUTPUT_ROOT}/${run_id}"
  compare_json="${AUDIT_ROOT}/${run_id}.json"
  mkdir -p "${output_dir}"

  env \
    GS_PROJECT_ROOT="${PROJECT_ROOT}" \
    TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO}" \
    TRIPOSPLAT_CKPTS="${TRIPOSPLAT_CKPTS}" \
    TRIPOSPLAT_RNF8_STAGES=3 \
    TRIPOSPLAT_RNF8_RESIDUAL_MODE=nf24_i16 \
    TRIPOSPLAT_QKV_POSTPROCESS_LIBRARY="${QKV_POSTPROCESS_LIBRARY}" \
    TRIPOSPLAT_PACKED_SDPA_LIBRARY="${PACKED_SDPA_LIBRARY}" \
    TRIPOSPLAT_NATIVE_SDPA_LIBRARY="${NATIVE_SDPA_LIBRARY}" \
    TRIPOSPLAT_NATIVE_SDPA_SYMBOL=triposplat_sdpa_f32_avx512_exact_q8t512 \
    TRIPOSPLAT_NATIVE_SDPA_THREADS="${MODEL_THREADS}" \
    OMP_NUM_THREADS="${MODEL_THREADS}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    TORCH_NUM_THREADS="${MODEL_THREADS}" TORCH_NUM_INTEROP_THREADS=1 \
    LD_LIBRARY_PATH="${GCC_RUNTIME_DIR}${GCC_RUNTIME_DIR:+:}${LD_LIBRARY_PATH:-}" \
    "${VENV_PY}" scripts/run_triposplat_rnf8_packed_v3_param_batch.py \
      --input "${INPUT}" --output-dir "${output_dir}" \
      --steps "${STEPS}" --guidance-scale 3 --shift 3 --sampler-solver "${solver}" \
      --canvas-size 1024 --seed 0 --device cpu --model-dtype float32 \
      --noise-npz "${NOISE_NPZ}" --condition-npz "${CONDITION_NPZ}" \
      --static-condition-cache --position-embed-cache \
      --cfg-deduplicate-state-forward --cfg-deduplicate-state-assume-duplicated \
      --negative-condition-compression --negative-condition-internal-timing \
      --negative-condition-selective-final-block --negative-condition-selective-final-positive-only \
      --negative-condition-inplace-elementwise \
      --attention-backend native_avx512_exact --attention-compute-dtype model --attention-contiguous-qkv \
      --selective-final-block --selective-final-block-backend native_avx512_exact --selective-final-block-compute-dtype model \
      --native-avx512-linear --native-avx512-linear-library "${RNF8_LIBRARY}" --native-avx512-linear-threads "${MODEL_THREADS}" \
      --native-avx512-gelu --native-avx512-gelu-library artifacts/backends/libtriposplat_gelu_avx512.so --native-avx512-gelu-threads "${MODEL_THREADS}" \
      --native-avx512-norm-rope --native-avx512-norm-rope-library artifacts/backends/libtriposplat_norm_rope_avx512.so --native-avx512-norm-rope-threads "${MODEL_THREADS}" \
      --native-avx512-silu --native-avx512-silu-library artifacts/backends/libtriposplat_activations_avx512.so --native-avx512-silu-threads "${MODEL_THREADS}" \
      --native-avx512-block-elementwise --native-avx512-block-elementwise-library artifacts/backends/libtriposplat_block_elementwise_avx512.so --native-avx512-block-elementwise-threads "${MODEL_THREADS}" \
      --native-avx512-repo --native-avx512-repo-library artifacts/backends/libtriposplat_repo_avx512.so --native-avx512-repo-threads "${MODEL_THREADS}" \
      --native-avx512-embeddings --native-avx512-embeddings-library artifacts/backends/libtriposplat_embeddings_avx512.so --native-avx512-embeddings-threads "${MODEL_THREADS}" \
      --native-avx512-sampler --native-avx512-sampler-library artifacts/backends/libtriposplat_sampler_avx512.so --native-avx512-sampler-threads "${MODEL_THREADS}" \
      --no-progress >"${output_dir}/runner.log" 2>&1

  "${VENV_PY}" scripts/compare_latent_npz.py \
    --reference "${REFERENCE_NPZ}" --candidate "${output_dir}/base_latent.npz" --output "${compare_json}" \
    >>"${output_dir}/runner.log" 2>&1
  "${VENV_PY}" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps({"solver":sys.argv[2],"steps":int(sys.argv[3]),"combined_rmse":p["combined_rmse_from_key_mse"],"latent_rmse":p["per_key"]["latent"]["rmse"],"camera_rmse":p["per_key"]["camera"]["rmse"]}))' "${compare_json}" "${solver}" "${STEPS}"
done
