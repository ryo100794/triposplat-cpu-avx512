#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="${VENV_PY:-${PROJECT_ROOT}/.venv/bin/python}"
TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO:-${PROJECT_ROOT}/vendor/TripoSplat}"
TRIPOSPLAT_CKPTS="${TRIPOSPLAT_CKPTS:-${PROJECT_ROOT}/models/TripoSplat/ckpts}"
INPUT="${INPUT:?Set INPUT to the raw source image}"
REFERENCE_NPZ="${REFERENCE_NPZ:?Set REFERENCE_NPZ to the float32 s20 baseline NPZ}"
RUN_ID="${RUN_ID:-cpu_end_to_end_avx512_q8_s20_g3_1024}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/artifacts/end_to_end/${RUN_ID}}"
MODEL_THREADS="${MODEL_THREADS:-8}"
SDPA_THREADS="${SDPA_THREADS:-4}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-262144}"
RESUME="${RESUME:-0}"
CAPACITY_CHECK_BYTES="${CAPACITY_CHECK_BYTES:-268435456}"

export GS_PROJECT_ROOT="${PROJECT_ROOT}"
export TRIPOSPLAT_REPO
export TRIPOSPLAT_CKPTS
export OMP_NUM_THREADS="${MODEL_THREADS}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "${PROJECT_ROOT}"

PREPARED_DIR="${RUN_DIR}/prepared"
CONDITION_DIR="${RUN_DIR}/condition"
FLOW_DIR="${RUN_DIR}/flow"
GAUSSIAN_DIR="${RUN_DIR}/gaussian"
RENDER_DIR="${RUN_DIR}/render"
NOISE_NPZ="${RUN_DIR}/noise_seed0.npz"
STARTED_EPOCH="$(date +%s)"
mkdir -p "${PREPARED_DIR}" "${CONDITION_DIR}" "${FLOW_DIR}" "${GAUSSIAN_DIR}" "${RENDER_DIR}"
"${VENV_PY}" scripts/check_output_capacity.py --directory "${RUN_DIR}" --bytes "${CAPACITY_CHECK_BYTES}"

run_if_missing() {
  local sentinel="$1"
  shift
  if [[ "${RESUME}" == "1" && -s "${sentinel}" ]]; then
    printf 'resume: %s\n' "${sentinel}"
    return 0
  fi
  "$@"
}

run_if_missing "${PREPARED_DIR}/prepared_rgb.webp" \
  "${VENV_PY}" scripts/triposplat_cpu_prepare_image.py \
    --input "${INPUT}" --output-dir "${PREPARED_DIR}" --canvas-size 1024 --erode-radius 1

run_if_missing "${CONDITION_DIR}/condition.npz" \
  "${VENV_PY}" scripts/encode_triposplat_condition_cpu.py \
    --input "${PREPARED_DIR}/prepared_rgb.webp" \
    --output "${CONDITION_DIR}/condition.npz" \
    --canvas-size 1024 --model-dtype bfloat16 --vae-deterministic

run_if_missing "${NOISE_NPZ}" \
  "${VENV_PY}" scripts/generate_triposplat_flow_noise.py --output "${NOISE_NPZ}" --seed 0

if [[ "${RESUME}" != "1" || ! -s "${FLOW_DIR}/base_latent.npz" ]]; then
  INPUT="${PREPARED_DIR}/prepared_rgb.webp" \
  CONDITION_NPZ="${CONDITION_DIR}/condition.npz" \
  NOISE_NPZ="${NOISE_NPZ}" \
  REFERENCE_NPZ="${REFERENCE_NPZ}" \
  OUTPUT_DIR="${FLOW_DIR}" \
  COMPARE_JSON="${FLOW_DIR}/compare_vs_float32_s20.json" \
  RUN_ID="${RUN_ID}_flow" \
  MODEL_THREADS="${MODEL_THREADS}" \
  SDPA_THREADS="${SDPA_THREADS}" \
  VENV_PY="${VENV_PY}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  TRIPOSPLAT_REPO="${TRIPOSPLAT_REPO}" \
  TRIPOSPLAT_CKPTS="${TRIPOSPLAT_CKPTS}" \
    bash scripts/run_s20_strict.sh
fi

run_if_missing "${GAUSSIAN_DIR}/output.ply" \
  "${VENV_PY}" scripts/run_triposplat_encoded_external_noise.py \
    --input "${PREPARED_DIR}/prepared_rgb.webp" \
    --output-dir "${GAUSSIAN_DIR}" \
    --latent-npz "${FLOW_DIR}/base_latent.npz" \
    --num-gaussians "${NUM_GAUSSIANS}" \
    --canvas-size 1024 --seed 0 --device cpu \
    --model-dtype float32 --decoder-dtype float32 \
    --decoder-random-mode numpy --decoder-random-seed 0 \
    --save-decoder-random-npz "${RUN_DIR}/decoder_random_seed0.npz" \
    --lowmem-export --export-chunk-size 32768

run_if_missing "${RENDER_DIR}/manifest.json" \
  "${VENV_PY}" scripts/render_triposplat_ply_official_view.py \
    --ply "${GAUSSIAN_DIR}/output.ply" \
    --source "${PREPARED_DIR}/prepared_rgb.webp" \
    --output-dir "${RENDER_DIR}" --width 1024 --height 1024

run_if_missing "${GAUSSIAN_DIR}/viewer.html" \
  "${VENV_PY}" scripts/make_triposplat_ply_viewer_html_webgl.py \
    --ply "${GAUSSIAN_DIR}/output.ply" \
    --output "${GAUSSIAN_DIR}/viewer.html" \
    --title "TripoSplat CPU AVX-512 strict s20" \
    --default-view front_x --fov-deg 45 --distance-scale 2.4 \
    --thumbnail-image "${RENDER_DIR}/official_spark_default.png"

"${VENV_PY}" scripts/summarize_triposplat_end_to_end.py \
  --run-dir "${RUN_DIR}" --raw-input "${INPUT}" \
  --started-epoch "${STARTED_EPOCH}" --output "${RUN_DIR}/manifest.json"

printf 'run_dir=%s\n' "${RUN_DIR}"
