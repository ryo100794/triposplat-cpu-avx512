#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CPUSET="${CPUSET:-26,28,81,83}"
BASELINE_CC="${BASELINE_CC:-gcc}"
GCC13_ROOT="${GCC13_ROOT:-$ROOT/toolchains/gcc13/root}"
CANDIDATE_CC="${CANDIDATE_CC:-$GCC13_ROOT/usr/bin/gcc-13}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-gcc13_znver4}"
CANDIDATE_ARCH_FLAGS="${CANDIDATE_ARCH_FLAGS:--march=znver4 -mtune=znver4}"
CANDIDATE_OPENMP_FLAGS="${CANDIDATE_OPENMP_FLAGS:--fopenmp}"
CANDIDATE_LDFLAGS="${CANDIDATE_LDFLAGS:-}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTDIR="${OUTDIR:-$ROOT/artifacts/audits/p0_cpu_kernels_$RUN_ID}"
THREADS_LIST="${THREADS_LIST:-1 2 3 4}"
WARMUP="${WARMUP:-1}"
REPEAT="${REPEAT:-3}"

BASE_GEMM="$OUTDIR/libtriposplat_gemm_nf24_gcc9.so"
CAND_GEMM="$OUTDIR/libtriposplat_gemm_nf24_${CANDIDATE_LABEL}.so"
BASE_SDPA="$OUTDIR/libtriposplat_sdpa_q8t512_gcc9.so"
CAND_SDPA="$OUTDIR/libtriposplat_sdpa_q8t512_${CANDIDATE_LABEL}.so"

mkdir -p "$OUTDIR"
test -x "$PYTHON"
test -x "$CANDIDATE_CC"

CANDIDATE_ENV=()
if [[ "$CANDIDATE_CC" == "$GCC13_ROOT"/* ]]; then
  CANDIDATE_ENV=(
    env
    "GCC_EXEC_PREFIX=$GCC13_ROOT/usr/lib/gcc/"
    "COMPILER_PATH=$GCC13_ROOT/usr/lib/gcc/x86_64-linux-gnu/13"
    "LIBRARY_PATH=$GCC13_ROOT/usr/lib/gcc/x86_64-linux-gnu/13:$GCC13_ROOT/usr/lib/x86_64-linux-gnu"
  )
fi

(
  cd "$ROOT"
  CC="$BASELINE_CC" OUT="$BASE_GEMM" \
    ARCH_FLAGS='-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma' \
    bash scripts/build_gemm_nf24_i16_avx512.sh
  CC="$BASELINE_CC" OUT="$BASE_SDPA" KEY_TILE=512 \
    ARCH_FLAGS='-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -march=native' \
    bash scripts/build_native_sdpa_avx512_exact_q8t512.sh
  "${CANDIDATE_ENV[@]}" CC="$CANDIDATE_CC" OUT="$CAND_GEMM" \
    ARCH_FLAGS="$CANDIDATE_ARCH_FLAGS" \
    OPENMP_FLAGS="$CANDIDATE_OPENMP_FLAGS" LDFLAGS="$CANDIDATE_LDFLAGS" \
    bash scripts/build_gemm_nf24_i16_avx512.sh
  "${CANDIDATE_ENV[@]}" CC="$CANDIDATE_CC" OUT="$CAND_SDPA" KEY_TILE=512 \
    ARCH_FLAGS="$CANDIDATE_ARCH_FLAGS" \
    OPENMP_FLAGS="$CANDIDATE_OPENMP_FLAGS" LDFLAGS="$CANDIDATE_LDFLAGS" \
    bash scripts/build_native_sdpa_avx512_exact_q8t512.sh
)

{
  printf 'run_id=%s\ncpuset=%s\nthreads_list=%s\nwarmup=%s\nrepeat=%s\ncandidate_label=%s\n' \
    "$RUN_ID" "$CPUSET" "$THREADS_LIST" "$WARMUP" "$REPEAT" "$CANDIDATE_LABEL"
  uname -a
  lscpu
  "$BASELINE_CC" --version
  "$CANDIDATE_CC" --version
  printf 'baseline_gemm_sha256='; sha256sum "$BASE_GEMM" | cut -d' ' -f1
  printf 'candidate_gemm_sha256='; sha256sum "$CAND_GEMM" | cut -d' ' -f1
  printf 'baseline_sdpa_sha256='; sha256sum "$BASE_SDPA" | cut -d' ' -f1
  printf 'candidate_sdpa_sha256='; sha256sum "$CAND_SDPA" | cut -d' ' -f1
  printf 'perf_event_paranoid='; cat /proc/sys/kernel/perf_event_paranoid
  pgrep -af python || true
} > "$OUTDIR/host_and_build.txt"

for threads in $THREADS_LIST; do
  taskset -c "$CPUSET" env \
    OMP_NUM_THREADS="$threads" OMP_PROC_BIND=true OMP_PLACES=cores \
    "$PYTHON" "$ROOT/scripts/bench_native_rnf8_avx512.py" \
      --baseline "$BASE_GEMM" --candidate "$CAND_GEMM" \
      --baseline-code0-dtype int16 --candidate-code0-dtype int16 \
      --shape 12294,1024,4096 \
      --shape 12294,4096,1024 \
      --shape 12294,1024,3072 \
      --shape 12294,1024,1024 \
      --threads "$threads" --warmup "$WARMUP" --repeat "$REPEAT" \
      --output-json "$OUTDIR/gemm_gcc9_vs_${CANDIDATE_LABEL}_threads_${threads}.json"

  for variant in gcc9 "$CANDIDATE_LABEL"; do
    if [[ "$variant" == "gcc9" ]]; then
      library="$BASE_SDPA"
    else
      library="$CAND_SDPA"
    fi
    taskset -c "$CPUSET" env \
      OMP_NUM_THREADS="$threads" OMP_PROC_BIND=true OMP_PLACES=cores \
      "$PYTHON" "$ROOT/scripts/bench_native_sdpa_avx512_exact.py" \
        --library "$library" \
        --symbol triposplat_sdpa_f32_avx512_exact_q8t512 \
        --case self8194,8194,8194,-1,0 \
        --case self12294,12294,12294,-1,0 \
        --heads 16 --threads "$threads" --torch-threads "$threads" \
        --warmup "$WARMUP" --repeat "$REPEAT" --skip-torch-timing \
        --output-json "$OUTDIR/sdpa_${variant}_threads_${threads}.json"
  done
done

printf '%s\n' "$OUTDIR"
