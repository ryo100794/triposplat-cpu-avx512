# TripoSplat CPU AVX-512

CPU-only native AVX-512 backends and reproducible evaluation scripts for the
[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
flow model.

> **This repository does not redistribute TripoSplat checkpoints.** The
> validated `q8` baseline name means an eight-query execution block
> (`QUERY_BLOCK=8`), not 8-bit weights or activations. The exact strict s20 baseline remains float32. The selected low-resource NF24
> int16 configuration and its historical NF8/residual-NF8 evaluations are
> reported separately below.

Japanese documentation: [README_ja.md](README_ja.md)

## Validated result

The strict CPU run used the upstream TripoSplat source at commit
`a78fa12d06dbf1381ca548bfac32bb68cb8c451d` and an AMD EPYC 9654.

| Metric | Result |
|---|---:|
| Canvas / sampler | 1024 / 20 steps / guidance 3.0 / shift 3.0 |
| Wall time | 3322.886 s (55 min 22.886 s) |
| Original CPU float32 baseline | 10856.388 s |
| Speedup | 3.27x |
| Native fallback count | 0 across 20 monitored fields |
| Latent RMSE vs. CPU float32 | 2.90157e-5 |
| Camera RMSE vs. CPU float32 | 3.72623e-6 |
| Combined RMSE | 2.06857e-5 |
| NaN / Inf | 0 / 0 |
| Observed Python RSS | approximately 2.37-2.70 GiB |

The detailed evidence and stage timing are in
[`docs/triposplat_native_avx512_q8_s20_validation_20260720_ja.md`](docs/triposplat_native_avx512_q8_s20_validation_20260720_ja.md).

## What is implemented

- Float32 AVX-512 GEMM for all active Flow-model Linear layers.
- Exact dense, key-bias, and final cross SDPA with online softmax.
- Eight-query SDPA blocking with a validated 512-key tile that shares K/V loads across query rows.
- GELU(tanh), SiLU, LayerNorm, multi-head RMSNorm, and RoPE kernels.
- Block modulation, residual, RePo, position/timestep embedding kernels.
- Native CFG and Euler sampler updates.
- CPU background removal, DINO/Flux-VAE condition encoding, deterministic
  noise generation, Gaussian decoding, low-memory export, reference rendering,
  and a standalone WebGL viewer.
- Packed nonlinear NF8/residual-NF8 evaluation and the selected 24-bit NF24 int16 AVX-512 GEMM for all
  206 Flow-model Linear modules, without retaining float32 Linear weights.
- A resumable NF24 checkpoint converter and direct loader that avoids loading the
  official Flow checkpoint at startup.
- Strict wrappers that raise on unsupported execution instead of silently
  falling back to PyTorch.
- Reproducible latent/camera comparison against a float32 baseline.

The SDPA keeps full attention semantics. It does not use top-k attention,
token pruning, low-rank attention, or a materialized `Lq x Lk` attention
matrix.

## Scope and quantization status

The float32 strict result above covers the Flow model plus CFG/Euler sampler.
The repository now also includes a CPU-only end-to-end runner from a raw image
through background removal, DINO/Flux-VAE condition encoding, s20 Flow
inference, Gaussian decoding, PLY/SPLAT export, reference rendering, and a
standalone WebGL viewer. One 1024-pixel validation run completed with 262,144
Gaussians and preserved the same Flow quality metrics as the strict baseline.

Packed nonlinear quantization now passes the one-hour s20 gate. The selected
NF24 int16 layout uses an irregular NF8-derived first stage and two shared-scale
signed-int8 residual stages. The first two codes are combined as
`q01 = 4*q0 + q1`; AVX-512 decodes `(scale/1024) * (256*q01 + q2)` before
FMA. It stores 24 bits per weight, retains no float32 Linear weights, and covers
all 206 active Linear modules with zero fallback.

| Linear weights | Packed/original bytes | Validation | Result |
|---|---:|---:|---|
| NF8, one stage (8 bits) | 25.15% | s1 combined RMSE 2.51245e-2 | rejected on quality |
| Residual NF8, two stages (16 bits) | 50.20% | s1 combined RMSE 4.50681e-4 | rejected on quality |
| NFR8x3, three streams (24 bits) | 75.26% | s20 combined RMSE 2.31568e-5 | quality pass, superseded on speed |
| NF24 int16, combined first two stages (24 bits) | 75.11% | s20 combined RMSE 9.37666e-5 | selected; 2747.270 s |

The selected NF24 int16 row-tile-8 + exact SDPA query-8/key-tile-512 s20 run took
2747.270 s with a project-local GCC13 Zen 4 build. This is 724.060 s (20.9%)
faster than the previous 3471.330 s NF24 implementation, 575.616 s faster than
the strict float32 AVX-512 run, and 8109.118 s faster than the original CPU
float32 baseline. Native packed Linear time was 1008.472 s, SDPA time was
1432.319 s, camera RMSE was 8.93055e-6, and NaN/Inf/fallback counts were zero.
This meets the under-3000-second gate without changing TripoSplat semantics.

A resumable prepacked checkpoint format is also implemented. It stores
1,113,368,944 bytes across 206 Linear shards plus non-Linear state. Direct load
did not call the source-checkpoint loader and produced bit-identical latent and
camera arrays. Process-tree peak RSS fell from 3,437,973,504 to 2,551,123,968
bytes (25.8%). Checksum verification made this direct-load s1 run slower than
runtime packing, so the direct loader is currently a disk/memory improvement,
not a speed claim. Derived packed weights are not distributed by this repo.

Current evidence is in
[`docs/triposplat_nf24_i16_q8t512_s20_validation_20260721_ja.md`](docs/triposplat_nf24_i16_q8t512_s20_validation_20260721_ja.md).
The superseded NFR8x3 result remains documented in
[`docs/triposplat_nfr8x3_s20_validation_20260721_ja.md`](docs/triposplat_nfr8x3_s20_validation_20260721_ja.md).
The next optimization audit and prioritized future work are documented in
[`docs/triposplat_cpu_optimization_future_work_20260721_ja.md`](docs/triposplat_cpu_optimization_future_work_20260721_ja.md).
P0/P1 measurements and decisions are recorded in
[`docs/triposplat_cpu_p0_p1_validation_20260722_ja.md`](docs/triposplat_cpu_p0_p1_validation_20260722_ja.md).

## Requirements

- Linux x86-64 CPU with AVX-512F, AVX-512DQ, AVX-512BW, AVX-512VL, and FMA.
- GCC with OpenMP support and GNU binutils (`objdump`).
- Python 3.11 or later in a virtual environment.
- PyTorch and torchvision installed for the target CPU platform.
- Upstream TripoSplat checkpoints. Checkpoints are not stored in this repo.

## Setup

```bash
git clone https://github.com/ryo100794/triposplat-cpu-avx512.git
cd triposplat-cpu-avx512

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# Install torch and torchvision for your platform first.
.venv/bin/pip install -r requirements.txt

bash scripts/setup_upstream.sh
# Download the official TripoSplat checkpoint into this directory:
mkdir -p models/TripoSplat/ckpts

bash scripts/build_all.sh
```

The upstream project documents checkpoint download methods in its own README.
This repository neither redistributes nor modifies those weights.

## Strict s20 run

The strict run needs a prepared RGB image, encoded condition NPZ, deterministic
initial-noise NPZ, and the CPU float32 reference NPZ used for quality gating.

```bash
INPUT=/path/to/prepared_rgb.webp \
CONDITION_NPZ=/path/to/condition_1024.npz \
NOISE_NPZ=/path/to/noise_1024_seed0.npz \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
bash scripts/run_s20_strict.sh
```

All paths can be overridden with environment variables. `MODEL_THREADS`
defaults to 8 and `SDPA_THREADS` defaults to 4, matching the validated host.
Tune them for each CPU and run the quality comparison again.

The runner writes latent/camera NPZ data and a manifest. It does not claim that
the final Gaussian decoder or viewer has run.

## CPU end-to-end run

```bash
INPUT=/path/to/source.png \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
bash scripts/run_cpu_end_to_end_strict.sh
```

Use `RESUME=1` to continue a run after a completed stage. The runner performs a
capacity probe before allocating outputs because checkpoints and the Python
environment can leave little room on quota-limited filesystems.

## Packed nonlinear quantization

`NF24 int16` is the selected low-resource layout:

```bash
STEPS=20 RNF8_STAGES=3 \
RNF8_RESIDUAL_MODE=nf24_i16 \
RNF8_LIBRARY=artifacts/backends/libtriposplat_gemm_nf24_i16_avx512.so \
SDPA_LIBRARY=artifacts/backends/libtriposplat_sdpa_avx512_exact_q8t512.so \
SDPA_SYMBOL=triposplat_sdpa_f32_avx512_exact_q8t512 \
bash scripts/run_rnf8_strict.sh
```

Convert an official Flow checkpoint once, then load packed weights directly:

```bash
.venv/bin/python scripts/pack_triposplat_nf24_i16_checkpoint.py \
  --checkpoint /path/to/triposplat_fp16.safetensors \
  --output-dir /path/to/triposplat_nf24_i16_v1

TRIPOSPLAT_RNF8_PREPACKED_DIR=/path/to/triposplat_nf24_i16_v1 \
TRIPOSPLAT_RNF8_PREPACKED_VERIFY=1 \
STEPS=1 RNF8_STAGES=3 RNF8_RESIDUAL_MODE=nf24_i16 \
RNF8_LIBRARY=artifacts/backends/libtriposplat_gemm_nf24_i16_avx512.so \
SDPA_LIBRARY=artifacts/backends/libtriposplat_sdpa_avx512_exact_q8t512.so \
SDPA_SYMBOL=triposplat_sdpa_f32_avx512_exact_q8t512 \
bash scripts/run_rnf8_strict.sh
```

The converter is resumable and records source/shard SHA256 values. Matching
float32 reference NPZ files remain required for quality gating.

For the complete raw-image-to-viewer workflow, use:

```bash
INPUT=/path/to/source.png \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
TRIPOSPLAT_RNF8_PREPACKED_DIR=/path/to/triposplat_nf24_i16_v1 \
bash scripts/run_cpu_low_resource_nf24.sh
```

## SDPA microbenchmark

```bash
bash scripts/build_native_sdpa_avx512_exact_q8.sh
.venv/bin/python scripts/bench_native_sdpa_avx512_exact.py \
  --case self8194,8194,8194,-1,0 \
  --heads 16 --threads 4 --torch-threads 4 \
  --output-json artifacts/bench_sdpa_q8.json
```

Do not execute AVX-512 binaries on a CPU that lacks the required instruction
set.

## Repository layout

- `scripts/native_*.c`, `scripts/gemm_*.c`: native kernels.
- `scripts/native_*_patch.py`: strict PyTorch integration boundaries.
- `scripts/build_all.sh`: builds all libraries used by the strict run.
- `scripts/run_s20_strict.sh`: parameter-locked validation entry point.
- `scripts/run_cpu_end_to_end_strict.sh`: raw-image to Gaussian/viewer pipeline.
- `scripts/run_cpu_low_resource_nf24.sh`: selected NF24 s20 end-to-end entry point.
- `scripts/pack_triposplat_nf24_i16_checkpoint.py`: resumable packed-checkpoint converter.
- `scripts/run_nf8_strict.sh`, `scripts/run_rnf8_strict.sh`: packed nonlinear
  quantization evaluation entry points.
- `scripts/run_triposplat_quantized_param_batch.py`: research and trace runner.
- `docs/`: Japanese model, equation, parameter, milestone, and validation docs.

## License and upstream attribution

Original code in this repository is released under the MIT License. TripoSplat
is an independent upstream project by VAST/TripoAI and is also MIT-licensed.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). TripoSplat source, model
weights, sample images, and generated assets are not vendored here.
