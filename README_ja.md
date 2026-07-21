# TripoSplat CPU AVX-512

[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
のflow modelをCPU-onlyで実行するためのnative AVX-512 backendと検証scriptです。

> **このrepositoryは量子化済みTripoSplat modelを含みません。** `q8` は8-bit
> 量子化ではなく、8本のqueryを同時処理する `QUERY_BLOCK=8` を意味します。
> 合格したstrict s20ではmodel weight、Q/K/V、softmax、Linear演算はfloat32です。

## 検証結果

upstream commit `a78fa12d06dbf1381ca548bfac32bb68cb8c451d`、AMD EPYC
9654で検証しました。

| 項目 | 結果 |
|---|---:|
| 条件 | canvas 1024、20 steps、guidance 3.0、shift 3.0 |
| wall time | 3322.886秒（55分22.886秒） |
| 元のCPU float32 | 10856.388秒 |
| 高速化率 | 3.27倍 |
| native fallback | 20監視項目の合計0 |
| latent RMSE | 2.90157e-5 |
| camera RMSE | 3.72623e-6 |
| combined RMSE | 2.06857e-5 |
| NaN / Inf | 0 / 0 |
| 観測Python RSS | 約2.37-2.70 GiB |

詳細は
[`docs/triposplat_native_avx512_q8_s20_validation_20260720_ja.md`](docs/triposplat_native_avx512_q8_s20_validation_20260720_ja.md)
に記録しています。

## 実装範囲

- 有効な全Linear層のfloat32 AVX-512 GEMM
- dense、key-bias、final crossに対応するexact SDPAとonline softmax
- K/V loadを8 queryで共有するquery blocking
- GELU(tanh)、SiLU、LayerNorm、multi-head RMSNorm、RoPE
- modulation、residual、RePo、position/timestep embedding
- native CFG、Euler sampler
- 未対応時にPyTorchへ黙って戻らず例外にするstrict wrapper
- 元のfloat32 latent/cameraに対する品質比較

SDPAはfull attentionの意味を維持します。top-k、token削減、low-rank attention、
巨大な `Lq x Lk` attention matrixは使いません。

## 量子化について

合格したs20は量子化modelではありません。研究runnerにはfake quant、dynamic
quantization、非線形int4などの比較用optionが残っていますが、
[`scripts/run_s20_strict.sh`](scripts/run_s20_strict.sh) はそれらを有効化しません。
model全体の非線形量子化とpacked low-bit GEMMは未達です。

## 検証境界

今回の実測は準備済みcondition NPZとnoise NPZから始まり、flow modelとCFG/Euler
samplerを対象とします。DINO画像encode、背景除去、Gaussian decode、rendererを
含むend-to-end CPU完了を主張するものではありません。

## 必要環境

- AVX-512F/DQ/BW/VLとFMAを備えたLinux x86-64 CPU
- OpenMP対応GCCとGNU binutils
- `.venv` 内のPython 3.11以降
- 対象CPU向けPyTorch、torchvision
- 公式TripoSplat checkpoint

model、checkpoint、入力画像、latent、生成物はrepositoryへ含めません。

## Setup

```bash
git clone https://github.com/ryo100794/triposplat-cpu-avx512.git
cd triposplat-cpu-avx512

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
# torchとtorchvisionは利用環境に合うものを先に導入します。
.venv/bin/pip install -r requirements.txt

bash scripts/setup_upstream.sh
mkdir -p models/TripoSplat/ckpts
# 公式checkpointを上記directoryへ取得します。

bash scripts/build_all.sh
```

## strict s20

```bash
INPUT=/path/to/prepared_rgb.webp \
CONDITION_NPZ=/path/to/condition_1024.npz \
NOISE_NPZ=/path/to/noise_1024_seed0.npz \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
bash scripts/run_s20_strict.sh
```

`MODEL_THREADS=8`、`SDPA_THREADS=4` が検証時の既定値です。CPUごとにthread数を
測定し、変更後は必ず品質比較をやり直してください。runnerの出力はlatent/camera
NPZとmanifestであり、最終Gaussian生成まで完了したという意味ではありません。

## SDPA単体測定

```bash
bash scripts/build_native_sdpa_avx512_exact_q8.sh
.venv/bin/python scripts/bench_native_sdpa_avx512_exact.py \
  --case self8194,8194,8194,-1,0 \
  --heads 16 --threads 4 --torch-threads 4 \
  --output-json artifacts/bench_sdpa_q8.json
```

必要なAVX-512命令を持たないCPUではnative binaryを実行しないでください。

## Directory

- `scripts/native_*.c`, `scripts/gemm_*.c`: native kernel
- `scripts/native_*_patch.py`: strict PyTorch integration
- `scripts/build_all.sh`: strict構成の全library build
- `scripts/run_s20_strict.sh`: parameter固定のs20検証entry point
- `scripts/run_triposplat_quantized_param_batch.py`: 実験・trace runner
- `docs/`: model構造、数式、parameter、milestone、検証報告

## License

このrepositoryのoriginal codeはMIT Licenseです。TripoSplatはVAST/TripoAIによる
独立したMIT Licenseのupstream projectです。詳細は
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
