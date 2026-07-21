# TripoSplat CPU AVX-512

[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
のflow modelをCPU-onlyで実行するためのnative AVX-512 backendと検証scriptです。

> **このrepositoryはTripoSplat checkpointを再配布しません。** 合格したbaselineの
> `q8` は8-bit量子化ではなく、8本のqueryを同時処理する `QUERY_BLOCK=8` を
> 意味します。exact strict s20 baselineはfloat32です。検証済み低リソース採用構成のNF24 int16と、
> その前段のNF8/residual-NF8評価を分けて下記に記載します。

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
- K/V loadを8 queryで共有し、検証済みkey tile 512を使うquery blocking
- GELU(tanh)、SiLU、LayerNorm、multi-head RMSNorm、RoPE
- modulation、residual、RePo、position/timestep embedding
- native CFG、Euler sampler
- CPU背景除去、DINO/Flux-VAE condition encode、決定的noise生成、Gaussian decode、
  low-memory export、reference render、単一WebGL viewer
- Flow modelの全206 Linearを対象に、float32 Linear weightを保持しないpacked NF8評価と、
  採用した24-bit NF24 int16 AVX-512 GEMM
- 公式Flow checkpointを起動時にloadしない、再開可能なNF24 converterと直接loader
- 未対応時にPyTorchへ黙って戻らず例外にするstrict wrapper
- 元のfloat32 latent/cameraに対する品質比較

SDPAはfull attentionの意味を維持します。top-k、token削減、low-rank attention、
巨大な `Lq x Lk` attention matrixは使いません。

## 検証境界と量子化状況

上表のfloat32 strict結果はFlow modelとCFG/Euler samplerを対象とします。現在は
raw画像からCPU背景除去、DINO/Flux-VAE condition encode、s20 Flow推論、Gaussian
decode、PLY/SPLAT export、reference render、単一WebGL viewerまで実行するCPU-only
end-to-end runnerも実装済みです。1024 pixel、262,144 Gaussiansの1検証runが完走し、
Flow部分はstrict baselineと同じ品質指標を維持しました。

packed非線形量子化は1時間未満のs20 gateまで完了しました。採用したNF24 int16は、
不等間隔のNF8由来1段目と、共有scaleのsigned-int8 residual 2段から成ります。
先頭2 codeを `q01 = 4*q0 + q1` としてint16へまとめ、AVX-512内で
`(scale/1024) * (256*q01 + q2)` をdecodeしてFMAします。1 weight 24 bit、
全206 Linear、float32 Linear weight保持0、fallback 0です。

| Linear weight | packed/original byte | 検証 | 判定 |
|---|---:|---:|---|
| 1段NF8（8 bit） | 25.15% | s1 combined RMSE 2.51245e-2 | 品質不合格 |
| 2段residual NF8（16 bit） | 50.20% | s1 combined RMSE 4.50681e-4 | 品質不合格 |
| NFR8x3、3 stream（24 bit） | 75.26% | s20 combined RMSE 2.31568e-5 | 品質合格、速度で更新 |
| NF24 int16、先頭2段統合（24 bit） | 75.11% | s20 combined RMSE 9.37666e-5 | 採用、3471.330秒 |

採用したNF24 int16 + SDPA key tile 512のs20は3471.330秒です。旧NFR8x3の
4640.813秒、元のCPU float32 10856.388秒から短縮しました。native packed Linearは
1519.392秒、SDPAは1645.894秒、camera RMSEは8.93055e-6で、NaN、Inf、fallbackは
すべて0です。TripoSplatの意味を変えず3600秒未満のgateを達成しました。

事前pack済みcheckpoint形式も実装しました。206 Linear shardと非Linear stateの合計は
1,113,368,944 byteです。直接loadは公式checkpoint loaderを呼ばず、runtime-pack版と
latent/cameraがbit完全一致しました。process-tree peak RSSは3,437,973,504 byteから
2,551,123,968 byteへ25.8%減少しました。checksum検証を含む直接load s1は今回
runtime-packより遅いため、現時点ではdisk/memory peak改善として採用し、速度改善とは
扱いません。派生weightはrepositoryへ含めません。

最新の証跡は
[`docs/triposplat_nf24_i16_q8t512_s20_validation_20260721_ja.md`](docs/triposplat_nf24_i16_q8t512_s20_validation_20260721_ja.md)、
旧NFR8x3の履歴は
[`docs/triposplat_nfr8x3_s20_validation_20260721_ja.md`](docs/triposplat_nfr8x3_s20_validation_20260721_ja.md)
に記録しています。

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

## CPU end-to-end

```bash
INPUT=/path/to/source.png \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
bash scripts/run_cpu_end_to_end_strict.sh
```

完了済みstageから再開するときは `RESUME=1` を指定します。quota制限環境で途中の
zero-byte出力を防ぐため、runnerは出力確保前にcapacity probeを行います。

## packed非線形量子化

採用構成は `NF24 int16` です。

```bash
STEPS=20 RNF8_STAGES=3 \
RNF8_RESIDUAL_MODE=nf24_i16 \
RNF8_LIBRARY=artifacts/backends/libtriposplat_gemm_nf24_i16_avx512.so \
SDPA_LIBRARY=artifacts/backends/libtriposplat_sdpa_avx512_exact_q8t512.so \
SDPA_SYMBOL=triposplat_sdpa_f32_avx512_exact_q8t512 \
bash scripts/run_rnf8_strict.sh
```

公式Flow checkpointを一度だけ変換し、packed weightを直接loadできます。

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

converterは再開可能で、source/shard SHA256をmanifestへ記録します。同条件の
float32 reference NPZによる品質gateは引き続き必要です。

raw画像からviewerまでの単一入口:

```bash
INPUT=/path/to/source.png \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
TRIPOSPLAT_RNF8_PREPACKED_DIR=/path/to/triposplat_nf24_i16_v1 \
bash scripts/run_cpu_low_resource_nf24.sh
```

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
- `scripts/run_cpu_end_to_end_strict.sh`: raw画像からGaussian/viewerまでのpipeline
- `scripts/run_cpu_low_resource_nf24.sh`: 採用NF24 s20のend-to-end単一入口
- `scripts/pack_triposplat_nf24_i16_checkpoint.py`: 再開可能なpacked checkpoint converter
- `scripts/run_nf8_strict.sh`, `scripts/run_rnf8_strict.sh`: packed非線形量子化評価
- `scripts/run_triposplat_quantized_param_batch.py`: 実験・trace runner
- `docs/`: model構造、数式、parameter、milestone、検証報告

## License

このrepositoryのoriginal codeはMIT Licenseです。TripoSplatはVAST/TripoAIによる
独立したMIT Licenseのupstream projectです。詳細は
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
