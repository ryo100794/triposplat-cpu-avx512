# TripoSplat CPU AVX-512

[VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat)
のflow modelをCPU-onlyで実行するためのnative AVX-512 backendと検証scriptです。

> **このrepositoryはTripoSplat checkpointを再配布しません。** 合格したbaselineの
> `q8` は8-bit量子化ではなく、8本のqueryを同時処理する `QUERY_BLOCK=8` を
> 意味します。strict s20 baselineはfloat32です。packed非線形weightを実行する
> NF8/RNF8 runnerは別の実験実装として下記に結果を記載します。

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
- CPU背景除去、DINO/Flux-VAE condition encode、決定的noise生成、Gaussian decode、
  low-memory export、reference render、単一WebGL viewer
- Flow modelの全206 Linearを対象に、float32 Linear weightを保持しないpacked NF8、
  2段/3段residual-NF8 AVX-512 GEMM
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

packed非線形量子化もs20検証まで完了しました。採用したNFR8x3は、1段目を
非線形NF8、2段目と3段目を符号付き対称int8 residualとする方式です。1 weight
あたり24 bitで、float32 Linear weightを保持せず、全206 LinearをAVX-512で
packed状態のまま実行します。

| Linear weight | packed/original byte | 検証 | 判定 |
|---|---:|---:|---|
| 1段NF8（8 bit） | 25.15% | s1 combined RMSE 2.51245e-2 | 品質不合格 |
| 2段residual NF8（16 bit） | 50.20% | s1 combined RMSE 4.50681e-4 | 品質不合格 |
| 3段NF8 residual（24 bit） | 75.26% | s4 combined RMSE 1.16460e-4 | s4合格 |
| NFR8x3: NF8 + 2段signed-int8 residual（24 bit） | 75.26% | s20 combined RMSE 2.31568e-5 | s20合格 |

採用したNFR8x3のs20は全体4640.813秒、native packed Linear 2365.633秒で、
Linear fallback、NaN、Infはいずれも0でした。元のCPU float32に対するlatent RMSEは
3.25672e-5、camera RMSEは3.44202e-6です。strict native float32 Gaussianとの
6視点render比較は平均PSNR 76.24 dB、worst view 69.33 dBでした。Linear weightは
1,480,996,180 byteから1,114,596,688 byteへ減少しています。

現在は公式float32 checkpointをloadした後にruntimeでpackします。pack後はfloat32
Linear weightを解放しますが、事前pack済みweight loaderを実装するまでは起動時peakと
元checkpoint file容量は減りません。

これにより、品質を維持するpacked s20実装と、元のCPU float32 10856.388秒に対する
2.34倍の高速化は確認できました。一方、strict native float32 AVX-512の3322.886秒
より39.7%遅く、1時間未満の低リソース速度目標は未達です。残る課題はTripoSplatの
意味を変えず、residual stageのdecode/GEMM overheadを削減することです。

詳細は
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

```bash
STEPS=1 bash scripts/run_nf8_strict.sh
STEPS=1 RNF8_STAGES=3 bash scripts/run_rnf8_strict.sh

STEPS=20 RNF8_STAGES=3 \
RNF8_RESIDUAL_MODE=symmetric_int8 \
RNF8_LIBRARY=artifacts/backends/libtriposplat_gemm_nfr8_avx512.so \
bash scripts/run_rnf8_strict.sh
```

決定的な同条件のfloat32 reference NPZが必要です。生成される比較JSONを確認し、
品質gateを通過した構成だけを長いsampler runへ進めます。

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
- `scripts/run_nf8_strict.sh`, `scripts/run_rnf8_strict.sh`: packed非線形量子化評価
- `scripts/run_triposplat_quantized_param_batch.py`: 実験・trace runner
- `docs/`: model構造、数式、parameter、milestone、検証報告

## License

このrepositoryのoriginal codeはMIT Licenseです。TripoSplatはVAST/TripoAIによる
独立したMIT Licenseのupstream projectです。詳細は
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。
