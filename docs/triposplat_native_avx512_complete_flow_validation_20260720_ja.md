# TripoSplat flow推論 完全native AVX-512構成 検証報告 2026-07-20

## 1. 結論

AMD EPYC 9654上で、今回使用しているTripoSplatのCPU flow推論経路をfloat32 native AVX-512カーネルへ置き換え、strict modeでs1とs4を完走した。

- s1: 全native監視19項目のfallback合計0
- s4: 全native監視19項目のfallback合計0
- s4総時間: 743.843秒、8 threads
- s4基準比combined RMSE: 8.35589e-6
- 実行中に観測したPython RSS: 約2.65-2.70 GiB
- operator auditで大規模なPyTorch算術演算は0件

ここでいう「完全native」は、準備済みcondition NPZとnoise NPZを入力にしたflow modelとCFG/Euler samplerの有効経路を指す。DINO画像encoder、背景除去、VAE/GS decoder、renderer、および無効化されている実験用量子化分岐までは含めない。

## 2. native化した有効経路

| 区分 | native実装 | 主な処理 |
|---|---|---|
| Linear | `libtriposplat_gemm_f32_avx512.so` | flow model内の全206 `nn.Linear`、range出力を含む |
| Attention | `libtriposplat_sdpa_avx512_exact.so` | positive dense、negative key-bias、final cross SDPA |
| Activation | `libtriposplat_gelu_avx512.so`, `libtriposplat_activations_avx512.so` | GELU(tanh)、SiLU |
| Norm/rotation | `libtriposplat_norm_rope_avx512.so` | LayerNorm、multi-head RMSNorm、RoPE complex multiply |
| Block elementwise | `libtriposplat_block_elementwise_avx512.so` | modulation、gated residual、plain add |
| RePo | `libtriposplat_repo_avx512.so` | feature multiply、動的phasor生成 |
| Embedding | `libtriposplat_embeddings_avx512.so` | 固定3D位置埋め込み、timestep埋め込み |
| Sampler | `libtriposplat_sampler_avx512.so` | CFG、Euler、可変刻みAB2 |

各Python patchはCPU float32、contiguous、shape、batchを検査する。strict modeでは条件不一致時にPyTorchへ黙って戻らず例外にする。s1/s4とも例外はなく、manifest内の全fallback合計は0だった。

## 3. samplerで置き換えた式

### CFG

正条件予測を `v_pos`、負条件予測を `v_neg`、guidance scaleを `g` とすると、元実装は次式である。

```text
v_cfg = g * v_pos - (g - 1) * v_neg
```

`triposplat_cfg_combine_inplace_f32_avx512` は16個のfloat32を1本のZMM registerで処理し、結果を `v_pos` の領域へ直接書き戻す。中間テンソル `g*v_pos` と `(g-1)*v_neg` は作らない。

### Euler

現在状態を `x_i`、step幅を `dt_i` とすると、更新式は次のとおりである。

```text
x_(i+1) = x_i - dt_i * v_cfg
```

`triposplat_euler_update_inplace_f32_avx512` は状態へ直接書き戻す。s4ではCFGとEulerを各4回、各524,308要素処理した。

### 可変刻みAB2

AB2は今回のEuler s4では呼ばれていないが、同じlibraryへ実装し単体比較を通した。

```text
r = dt_i / max(dt_(i-1), 1e-12)
v_eff = (1 + r/2) * v_i - (r/2) * v_(i-1)
x_(i+1) = x_i - dt_i * v_eff
```

単体試験ではCFG、Euler、AB2のすべてがPyTorch参照式に対してrelative RMSE 0、max abs 0だった。コンパイルは `-ffp-contract=off` とし、元式のmul/sub順序を保持した。

## 4. 実行時operator audit

`torch.profiler` を `FlowEulerCfgMultiVariantSampler.sample` の範囲だけに適用した。全39種類の集約operator中、算術候補は次だけだった。

| operator | calls | input shape | self CPU time |
|---|---:|---|---:|
| `aten::add` | 6 | `[3]` とscalar | 2.74 ms |

`aten::mm`、`aten::bmm`、softmax、GELU、SiLU、norm、特徴量tensorに対するmul/div/subは残っていない。残る6加算は3要素の座標・制御補助計算で、forward全体に占める比率は0.001%未満である。

一方、次のdata movement/layout operatorは残る。

- `copy_`, `clone`, `contiguous`
- `cat`, `index_select`, `repeat`
- `view`, `reshape`, `permute`, `slice`, `chunk`
- `empty`, `empty_like`, `empty_strided`

これらは算術backendではなくtensor配置と入出力領域の管理である。「PyTorchを一切呼ばない単体Cアプリケーション」ではない点は区別する。

## 5. 品質結果

### s1、2 threads、strict

run ID:

```text
runtime_native_avx512_complete_model_sampler_s1_g3_1024_20260720
```

| 項目 | 結果 |
|---|---:|
| elapsed | 343.610秒 |
| flow forward | 330.579秒 |
| latent RMSE | 3.36863e-6 |
| camera RMSE | 3.52404e-6 |
| combined RMSE | 3.44721e-6 |
| native fallback合計 | 0 |

### s4、8 threads、strict、単独実行

run ID:

```text
runtime_native_avx512_complete_model_sampler_s4_g3_1024_t8_20260720
```

| 項目 | 結果 |
|---|---:|
| elapsed | 743.843秒 |
| step 1 forward | 181.840秒 |
| step 2 forward | 180.911秒 |
| step 3 forward | 184.784秒 |
| step 4 forward | 186.012秒 |
| latent RMSE | 1.09704e-5 |
| latent max abs | 4.23372e-4 |
| camera RMSE | 4.39222e-6 |
| combined RMSE | 8.35589e-6 |
| NaN | 0 |
| native fallback合計 | 0 |

s4は公式float32同条件の `baseline_conditionnpz_exact_s4_g3_1024_20260719` と比較した。誤差は4 stepの累積後も小さく、出力は全要素finiteだった。

## 6. threadsとメモリ

2/4/8 threads s1を比較した。4と8は同時実行したため、絶対値には相互競合が含まれる。

| threads | forward | 条件 |
|---:|---:|---|
| 2 | 330.579秒 | 単独 |
| 4 | 364.199秒 | 8-thread runと並列 |
| 8 | 306.819秒 | 4-thread runと並列 |

並列状態でも8 threadsが先に完走したため、s4は8 threadsを採用した。s4単独時は1 stepあたり約183.4秒だった。

リモートcgroup hard limitは32,000,000,000 bytesだった。運用上のsoft targetを常用約12GB、瞬間約14GBへ緩めたが、今回の1 process RSSは約2.65-2.70 GiBに留まった。4/8 threadsの2 process並列時もPython RSS合計は約4.48 GiBで、OOMは発生しなかった。

## 7. 公開版での再現方法

公開repositoryでは `scripts/build_all.sh` でnative libraryをbuildし、`scripts/run_s20_strict.sh` を入口にする。Pythonは `.venv` を使い、TripoSplat upstream、checkpoint、入力、condition/noise、比較基準のpathを環境変数で指定する。

## 8. 検証証跡

runのmanifest、latent、比較JSONはprivateな入力依存成果物なのでrepositoryへ含めない。公開版には再現script、kernel source、品質・時間の集計文書を含める。最終s20結果は `triposplat_native_avx512_q8_s20_validation_20260720_ja.md` を参照する。

## 9. 2026-07-21更新と残件

初版後、DINO encoderとGS decoderを含むCPU end-to-end、全206 LinearのNFR8x3、
s20品質・速度、最終Gaussian/viewerまでを検証した。したがって初版の項目3、4、5は
達成済みである。詳細は `triposplat_nfr8x3_s20_validation_20260721_ja.md` を参照する。

現在残る課題:

1. exact SDPAのthread scalingとcache blockingを改善する。
2. data movementの `copy_/clone/contiguous/cat` を減らし、scratch bufferを再利用する。
3. NFR8x3 s20を4640.813秒から3600秒未満へ短縮する。
4. 事前pack済みweightを直接loadし、起動時もfloat32 Linear weightを不要にする。
5. NFR8x3をraw画像からviewerまでの単一entry pointへ統合する。

strict float32構成の主要flow演算はnative化済みである。NFR8x3も全Linearをpacked
AVX-512で実行するが、速度ではstrict float32 backendが依然として優位である。


## 10. 2026-07-21 NF24 int16による残件完了

項目3-5は次の構成で完了した。

- 全206 Linear: NF24 int16、24 bit/weight、float32 Linear weight保持0、fallback 0
- exact SDPA: query block 8、key tile 512
- s20: 3471.330秒、combined RMSE 9.37666e-5、camera RMSE 8.93055e-6
- 事前pack直接load: source checkpoint loader不使用、runtime-pack出力とbit一致
- process-tree peak RSS: 3,437,973,504 byteから2,551,123,968 byteへ25.8%削減
- 単一入口: `scripts/run_cpu_low_resource_nf24.sh`

したがって3600秒未満、事前pack直接load、raw画像からviewerまでのNF24統合入口は
達成済みである。exact SDPAの追加最適化と1800秒未満は次期性能目標であり、本報告の
必須残件ではない。詳細は
`triposplat_nf24_i16_q8t512_s20_validation_20260721_ja.md` を参照する。
