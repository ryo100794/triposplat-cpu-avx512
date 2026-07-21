# TripoSplat CPU-only native AVX-512 q8 s20 検証報告 2026-07-20

## 1. 結論

AMD EPYC 9654上で、準備済みcondition/noiseを入力にするTripoSplat flow推論をCPU-only、20 steps、guidance 3.0、1024 canvasで完走した。

| 判定項目 | 結果 | 合格条件 | 判定 |
|---|---:|---:|---|
| wall time | 3322.886秒（55分22.886秒） | 3600秒未満 | 合格 |
| native fallback | 20監視項目、合計0 | 0 | 合格 |
| latent RMSE | 2.90157e-5 | combined RMSE 5.0e-4以下 | 合格 |
| camera RMSE | 3.72623e-6 | 1.0e-4以下 | 合格 |
| combined RMSE | 2.06857e-5 | 5.0e-4以下 | 合格 |
| NaN / Inf | 0 / 0 | 0 / 0 | 合格 |

比較対象は同一condition、noise、seed、step、guidance、shiftを使った元のCPU float32 s20である。元のCPU float32 s20は10856.388秒だったため、今回の構成は約3.27倍速く、wall timeを69.4%短縮した。

run ID:

```text
runtime_native_avx512_complete_model_sampler_q8_s20_g3_1024_mt8_st4_20260720
```

## 2. q8の意味

この実装名の `q8` は8-bit量子化ではない。1回の内側ループで8本のqueryを同時に処理する `QUERY_BLOCK=8` を表す。

- Q、K、V、出力はfloat32のまま
- softmaxもfloat32のonline softmax
- attention matrix全体は生成しない
- 近似的なtoken削減、top-k、低rank化は使わない
- SIMD幅はAVX-512の16 float/ZMM

したがって、今回の誤差は量子化誤差ではなく、演算順序とnative kernelへの置換に伴うfloat32丸め差である。非線形量子化モデルの完成を意味しない。

## 3. exact SDPAの式

head dimensionを `d=64`、queryを `q_i`、keyを `k_j`、valueを `v_j` とする。`i` はquery行、`j` はkey行を表す。

通常のscaled dot-product attentionは次式である。

```text
s_ij = dot(q_i, k_j) / sqrt(d) + b_j
p_ij = exp(s_ij) / sum_t exp(s_it)
o_i  = sum_j p_ij * v_j
```

`b_j` はnegative compact経路で使うkey biasで、biasなし経路では0である。巨大な `s_ij` 行列を保存せず、keyを順に読んで各queryの最大値 `m_i`、正規化分母 `l_i`、未正規化出力 `a_i` を更新する。

```text
m_i' = max(m_i, s_ij)
alpha_i = exp(m_i - m_i')
beta_ij = exp(s_ij - m_i')
l_i' = alpha_i * l_i + beta_ij
a_i' = alpha_i * a_i + beta_ij * v_j
o_i = a_i / l_i
```

最大値が更新されたときに過去の累積値を `alpha_i` で再スケールするため、通常softmaxと同じ意味を保ったまま数値的なoverflowを避けられる。

## 4. q8でdata movementが減る理由

従来のquery block 4版は、同じkey/value列を4 queryごとに読み直していた。q8版は8 queryのscoreとonline softmax状態をregister/stack上で並行管理する。

1. 1本の `k_j` を読み込む。
2. 8本の `q_i` とのdot productを計算する。
3. 1本の `v_j` をZMM単位で読み込む。
4. 8本の出力accumulatorを、それぞれの `beta_ij` でFMA更新する。

同じ8 queryを処理する場合、q4ではK/V走査が2回必要だが、q8では1回になる。計算量の次数 `O(Lq * Lk * d)` は変わらない一方、K/Vのload、loop制御、共有可能なaddress計算を減らす。query blockを無制限に増やすとregister pressure、spill、thread scalingが悪化するため、このCPUでは実測で8を採用した。

## 5. kernelとthread選択

native SDPA library:

```text
artifacts/backends/libtriposplat_sdpa_avx512_exact_q8.so
symbol: triposplat_sdpa_f32_avx512_exact_q8
SHA256: ef7e7b86e115807d0c09ce55a1be7c649409b49356000dbe4814eb07d58ce78d
```

source SHA256:

```text
scripts/native_sdpa_avx512_exact_q8.c
2033aa8ac865e6a04ca29a5cb6f005de704674bed78368b4dc6a9cd629436402
```

実長ベンチではq4比で、L=8194が14.8%、L=12294が4.1%短縮した。q8のthread sweepでは4 threadsが最速だった。

| SDPA threads | L=8194 | L=12294 |
|---:|---:|---:|
| 2 | 2.390秒 | 5.393秒 |
| 3 | 1.600秒 | 3.609秒 |
| 4 | 1.197秒 | 2.722秒 |
| 6 | 1.323秒 | 2.974秒 |
| 8 | 1.326秒 | 3.464秒 |
| 12 | 1.390秒 | 3.375秒 |
| 16 | 1.369秒 | 3.512秒 |

そのため、Linear、MLP、Norm、RePo、samplerなどは8 threads、SDPAだけ4 threadsとした。逆アセンブルではZMM命令を確認し、主な命令数は `vmovaps` 204、`vfmadd231ps` 59、`vmulps` 54、`vmovups` 40だった。

## 6. 段階検証

### s1 strict

| 項目 | 値 |
|---|---:|
| elapsed | 180.861秒 |
| forward | 170.567秒 |
| SDPA | 108.378秒 |
| attention total | 127.428秒 |
| MLP | 34.789秒 |
| combined RMSE | 3.44721e-6 |
| fallback | 0 |

### s4 strict

| 項目 | q8 | 旧q4 |
|---|---:|---:|
| elapsed | 644.162秒 | 743.843秒 |
| forward | 634.145秒 | 733.469秒 |
| SDPA | 387.492秒 | 486.008秒 |
| combined RMSE | 8.35589e-6 | 8.35589e-6 |
| fallback | 0 | 0 |

q8はs4のwall timeを13.4%、SDPA時間を20.3%短縮し、品質値はq4 strictと同じだった。

### s20 strict

| 項目 | 値 |
|---|---:|
| elapsed | 3322.886秒 |
| forward合計 | 3313.178秒 |
| forward平均 | 165.659秒/step |
| forward最大 | 173.944秒 |
| forward最終step | 163.038秒 |
| SDPA | 2065.791秒、960 calls |
| attention total | 2446.504秒、960 calls |
| MLP | 699.802秒、960 calls |
| fallback監視項目 | 20 |
| fallback合計 | 0 |

SDPAはforwardの約62.4%、attention全体は約73.8%、MLPは約21.1%を占める。次の速度目標を1800秒未満に置く場合も、第一の対象はSDPAである。

## 7. 品質比較

元のCPU float32 s20に対する差分:

| tensor | RMSE | MAE | max abs | NaN | Inf |
|---|---:|---:|---:|---:|---:|
| latent | 2.90157e-5 | 1.24095e-5 | 0.00248206 | 0 | 0 |
| camera | 3.72623e-6 | 2.91765e-6 | 6.49691e-6 | 0 | 0 |

combined RMSEは `2.06857e-5` で、s20 gate `5.0e-4` の約4.1%である。最終NPZのSHA256は次のとおり。

```text
724f75ee522bc463c49432db10f4c6df2016b55808bbb18b7edbc21ba3fea2ec
```

## 8. native化の境界

今回fallback 0を確認した対象は、準備済みcondition/noiseからlatent/cameraを生成するflow modelとCFG/Euler samplerの有効経路である。

native化済み:

- 全206 Linearのfloat32 AVX-512 GEMM
- exact dense/key-bias/final cross SDPA
- GELU(tanh)、SiLU
- LayerNorm、multi-head RMSNorm、RoPE
- modulation、residual、add
- RePo feature multiplyとphasor
- position/timestep embedding
- CFGとEuler sampler

今回のs20に含まれないもの:

- 入力画像からconditionを作るDINO/image encoder
- 背景除去
- latent/cameraからGaussianを作るVAE/GS decoder
- rendererとviewer
- モデルweightの非線形量子化、packed int4 GEMM

したがってこれはTripoSplat全体のend-to-end CPU完了ではなく、最重部であるflow推論のstrict CPU目標達成である。TripoSplatの機能追加や別手法による奥行き補正は行っていない。

## 9. メモリ

実行中に観測したPython RSSは約2.37-2.70 GiBだった。remote cgroup上限は約32GBであり、OOMは発生していない。online softmaxにより `Lq x Lk` attention matrixを保持しないため、1024 canvasの長系列でもメモリ増加を抑えている。

## 10. 再現方法

```bash
git clone https://github.com/ryo100794/triposplat-cpu-avx512.git
cd triposplat-cpu-avx512
python3 -m venv .venv
bash scripts/setup_upstream.sh
bash scripts/build_all.sh
bash scripts/run_s20_strict.sh
```

公開版ではPythonを `.venv` に入れ、入力、condition、noise、比較基準を環境変数で指定する。詳細はrepositoryのREADMEを参照する。

## 11. 公開成果物の方針

公開repositoryにはsource、build/run script、検証文書だけを含める。model、checkpoint、入力画像、condition/noise、latent、生成物、private storage情報は含めない。

## 12. 2026-07-21更新

この節の初版で未達としていた次の項目は、その後の検証で達成した。

- CPU-only raw画像から背景除去、DINO/Flux-VAE condition、s20 Flow、GS decoder、
  PLY/SPLAT、renderer、単一WebGL viewerまで完走
- 全206 Linearを対象とするpacked非線形/residual量子化
- NFR8x3でs20品質gate、Linear fallback 0
- strict float32 Gaussianとの6視点比較

NFR8x3 s20の実測は次のとおり。

| 項目 | 結果 |
|---|---:|
| wall time | 4640.813秒 |
| combined RMSE | 2.31568e-5 |
| camera RMSE | 3.44202e-6 |
| packed/original Linear weight | 75.2599% |
| 6視点mean / worst PSNR | 76.24 / 69.33 dB |

詳細は `triposplat_nfr8x3_s20_validation_20260721_ja.md` を参照する。残る速度目標は
NFR8x3を3600秒未満、次に1800秒未満へ短縮すること、起動時float32 checkpointを
不要にする事前pack済みweight loaderを実装することである。


## 13. 2026-07-21 NF24 int16 follow-up

前節末尾の残件はNF24 int16 + SDPA key tile 512で完了した。s20は3471.330秒、
combined RMSE 9.37666e-5、camera RMSE 8.93055e-6、全206 Linear fallback 0である。
これによりNFR8x3の4640.813秒から25.2%短縮し、3600秒未満を達成した。

事前pack済み206 Linearを直接loadする経路も追加した。公式Flow checkpoint loaderを
呼ばず、runtime-pack版のlatent/cameraとbit完全一致し、process-tree peak RSSを
25.8%削減した。raw画像からviewerまでの単一入口は
`scripts/run_cpu_low_resource_nf24.sh` である。詳細は
`triposplat_nf24_i16_q8t512_s20_validation_20260721_ja.md` を参照する。
