# TripoSplat CPU P3 bounded approximation監査

日付: 2026-07-22

## 1. 目的と区分

P3では公式TripoSplatへ機能を追加せず、CPU実装の一部だけに管理された誤差を許し、
演算量とmemory trafficを減らせるか調べた。`strict_nf24`の結果とは混ぜず、候補は
`bounded_approx`として扱う。s1の昇格gateはcombined RMSE `2e-4`以下、NaN/Inf 0である。

今回のBF16実装は速度実装ではなく、FP32中間値をBF16へ丸めてFP32へ戻す感度プローブである。
したがって合格は「その箇所を将来BF16 storage/dotへ変更できる可能性」を示すだけで、
現在のrunnerへ採用できることを意味しない。

## 2. packed K/V BF16

Q、softmax reduction、accumulatorはFP32のまま、packed K、VだけをBF16 roundtripした。
実形状microbenchmarkの局所誤差は次のとおりだった。

| 対象 | packed K RMSE | packed V RMSE | 最大絶対差 |
|---|---:|---:|---:|
| K | 1.66384e-3 | 0 | 1.55602e-2 |
| V | 0 | 1.66023e-3 | 1.56183e-2 |
| K/V | 1.66384e-3 | 1.66023e-3 | 1.56183e-2 |

s1の最終出力はすべて不合格だった。

| 対象 | elapsed | combined RMSE | latent RMSE | camera RMSE | 判定 |
|---|---:|---:|---:|---:|---|
| V | 143.024秒 | 6.68114e-4 | 3.55428e-4 | 8.75456e-4 | 不合格 |
| K | 142.086秒 | 9.09946e-4 | 8.76525e-4 | 9.42182e-4 | 不合格 |
| K/V | 141.807秒 | 1.26249e-3 | 8.13363e-4 | 1.58940e-3 | 不合格 |

KはQK scoreを通じてsoftmax分布を変えるため、V単独より最終誤差が大きかった。全24 blockの
K/V一括BF16化は棄却する。次に再開する場合はblock別Vだけを測り、K、Q、softmaxはFP32に保つ。

## 3. MLP hidden BF16感度

対象は各main blockの`fc1 -> GELU`出力である。全層一括ではcombined RMSE `1.55174e-3`、
四分位6層群もすべて不合格だった。

| block群 | combined RMSE | latent RMSE | camera RMSE |
|---|---:|---:|---:|
| 0-5 | 7.66176e-4 | 3.06720e-4 | 1.03922e-3 |
| 6-11 | 5.85303e-4 | 2.19632e-4 | 7.98074e-4 |
| 12-17 | 4.44191e-4 | 2.21091e-4 | 5.87987e-4 |
| 18-23 | 1.10196e-3 | 7.10521e-4 | 1.38701e-3 |

最も低感度だった12-17を単層へ分解した。

| block | combined RMSE | camera RMSE | s1 gate |
|---:|---:|---:|---|
| 12 | 1.19667e-4 | 1.40688e-4 | 合格 |
| 13 | 1.98613e-4 | 2.62263e-4 | 境界合格 |
| 14 | 1.77773e-4 | 2.34235e-4 | 合格 |
| 15 | 1.32991e-4 | 1.61473e-4 | 合格 |
| 16 | 2.12070e-4 | 2.83862e-4 | 不合格 |
| 17 | 2.52206e-4 | 3.45654e-4 | 不合格 |

余裕のあるblock 12と15を同時に丸めた場合はcombined `1.24845e-4`、latent
`1.30439e-4`、camera `1.18988e-4`だった。誤差は単純加算ではなく、組合せ評価が必要である。

ただしroundtrip時間はblock 12+15で0.159秒増え、s1 wall timeは145.027秒だった。BF16値を
保持したまま`vdpbf16ps`でfc2を実行するkernelは未実装なので、この候補はs4へ昇格せず未採用とする。

## 4. NF24/NF16層別混在

既存の全Linear RNF8x2、すなわち16 bit weight候補はs1 combined RMSE `4.50681e-4`で
不合格だった。現行native patchはstrict modeで全Linear coverageを要求し、選択外moduleが
1つでもあると停止する。このためNF24とNF16を同一model内へ混在させる正式runnerはまだない。

次の実装は二段階にする。

1. patchへ`require_full_coverage=False`を追加し、指定moduleだけを置換できるようにする。
2. 低感度moduleをRNF8x2、残りをNF24で置換し、全Linearの和集合coverageと重複0を検証する。
3. まずblock 12/15のMLP `fc1`、`fc2`を個別評価し、QKV、camera、finalはNF24を維持する。
4. weightの再量子化を避けるため、mixed manifestとprepacked loaderを追加する。
5. 容量だけでなくGEMM時間が短縮しない場合は採用しない。

## 5. late condition freeze

standaloneの`late_selective_condition_freeze`実装は存在するが、採用中のnegative-condition圧縮は
main blockを独自loopで実行するため、その`block.forward` patchを通らない。既存実装を有効に
しただけでは現在のNF24 packed-v3経路を評価できない。

統合時は圧縮loopの末尾N blockで、latent/camera queryだけを計算し、condition rowを前blockの
値で固定する。N=1は最終的に消費されないcondition出力の省略なので厳密、N=2以上は後続blockの
K/Vが変わる近似である。N=2から1段ずつ増やし、正枝と負枝を別々に測る。camera誤差が増えやすい
ため、s1 combinedだけでなくcameraと6視点を必須にする。

## 6. 寄与上限skip

### 6.1 MLP tile

hidden tileを`h`、activationを`a_h`、fc2 weightを`W2_h`とすると、

```text
norm(a_h W2_h) <= norm(a_h) * norm(W2_h)
```

で出力寄与を上から抑えられる。ただしfc1計算後に判定するとfc1時間は減らず、W2 tileだけの
skipになる。checkpoint pack時に`norm(W2_h)`を保存し、実行時の`norm(a_h)`をGELU epilogueで
同時計算する。上限がresidual normに対する予算以下のtileだけを省く。現時点ではkernelも
実測分布もないため未実装課題であり、無根拠なthreshold skipは採用しない。

### 6.2 attention mass

処理済みsoftmax最大値を`m`、指数和を`l`、未処理key tileのscore上限を`u_t`、key数を
`n_t`とすると、追加mass上限は次で評価できる。

```text
r_t <= n_t * exp(u_t - m) / l
```

key normだけを使うCauchy-Schwarz境界は緩すぎる可能性が高い。最初にskipせず全tileの`r_t`
分布だけを記録し、実際に誤差予算以下となるtile比を測る。比率が低ければ実装しない。
centroid/radius境界を追加する場合も、その境界計算時間をQK/PV削減時間から差し引く。

## 7. refから採る実行順序

初期refから採るのはGaussianの式ではなく、局所生成、即時消費、固定workspace、寄与上限skip、
stage再利用という実行原則である。今回の結果を踏まえた順番は次のとおり。

1. QKV/RoPE/RMSNorm/packのように次段が直ちに読む境界だけを融合する。
2. 大きなweightを再走査するMLP row streamingは行わない。
3. model、packed weight、負condition、schedule、embedding、scratchをframe間で再利用する。
4. 近似はMLP block 12/15のように実測で低感度な箇所から入れる。
5. skipは出力寄与の上限を計算できる場合だけ行う。
6. K/V、QKV、state、velocityはstep依存なのでstrict cacheへ入れない。

## 8. P3判定

- 全層K、V、K/V BF16: 品質不合格。
- 全層・6層群MLP BF16: 品質不合格。
- block 12+15 MLP BF16: s1品質合格だが速度kernel未実装のため未採用。
- NF24/NF16混在: partial coverageとmixed prepackが未実装。
- late condition freeze: packed negative-condition経路への統合が未実装。
- MLP/attention寄与skip: 上限式は定義済み、分布計測とkernelが未実装。

したがってP3から現行runtimeへ昇格する候補はない。成果は、誤差を入れられる層と入れられない
演算を実測で分離し、次の実装条件を具体化したことである。raw成果物はGDriveへchecksum確認後に
移し、共有先から削除した。
