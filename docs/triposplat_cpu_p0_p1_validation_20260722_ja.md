# TripoSplat CPU最適化 P0/P1検証記録 2026-07-22

## 1. この記録の位置付け

本記録は、`triposplat_cpu_optimization_future_work_20260721_ja.md`で定義した課題のうち、
測定条件の正常化（P0）と厳密なstreaming/fusion候補（P1）の初回結果をまとめる。

目的は公式TripoSplatの出力を別手法で補正することではない。推論モデルが行う演算をCPU向けに
並べ替え、中間結果を再利用し、同じ出力品質をより少ない時間と一時領域で得ることである。

## 2. 測定条件

| 項目 | 条件 |
|---|---|
| CPU | AMD EPYC 9654 (Zen 4) |
| affinity | CPU 26, 28, 81, 83。3 thread測定は26, 28, 81 |
| baseline compiler | GCC 9.4.0 |
| candidate compiler | project-local GCC 13.1.0、`-march=znver4 -mtune=znver4` |
| Python | project-local venv |
| model/runtime dtype | float32 |
| weight | NF24、`int16(q0*4+q1) + int8(q2)` |
| SDPA | exact online softmax、query tile 8、key tile 512 |
| hardware counter | `perf_event_paranoid=4`のため未取得 |

GCC13はsystemへinstallせず、`toolchains/gcc13`へ置いた。
GCC9とGCC13の`cc1`、header、library探索環境はbuildごとに分離した。

## 3. P0 compiler結果

GCC13は、1から3 threadの初回正常化測定では4つのhot GEMMを概ね5から10%短縮した。
4 CPU再測定ではhost変動があり、shape別の効果は一様ではなかった。

| GEMM shape `(M,K,N)` | GCC9 4T | GCC13 4T | GCC13 / GCC9 |
|---|---:|---:|---:|
| 12294,1024,4096 | 0.582秒 | 0.540秒 | 0.929 |
| 12294,4096,1024 | 0.721秒 | 0.751秒 | 1.041 |
| 12294,1024,3072 | 0.593秒 | 0.465秒 | 0.784 |
| 12294,1024,1024 | 0.197秒 | 0.183秒 | 0.929 |

GEMMはcompiler変更だけを一律採用するのではなく、shapeごとのdispatchと複数回測定が必要である。
一方、SDPAのGCC13効果は4 CPUでも明確だった。

| SDPA長 | GCC9 4T | GCC13 4T | 短縮率 | 出力RMSE |
|---|---:|---:|---:|---:|
| 8194 | 0.993秒 | 0.841秒 | 15.4% | 5.99e-9 |
| 12294 | 2.597秒 | 2.006秒 | 22.7% | 5.11e-9 |

Clang 18もproject-localに評価した。GEMM単体はGCC9より5.7から8.8%速かったが、SDPAは
2倍以上遅く、PyTorchのlibgompとClangのlibompを同一processへ入れるとOpenMP threadが
実質1本になる競合も確認した。この構成は不採用とし、toolchain本体もquota節約のため削除した。

## 4. SDPA workspaceとtile順序

### 4.1 persistent workspace

K/V packed領域をcallごとの`posix_memalign/free`からthread-local persistent workspaceへ変更した。
warmup後の測定では、call中の追加allocationは0回、free時間も0になった。

| sequence長 | workspace capacity | pack / call (4T GCC13) |
|---|---:|---:|
| 8194 | 67,239,936 bytes | 0.0080秒 |
| 12294 | 100,794,368 bytes | 0.0114秒 |

allocationそのものは約microsecond級で、主要ボトルネックではなかった。効果の中心は容量上限を
固定し、将来QKV出力をpacked K/Vへ直接書くための受け口を作ったことである。

### 4.2 key micro-tile

key tileを64、128、256、512で比較した。3 thread、長さ8194/12294の中央値は次の通り。

| key tile | L=8194 | L=12294 |
|---:|---:|---:|
| 64 | 1.696秒 | 3.853秒 |
| 128 | 1.301秒 | 3.027秒 |
| 256 | 1.201秒 | 3.017秒 |
| 512 | 1.173秒 | 2.498秒 |

小tile online softmaxはscratchを減らすが、このCPUではloop、merge、指数計算の増加が上回った。
したがってkey tile 512を維持する。refの「局所領域だけを作る」という方針も、局所領域を
小さくし続ければよいわけではなく、cache容量と演算overheadの実測で境界を選ぶ必要がある。

query blockも4と8を4 CPUで比較した。block 4はblock 8に対し、L=8194で6.4%、
L=12294で12.4%遅かった。score scratch削減よりK/V再走査増が上回るため、query 8を維持する。

## 5. NF24 GEMMの行tile

現行kernelは、1つの復号済み16-weight vectorを16入力行のFMAへ再利用していた。
再利用行数を増やすtile 24と、レジスタ依存を短くするtile 8を同じ入力で比較した。

### 5.1 tile 24は不採用

tile 24は全shapeで出力RMSE 0だったが、tile 16の1.69から1.82倍の時間を要した。
重み復号回数の削減より、24本のaccumulator、長い命令列、register schedulerへの負荷が大きい。

### 5.2 tile 8は採用候補

4 CPUでの結果を示す。全候補で出力RMSEと最大絶対誤差は0だった。

| GEMM shape `(M,K,N)` | tile 16 | tile 8 | tile 8 / tile 16 |
|---|---:|---:|---:|
| 12294,1024,4096 | 0.649秒 | 0.498秒 | 0.767 |
| 12294,4096,1024 | 0.679秒 | 0.539秒 | 0.794 |
| 12294,1024,3072 | 0.521秒 | 0.302秒 | 0.578 |
| 12294,1024,1024 | 0.188秒 | 0.121秒 | 0.642 |

3 CPUでは25.8から41.1%、4 CPUでは20.6から42.2%の短縮を確認した。これは
「再利用回数を最大化する」より「復号、broadcast、FMAがCPUの実行幅に収まる順序を選ぶ」方が
重要な例である。inline assemblerへ進む前にC intrinsicのtile選択だけで大きな余地が残っていた。

## 6. 実モデルgate

### 6.1 tile 16 + persistent SDPA

| run | thread | wall time | Linear | SDPA | combined RMSE | camera RMSE | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| s1 | 3 | 230.898秒 | 101.227秒 | 94.234秒 | 2.228e-5 | 2.225e-5 | 0 |
| s4 | 3 | 882.755秒 | 404.811秒 | 387.992秒 | 9.852e-5 | 2.056e-5 | 0 |

s4のlatent RMSEは1.378e-4、NaN/Infは0で、目標2e-4を通過した。

### 6.2 tile 8 s1

| run | thread | wall time | Linear | SDPA | combined RMSE | camera RMSE | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| tile 16 s1 | 3 | 230.898秒 | 101.227秒 | 94.234秒 | 2.228e-5 | 2.225e-5 | 0 |
| tile 8 s1 | 3 | 185.349秒 | 63.130秒 | 86.672秒 | 2.228e-5 | 2.225e-5 | 0 |

tile 8はwall timeを19.7%、Linearを37.6%短縮し、最終latent/camera比較値はtile 16と同一だった。

### 6.3 tile 8 s4/4 CPU

| wall time | Linear | SDPA | combined RMSE | camera RMSE | latent RMSE | fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 605.282秒 | 217.599秒 | 306.038秒 | 9.852e-5 | 2.056e-5 | 1.378e-4 | 0 |

NaN/Infは0で、s4品質目標2e-4を通過した。tile 16 s4は3 CPU測定のため、
605.282秒と882.755秒をthread差を無視した正式な速度比には使わない。

### 6.4 tile 8 s20/4 CPU

| wall time | Linear | SDPA | combined RMSE | camera RMSE | latent RMSE | fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 2747.270秒 | 1008.472秒 | 1432.319秒 | 9.377e-5 | 8.931e-6 | 1.323e-4 | 0 |

旧NF24 3471.330秒から724.060秒、20.9%短縮し、strict float32 AVX-512の
3322.886秒も575.616秒、17.3%下回った。3000秒未満というP1第二目標を達成した。
品質値は旧NF24 s20と同一で、NaN/Infは0だった。

## 7. 演算順序と再利用の判断

実測から、次の順で進める。

1. NF24はZen 4でtile 8を既定候補とし、hot shapeごとに8/16をdispatchできる形を残す。
2. SDPAはkey tile 512を維持し、persistent K/V workspaceを使う。
3. QKV projectionが生成したK/Vをworkspaceのpacked layoutへ直接書き、packを除く。
4. MLPは行chunkだけでなくhidden tile内で`fc1 -> GELU -> fc2 partial sum`を完結させる。
5. unified blockでPython/ctypes境界と中間tensor生成をまとめて減らす。

s4 tile 16の内部時間は、forward 859.749秒、Attention 541.340秒、SDPA 387.992秒、
MLP 271.210秒だった。MLP内はfc1 132.672秒、GELU 7.816秒、fc2 124.617秒である。
このためGELUだけのassembly化ではなく、前後のGEMMと中間bufferを融合する必要がある。

厳密再利用では、既存のcondition context、position embedding、CFG重複state、負condition token
圧縮に加え、workspace、schedule、timestep embedding、packed weightをrun/frame間cacheする。
Q/K/Vやvelocityはstateがstepごとに変わるためcacheしない。

## 8. 精度を使う経路

厳密経路の次に、別profile `bounded_approx`で次を評価する。

1. MLP hiddenとVをBF16にし、normalization、softmax、residual、sampler stateはFP32を維持する。
2. layer感度を測って低感度層だけNF16へ落とす。
3. 末尾blockからcondition row freezeを1 blockずつ広げる。
4. MLP tile寄与上限とattention残存mass上限で、省略誤差を数値管理する。

無条件top-k、全層一括NF16、前stepのQKV再利用は行わない。s1/s4/s20のRMSE gateに加え、
6視点のworst-view品質を通過した候補だけを昇格する。

## 9. 未達事項

tile 8のs4/4 CPUとs20/4 CPUは品質gate内で完了し、s20 2747.270秒で3000秒未満を達成した。以下が残る。
- hot shapeごとのtile dispatchをmanifestへ記録する。
- QKVからpacked K/Vへの直接storeを実装する。
- streaming MLPで`M x 4096`中間tensorをなくし、scratch peakを記録する。
- timestep/schedule cacheのframe間hitを実動画で測る。
- BF16 V/MLPのs1感度表を作る。
- hardware counterを許可した環境でIPC、LLC miss、帯域を取得する。

inline assemblerはこれらの後に、disassemblyとcounterでcompiler生成コードの問題が特定できた
内側loopだけへ適用する。
