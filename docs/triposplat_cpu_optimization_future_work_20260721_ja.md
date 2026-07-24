# TripoSplat CPU最適化の今後課題とアルゴリズム再検討 2026-07-21

## 1. 目的

この文書は、現行のNF24 int16 CPU実装を「すでに限界まで最適化済み」とは扱わず、
今後の性能課題を具体化するものである。対象は次の3点である。

1. 演算の順番と中間データの持ち方を変え、同じ写像をより少ない転送で実行する。
2. 同一または再利用可能な結果を、step内、step間、frame間で再計算しない。
3. 品質gateで管理できる範囲だけ精度を使い、寄与の小さい演算を低精度化または省略する。

TripoSplatへ機能を追加したり、ref rendererの補正で見た目を合わせたりすることは目的に
含めない。公式TripoSplatと同じ役割をCPU低リソース実装へ置き換えることが主目的である。

## 2. 現在地と結論

採用中の低リソース基準はNF24 int16 + exact SDPA key tile 512である。

| 項目 | 現在値 |
|---|---:|
| s20 wall time | 3471.330秒 |
| native packed Linear | 1519.392秒、43.8% |
| exact SDPA | 1645.894秒、47.4% |
| Linear + SDPA | 3165.286秒、91.2% |
| combined RMSE | 9.37666e-5 |
| camera RMSE | 8.93055e-6 |
| Linear fallback | 0 / 7252 calls |
| 元のCPU float32 | 10856.388秒 |
| strict float32 AVX-512 | 3322.886秒 |

NF24はweight常駐量を削減しているが、毎回int codeをfloat32へ復号してFMAするため、
strict float32 AVX-512より149秒遅い。したがって、現在の量子化はメモリ改善として成立
している一方、量子化された整数値をpacked dot命令で直接積和する段階には達していない。

また、現行コードにはAVX-512 intrinsicとOpenMPが入っているが、次の理由から
「完全に詰め切ったCPU実装」とはまだ判定できない。

- 測定hostはAMD EPYC 9654だが、GCC 9.4.0のnative target判定は`znver2`である。
- NF24 GEMM buildは明示的なZen 4 targetを指定していない。
- process affinityは4 CPUに制限され、過去の4/8 thread比較には条件混在の疑いがある。
- `perf_event_paranoid=4`のため、cache miss、帯域、front-end stallをhardware counterで
  分離できていない。
- SDPAはcallごとにK/V全体を確保、転置、解放し、512 key分のscoreをstackへ保持する。
- hot kernelはCだが、block全体はPython、PyTorch tensor、ctypes呼び出しへ分断される。
- 対象CPUが持つAVX-512 BF16/VNNIを採用経路では使用していない。

結論として、inline assemblerの前に、演算融合、layout、workspace、BF16/VNNI、
実行回数の順で改善余地が残る。assemblerだけで1800秒へ到達する見込みはない。

## 3. 初期refから採用する着眼点

初期に提示された別実装`ref`について、既存の調査記録では`gs_minimal.py`と
`gs_colmap_pipeline.py`系に次の特徴があった。これはTripoSplat Flowの代替式ではなく、
計算順序と記憶領域を設計するための参考である。

| refの着眼点 | Flow CPU実装への対応 |
|---|---|
| geometryを固定しgradientを持たない | inference-onlyの定数経路を事前計算する |
| 投影楕円のlocal bboxだけを処理 | token/hidden/keyの小tileだけを作り、その場で消費する |
| full-frame中間配列を作らない | QKV、score、MLP 4096 hiddenの全体materializeを避ける |
| local patchへin-place合成 | thread-local workspaceとfused residual更新を使う |
| 画面外、小半径、微小alphaを早期skip | 寄与上限が誤差予算以下のtileだけskipする |
| transmittanceが尽きたら終了 | attention残存massの上限で打ち切りを判定する |
| chunk/streamingでbounded memory | 固定サイズscratch arenaで全stepを処理する |
| stage成果物とmanifestを再利用 | frame/stepに依存しない結果をkey付きcacheにする |

重要なのは「Gaussianの式をFlowへ持ち込む」ことではない。同じ数式でも、全体を一度
配列化してから処理するか、必要な小領域だけ生成して直ちに合成するかで、memory trafficと
peak RSSが大きく変わる。refから採用すべき核心はこの実行順序である。

## 4. 厳密経路で見直す演算順序

ここでは数学的な処理対象を減らさない。浮動小数点の加算順序差は発生し得るため、
最終的には既存RMSE gateで判定する。

### 4.1 MLPをhidden tile単位で完結させる

現行MLPは概ね次式である。入力を`X`、weightを`W1, W2`、biasを`b1, b2`とする。

```text
H = GELU(X W1 + b1)       Hの幅は4096
Y = H W2 + b2             Yの幅は1024
```

現在は`H`全体をPyTorch tensorとして生成する。hidden軸をtile`T_h`へ分ければ、
`W1`と`W2`の対応tileを読み、次の部分和を直ちに`Y`へ加算できる。

```text
Y = b2
for each hidden tile h:
    H_h = GELU(X W1_h + b1_h)
    Y  += H_h W2_h
```

これはGELUを要素ごとに適用するため、hidden tile間の依存がない。`b2`を一度だけ加え、
全hidden tileを合計すれば同じ式である。全`M x 4096`中間値を保持せず、
`row_tile x hidden_tile`だけをthread-localに持てる。

実装課題:

- `1024x4096`のfc1と`4096x1024`のfc2を独立GEMMとして呼ばない。
- NF24 decode、fc1 accumulation、GELU、fc2をsoftware pipeline化する。
- fc2の部分和をregisterまたはL1/L2に留め、tile完了ごとのstore/loadを減らす。
- 行tileは現在の16固定ではなく、8/16/24/32を新toolchainで測り直す。

### 4.2 QKVからSDPAまでlayout変換を融合する

現行attentionは、QKV Linear、reshape/unbind、RoPE、q/k RMSNorm、layout変換、SDPAの
順で複数tensorを作る。SDPA側はさらにK/Vを`[D, L]`へ全転置している。

次の順序へ変える。

1. QKV projectionの出力tileを作る。
2. Q/KだけにRMSNormとRoPEを適用する。
3. Qはquery block layoutへ、K/VはSDPAが読むpacked layoutへ直接storeする。
4. packed K/VをそのままSDPAへ渡す。

これによりQKV値そのものは同じまま、独立したreshape、transpose、packを除ける。
特にK/Vのcallごとの`posix_memalign`、全転置、`free`はpersistent workspaceと直接storeへ
置き換える。

### 4.3 SDPAをmicro-tile online softmaxへ変える

現行は8 query x 512 keyのscore配列をstackへ置く。key tileを16または32のmicro-tileへ
分け、scoreを指数化した直後にVへ掛ければ、大きいscore配列を保持する必要がない。

現在までの最大scoreを`m`、指数和を`l`、重み付きV和を`o`とする。新tile側を
`m_t, l_t, o_t`とすると、online softmaxは次式で厳密にmergeできる。

```text
m_new = max(m, m_t)
l_new = exp(m - m_new) * l + exp(m_t - m_new) * l_t
o_new = exp(m - m_new) * o + exp(m_t - m_new) * o_t
```

最後に`o / l`を出力する。この順序ならqueryごとの64次元accumulatorをregisterに近い
場所へ保ち、`scores[8][512]`と`local_out[8][64]`のstack往復を減らせる。

### 4.4 block境界を越えて融合する

個別kernelを速くしても、PyTorch tensor生成とctypes境界が各演算に残る。最終的には
少なくとも1 unified blockを1 native callへまとめる。

```text
LayerNorm -> modulation -> QKV/RoPE/RMSNorm -> SDPA
          -> output projection -> gate/residual
          -> LayerNorm -> modulation -> tiled MLP -> gate/residual
```

入力と最終出力以外をnative scratch arenaへ置き、PyTorchへ中間tensorを返さない。
これがrefのlocal patch in-place合成に対応するFlow側の変更である。

### 4.5 hot shape専用kernelを作る

Linear時間の約97.9%は次の4 shapeに集中する。

| shape | 時間 | Linear内比率 |
|---|---:|---:|
| 1024 x 4096 | 503.459秒 | 33.14% |
| 4096 x 1024 | 479.925秒 | 31.59% |
| 1024 x 3072 | 367.119秒 | 24.16% |
| 1024 x 1024 | 137.271秒 | 9.03% |

汎用ABIを保ちながら、この4 shapeだけloop order、row tile、prefetch距離、scale/bias
配置を固定したkernelへdispatchする。`cam_out_layer 1024x5`のscalar tailは完全性課題だが、
40 calls合計0.001618秒であり性能優先度は低い。

## 5. 結果再利用の精査

### 5.1 すでに実行時に有効な再利用

s20 manifestで次を確認済みである。

| 再利用 | 状態 | 同等性 |
|---|---|---|
| condition embedder + context refiner | 有効 | timestep/state非依存なので厳密 |
| fixed absolute position embedding | 19 hit / 1 miss | 厳密 |
| CFGで重複するlatent/camera/timestep prefix | 有効 | runner条件下で厳密 |
| 負条件の同一token圧縮 | 有効、同一性検査あり | `log(M)` biasで代数的に同等 |
| final blockのlatent/camera行だけ計算 | 有効経路あり | finalでcondition行を出力しないため厳密 |
| stage成果物のresume | 実装済み | 入力、設定、checksum一致時のみ厳密 |

これらを再提案しても新しい短縮にはならない。以後はcache hit/missとkeyをmanifestへ残し、
誤って異なるframeや設定の値を使わないことが課題になる。

### 5.2 新たに厳密再利用できるもの

| 対象 | 再利用範囲 | 条件 |
|---|---|---|
| thread-local Linear/SDPA/MLP workspace | 全20 step | shape、dtype、thread数が同じ |
| samplerの20個の時刻と係数 | run間、video frame間 | steps/guidance/shift/solverが同じ |
| timestep embeddingと各block modulation | run間、video frame間 | checkpointと時刻列が同じ |
| packed weight、scale、bias、W2 tile norm | 全入力 | checkpoint/backend formatが同じ |
| 負conditionのstatic context | 同じ負prompt/configの全frame | encoder/checkpoint/preprocessが同じ |
| 画像condition context | 同じ入力frameの再実験 | 入力hashとencoder設定が同じ |
| rendererの3D covariance、color、opacity | 同じGaussianを多視点描画 | Gaussian parameterが同じ |

timestep embeddingは各stepで同じ値を使い、videoのframeが変わってもscheduleが同じなら
変化しない。20時刻分を一度だけ生成し、checkpoint hash、schedule、dtypeをcache keyにする。
ただし、modulationがstateやconditionを入力に含む実装箇所はcache対象外とする。

### 5.3 厳密には再利用できないもの

| 対象 | 理由 |
|---|---|
| step間のQ/K/Vとattention出力 | Euler更新でlatent/cameraが変わる |
| 異なるvideo frameの正condition | encoder入力が異なる |
| decoderの分岐結果 | latentと乱数系列に依存する |
| 近接camera間のdepth sort | 投影深度順が入れ替わる可能性がある |
| 前stepのFlow velocity | stateとtimestepの両方が変わる |

Linearについて`X_next = X + delta`なら`W X_next = W X + W delta`だが、denseな`delta`に
対する`W delta`は元と同じ規模のGEMMである。差分再利用はdeltaが疎、低rank、または
量子化で多くのtileを0と判定できる場合だけ有効であり、通常は厳密な短縮にならない。

## 6. 若干の精度を使う候補

以下は追加の演算省略を行わない`strict_nf24`経路とは別の`bounded_approx` profileとして実装する。refの微小alpha
skipと同様に、単なる経験的thresholdではなく、可能な限り出力寄与の上限で判定する。

### 6.1 BF16 activationとdot product

第一候補はweightをさらに粗くすることではなく、MLP hidden、V、projection中間値をBF16で
保持し、AVX-512 BF16 dot productを使うことである。

- LayerNorm統計、RMSNorm統計、softmax、residual、sampler stateはFP32を維持する。
- Q/K scoreはまずFP32を維持し、VとMLP hiddenからBF16化する。
- 最初と最後のblock、camera path、final outputはFP32/NF24を維持する。
- block単位、演算種別単位でBF16を有効化し、s1から感度を測る。

対象CPUはAVX-512 BF16を持つ。現在のNF24 decode + FP32 FMAを、BF16へ変換したweightと
activationの`vdpbf16ps`系へ置き換えられれば、単なるassembler化より大きい短縮余地がある。

### 6.2 NF24/NF16のlayer別混在

全層NF16はs1 combined RMSE 4.50681e-4で旧s1 gate 2e-4に不合格、全層NF8は
2.51245e-2で不合格だった。ただし全層を同じbit数にする必要はない。

1. step/block/layerごとに最終latentとcameraへの感度を測る。
2. 低感度層だけNF16へ落とす。
3. camera、final、先頭/末尾block、高感度channelはNF24を維持する。
4. 速度と容量が改善しない層はNF16化しない。

精度を落とす目的は保存量だけでなく、VNNI/BF16 packed dotへ移行できる層を増やすことに
置く。復号命令が増えて速度が落ちる形式は採用しない。

### 6.3 MLP hidden tileの寄与上限によるskip

hidden tile`h`のactivationを`a_h`、fc2 weightを`W2_h`とすると、出力寄与は
`a_h W2_h`である。例えばFrobenius normを使えば次の上限が得られる。

```text
norm(a_h W2_h) <= norm(a_h) * norm(W2_h)
```

`norm(W2_h)`はcheckpoint変換時に保存できる。動的な`norm(a_h)`との積が、そのrowの
residual normに対する誤差予算以下ならfc2 tileをskipする。fc1を計算した後のskipでは
fc1時間は減らないため、最終的には低価格なactivation範囲予測またはBF16 fc1と組み合わせる。

### 6.4 attention tileの残存mass上限によるskip

現在までのsoftmax最大値を`m`、指数和を`l`、未処理tileのscore上限を`u_t`、key数を
`n_t`とすると、そのtileが追加し得る相対massは概ね次で上から抑えられる。

```text
r_t <= n_t * exp(u_t - m) / l
```

`r_t`がattention用誤差予算より小さいtileだけQK、softmax、PVを省く。`u_t`には
`q dot k <= norm(q) * norm(k)`とprecompute済みkey normを使えるが、上限が緩すぎる
可能性がある。その場合はkey tileのcentroid/radius境界を検討する。無根拠なtop-kや
token削減はこの経路へ入れない。

### 6.5 condition row更新を後段blockで減らす

final blockでcondition rowを出力しない最適化は厳密である。これを1つ前、2つ前へ広げると、
condition rowが後続blockのlatent attentionへ与える影響を近似することになる。

- まず末尾2 blockでcondition rowをfreezeする。
- 合格すれば末尾3、4 blockへ1段ずつ広げる。
- camera RMSEとback viewを優先監視する。
- 全step一括ではなく、step後半だけの適用も比較する。

既存実装にlate selective condition freezeの評価経路があるため、新規大改造より先に測る。

### 6.6 CFG負branchの時間方向再利用

負conditionは固定だがstateはstepごとに変わるため厳密cacheではない。負branchのvelocityを
2 stepごとにだけ計算し、中間を補間すればFlow forward回数を減らせる可能性がある。
ただしCFGは出力へ直接入るため、これは高リスク候補である。

```text
v_cfg = v_neg + guidance * (v_pos - v_neg)
```

負branch近似誤差もguidanceで増幅される。BF16、mixed NF16、late condition freezeより後に
評価し、`fast_approx`としてstrict結果と混同しない。

### 6.7 samplerのstep削減または適応化

20回のforwardを19回にするだけで、現在値から約170秒を直接削れる。microkernelの数%
より効果が大きいが、公式Euler 20 stepというアルゴリズム条件を変える。

- 20 timestampを保つstrict profileでは実施しない。
- velocity変化または局所誤差推定によりstepを省くadaptive profileを別に作る。
- 既存AB2実装を使う場合も、Euler基準と同一とは表記しない。
- 最終latentだけでなく6視点、camera、Gaussian parameter分布を比較する。

600秒を目指す場合はkernel最適化だけでは届かず、この種のmodel evaluation回数削減が必要に
なる。ただし1800秒までは、まずBF16/VNNI、fusion、bounded skipを優先する。

## 7. 精度予算と昇格gate

現在のNF24 s20 combined RMSEは9.38e-5で、既存上限5e-4の約19%を使っている。
残りを一度に使わず、候補ごとに加算的な誤差予算を割り当てる。

### 7.1 profile区分

| profile | 意味 | 必須条件 |
|---|---|---|
| `strict_nf24` | 対象演算と公式20 stepを維持 | 現行gate、fallback 0、NaN/Inf 0 |
| `bounded_approx` | 実装低精度化と寄与skipを許容 | s20 combined <=5e-4、camera <=1e-4 |
| `fast_approx` | CFG再利用、step削減、block省略を許容 | 別成果物、別manifest、strictと名称を分離 |

`bounded_approx`ではさらに6視点のworst-view PSNR低下を0.25 dB以内とし、正面平均だけで
合格させない。Gaussian/viewer比較まで完了しなければ採用しない。

### 7.2 誤差を使う順番

1. MLP hidden/VのBF16。
2. 低感度層だけNF16。
3. 末尾condition row freeze。
4. MLP寄与上限skip。
5. attention mass上限skip。
6. CFG負branch再利用。
7. adaptive samplerまたはstep削減。

state、Euler update、camera、final output、normalization統計、softmax reductionは最後まで
FP32を維持する。stepごとに蓄積する値へ早期に誤差を入れると、後段で増幅されやすいためである。

### 7.3 評価順

各候補は次の順で昇格する。

1. kernel microbenchmarkで時間、帯域、scratch量を測る。
2. s1でcombined RMSE <=2e-4、NaN/Inf 0を確認する。
3. s4で目標2e-4、暫定上限3e-4を確認する。
4. s20でcombined <=5e-4、camera <=1e-4を確認する。
5. 6視点rendererとviewerを比較する。
6. 3回測定の中央値を採用し、単発の最速値を使わない。

## 8. 実装優先順位

### P0 測定条件の正常化

- project配下へ新しいGCC/Clang/AOCC toolchainを置き、Zen 4 targetでbuildする。
- system環境へinstallせず、compiler pathとversionをmanifestへ記録する。
- affinityを固定し、1/2/3/4 threadを測る。追加CPUを確保できた時だけ6/8も測る。
- `perf`が使える環境ではcycles、instructions、IPC、LLC miss、memory bandwidthを採る。
- pack、QK、softmax、PV、projectionを個別timing化する。

終了条件は、旧GCC9 buildとの差を同一CPU、同一affinityで説明できることである。

### P1 ref型の厳密streaming/fusion

1. SDPA persistent workspaceとblocked K/V pack。
2. SDPA micro-tile online softmax。
3. 4 hot shape専用NF24 kernel。
4. tiled fc1 + GELU + fc2 MLP。
5. QKV/RoPE/RMSNorm/packed layout融合。
6. unified block native callとpersistent thread team。

最初の速度目標はNF24でstrict float32の3322.886秒を下回ること、次に3000秒未満とする。

### P2 厳密cacheの拡張

- timestep schedule、embedding、静的modulationをframe間cacheする。
- scratch arenaをshapeごとに再利用する。
- cache keyへcheckpoint、input、preprocess、steps、guidance、shift、dtypeを含める。
- hit/miss、再利用byte、節約時間をmanifestへ記録する。

### P3 bounded approximation

1. BF16 MLP hidden/V。
2. sensitivity-based NF24/NF16混在。
3. late condition freezeを2 blockから評価。
4. MLP tile寄与skip。
5. attention mass上限skip。

目標は既存品質gate内で3000秒を十分に下回り、1800秒へ近づけることである。

### P4 model evaluation回数の削減

- CFG負branchの間引き/補間。
- adaptive Euler、AB2、step再配置。
- gateの小さいblock/branchの予測skip。

これは公式20-step同等実装ではなく、明示的な`fast_approx`研究課題とする。

### P5 assembly

新compiler、PGO、intrinsic kernel、layout/fusionを完了した後、disassemblyとcounterで原因が
特定できた内側loopだけを独立`.S` microkernelにする。inline assemblerでcompilerの
register allocationを広く拘束しない。候補はNF24 decode/FMAまたはSDPA QK/PVの最内周で、
最低5%のend-to-end寄与が見込める場合だけ着手する。

## 9. 時間目標の現実性

現行でLinear + SDPAは3165.286秒、その他は約306.044秒である。両hot kernelを同率で
短縮した単純なAmdahl見積りは次になる。

| hot kernel短縮 | 推定s20 |
|---|---:|
| 10% | 約3155秒 |
| 15% | 約2997秒 |
| 20% | 約2838秒 |
| 30% | 約2522秒 |

1800秒には、その他を固定した場合でもhot kernelを約52.8%短縮する必要がある。
これはloop unrollやassemblerだけではなく、BF16/VNNI、演算融合、中間値削減を組み合わせる
規模である。600秒はmodel evaluation回数を変えずに達成する可能性が低い。

## 10. 今後の成果物と記録要件

各実験は次を残す。

- git commit、compiler、flags、CPU、affinity、thread数。
- kernel別call数、wall time、pack time、scratch peak、fallback数。
- cache key、hit/miss、再利用対象と再利用byte。
- s1/s4/s20 latent、camera、NaN/Inf。
- 6視点PSNR/MAEとviewer成果物。
- `strict_nf24`、`bounded_approx`、`fast_approx`の区分。
- 中断再開可能なmanifestとchecksum。

大きい成果物はrepositoryへ入れず、既定のGoogle Drive成果物領域へchecksum付きで移動し、
転送確認後にremote/localの重複を削除する。repositoryにはcode、small JSON summary、
日本語検証文書だけを残す。

## 11. 次の具体的ゴール

次の実装ゴールを、単なるassembler導入ではなく次の完了条件で設定する。

1. 新toolchainと正常化した4 CPU affinityでGEMM/SDPAのprofileを取り直す。
2. SDPAのallocationを0回/callにし、packと本体を分離測定する。
3. `scores[8][512]`をmicro-tile online softmaxへ置き換え、s4品質gateを通す。
4. 1024->4096->1024 MLPをstreaming fusionし、4096-wide全体tensorをなくす。
5. NF24 s20を3322.886秒未満、次に3000秒未満へ短縮する。
6. その後にBF16 MLP/Vを`bounded_approx`で評価し、1800秒目標の実現性を再判定する。

この順序なら、refの「必要な局所領域だけを生成し、その場で合成して捨てる」という着眼を
Flowへ適用しつつ、TripoSplatの機能と品質評価を混同せずに改善を進められる。

## 12. 2026-07-22進捗

P0とP1前半の実測は
[`triposplat_cpu_p0_p1_validation_20260722_ja.md`](triposplat_cpu_p0_p1_validation_20260722_ja.md)
へ記録した。

- project-local GCC13/Zen 4 buildを確立し、4 CPU affinityで再測定した。
- SDPA workspaceをpersistent化し、warmup後allocation 0、free 0とした。
- SDPA key tile 64/128/256は512より遅く、query block 4も8より遅いため不採用とした。
- NF24 row tile 24は遅く、tile 8はhot GEMMを20.6から42.2%短縮した。
- tile 8 s20は2747.270秒、combined RMSE 9.37666e-5、fallback 0で完走した。
- 3322.886秒未満と3000秒未満のP1速度目標を両方達成した。

次はP1後半のQKVからpacked K/Vへの直接store、streaming MLP、unified blockを行う。
その後P2のframe間cache監査を完了し、P3のBF16 MLP/Vを別profileで評価する。


## 13. 2026-07-22 P0-P5完了監査と継続課題

P0-P5の初回実装・棄却監査は次の文書へ分離した。

- P1後半: [`triposplat_p1_qkv_order_reuse_audit_20260722_ja.md`](triposplat_p1_qkv_order_reuse_audit_20260722_ja.md)
- P2: [`triposplat_p2_frame_cache_audit_20260722_ja.md`](triposplat_p2_frame_cache_audit_20260722_ja.md)
- P3: [`triposplat_p3_bounded_approx_audit_20260722_ja.md`](triposplat_p3_bounded_approx_audit_20260722_ja.md)
- P4: [`triposplat_p4_model_evaluation_reduction_audit_20260722_ja.md`](triposplat_p4_model_evaluation_reduction_audit_20260722_ja.md)
- P5: [`triposplat_p5_assembly_gate_audit_20260722_ja.md`](triposplat_p5_assembly_gate_audit_20260722_ja.md)

今回までに、QKV後処理とpacked SDPAの48/48 coverage、frame-local cache、scratch共有、BF16感度、
3 NFE solver、生成assemblyを実測した。現行runtimeへ新たに昇格する近似・assembly候補はない。
strict側ではGEMM `-funroll-loops`が有望だが、3回paired中央値前なので正式採用していない。

次の未達事項を具体的な継続ゴールとする。

1. NF24 GEMM unrollをshape別dispatchし、baseline/candidate各3回のpaired s1中央値を取る。
2. 1が安定して2%以上のwall短縮ならs4、s20へ昇格し、exact gateを再確認する。
3. NF16 patchへpartial coverageを実装し、block 12/15 MLPだけNF16、残りNF24のmixed prepackを作る。
4. BF16はroundtripではなく、block 12/15のfc1-GELU出力を保持したままfc2を`vdpbf16ps`で実行する。
5. negative-condition圧縮loopへlate condition freeze N=2を統合し、正枝・負枝を別々に感度測定する。
6. MLP tile寄与上限とattention残存mass上限をskipなしで記録し、有効tile比が低ければ実装を中止する。
7. 次回strict s20でstep stateと正負velocityを保存し、s19/adaptive候補をoffline replayする。
8. hardware counterが使えるhostでstall原因を特定し、Amdahl 5%を満たす場合だけstandalone `.S`を再検討する。

演算見直しの優先順は、`exactなlayout/fusion -> frame/cache再利用 -> 低感度層の低精度dot ->
寄与上限skip -> NFE削減 -> assembly`を維持する。前step QKV、state依存velocity、正conditionを
strict cacheする案や、無根拠なtop-k/一括低精度化は引き続き採用しない。


## 9. 2026-07-24以降の実行ゴールと登録タスク

次の最適化は、品質を変えない経路を最優先とし、長い検証へ進む前に短いgateで棄却する。
タスクは依存関係付きで登録し、前段が不合格の場合は後段を実行しない。

### G1: NF24 GEMM exact最適化

1. `EXACT-01`: 支配的な4形状を通常版とunroll版で7回交互測定する。
2. `EXACT-02`: 2%以上の局所短縮が再現した形状だけを選択kernelへdispatchする。
3. `EXACT-03`: s1をbaseline/candidate各3回交互実行し、hash一致とfallback 0を確認する。
4. `EXACT-04`: s4を各3回実行し、累積短縮と分散を確認する。
5. `EXACT-05`: s20で20秒以上または1%以上短縮し、出力完全一致の場合だけ正式採用する。

`EXACT-01`ではMLP `1024 -> 4096`が8.32%、`4096 -> 1024`が4.94%短縮した。
QKVとAttention outは1%前後だったため選択対象外とした。`EXACT-03`のs1中央値は
baseline 156.08秒、candidate 155.19秒で0.57%短縮、Linearは1.38%短縮、全runで
latent/camera hash一致、fallback 0だった。
`EXACT-04`のs4中央値はbaseline 572.56秒、candidate 563.27秒で9.29秒、1.62%短縮、
Linearは3.11%短縮した。全6 runでlatent/camera hash一致、fallback 0を維持したため、
同時刻baselineと比較するs20正式採用gateへ昇格した。

`EXACT-05`の同時刻s20比較はbaseline 2744.50秒、candidate 2731.50秒で、13.01秒、
0.47%の短縮だった。Linearは1067.24秒から1058.59秒へ8.66秒、0.81%短縮し、選択kernelは
2008回実行された。latent/camera hashは完全一致、fallback 0を維持したが、正式採用条件の
20秒または1%以上には届かなかった。そのため選択dispatchは検証済み候補として保持するが、
strict既定経路には昇格せず、次の支配項であるSDPA exact最適化へ進む。

### G2: SDPA exact最適化

1. `SDPA-01`: QK、online softmax、PVを実sequence長で個別計測する。
2. `SDPA-02`: 不採用済みkey tile拡大を繰り返さず、query tileとregister圧力を探索する。
3. `SDPA-03`: 局所3%以上かつbit一致した候補だけs1反復へ進める。
4. `SDPA-04`: s20で1%以上短縮した場合だけ採用する。

### G3: 動画frame間の厳密再利用

1. `VIDEO-01`: model、packed weight、固定条件、scratchを保持する複数frame runnerを作る。
2. `VIDEO-02`: 異なる2 frameでframe依存stateが混入せず、2 frame目のworkspace再確保0を確認する。
3. `VIDEO-03`: 20 frameで定常時の秒/frameとmemory plateauを測る。

### G4: bounded approximation

1. `APPROX-01`: 感度が低かったblock 12と15だけに実BF16 dot kernelを実装する。
2. `APPROX-02`: s1 combined RMSE 2e-4以下かつ1%以上短縮を要求する。
3. `APPROX-03`: s4でcameraと6視点renderを確認する。
4. `APPROX-04`: 合格時だけs20の任意profileとして測り、strict既定値は置換しない。

### G5: model再実行を減らすoffline探索

1. `REPLAY-01`: strict s20でstep stateと正負velocityを一度だけ保存する。
2. `REPLAY-02`: 19個のs19省略位置を追加model runなしでoffline比較する。
3. `REPLAY-03`: offline gate通過候補だけを実推論とrenderで検証する。

優先順はG1、G2、G3、G4、G5とする。近似候補はstrict結果と混ぜず、exact候補が
正式gateを通過しない限りs20へ無条件昇格させない。
