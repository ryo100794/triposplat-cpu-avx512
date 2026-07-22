# TripoSplat P2 frame間cache監査

日付: 2026-07-22

## 1. 目的

動画をframeごとに別processで処理せず、1 process内でmodel、NF24 packed weight、workspace、
固定条件、schedule由来の値を再利用できるかを厳密経路で確認する。
異なるframeの正condition、latent、Q/K/Vを再利用することは目的に含めない。

## 2. 実装

既存の次の再利用を同一processに保持した。

- modelとNF24 packed weight
- 正/負conditionのcontext-refiner出力
- fixed position embedding
- packed AttentionのQ、K、V、output workspace
- negative-condition用key biasとselected-row index

さらに、scheduleが同じframe間で同じ値になるtimestep embeddingとAdaLN modulationへ、
64-entry LRU cacheを追加した。

cache key:

```text
timestep: 値列、shape、dtype、device
AdaLN:    cached timestep tensorのidentity、version、shape、dtype、device
```

通常tensorは出力versionが変わった場合にinvalidated missとする。PyTorch inference tensorは
version counterを持たないため、inference中にcached出力を変更しないことを契約にする。
state、condition、RePo、Q/K/Vはこのcacheへ入れない。

## 3. self-test

小型moduleで次を確認した。

- 同じtimestep値を別tensorで渡してもhitする。
- 異なるtimestep値はmissする。
- 通常tensorのcached出力をin-place変更するとinvalidated missになる。
- clear後のentry数は0になる。

## 4. 実モデル2 frame probe

同じprepared frame、noise、conditionを同一process内でs1として2回処理した。
これは速度benchmarkだけでなく、process-local cacheの境界を証明するprobeである。

| 指標 | frame 1 | frame 2 |
|---|---:|---:|
| sampler elapsed | 125.837337秒 | 123.108237秒 |
| first比 | 1.0000 | 0.9783 |
| latent/camera | 基準hash | frame 1と完全一致 |

2回目は2.729秒、2.17%短かった。ただしtimestep/AdaLNの実計算削減は2.3msだけであり、
残りはwarm page/cache、allocator、cloud実行変動を含む。したがって2.17%をtimestep cacheの
効果とは扱わない。

cache統計:

| 対象 | hit | miss | 再利用量 |
|---|---:|---:|---:|
| fixed position | 1 | 1 | manifest未集計 |
| timestep embedding | 1 | 1 | 4,096 byte |
| AdaLN modulation | 1 | 1 | 24,576 byte |
| 合計 | 2 | 2 | 28,672 byte |

packed Attentionは96回、selected-query 2回、key-bias 48回、fallback 0だった。
workspace allocationは2 frame合計でも4回、capacityは201,506,816 byteであり、2 frame目の
再確保はなかった。正/負condition contextの構築はsample loop外で各1回だけだった。

最終latent/camera SHA-256はP1 strict基準と一致した。

## 5. 実動画での再利用境界

frame間で再利用する:

- model、packed weight、native backend
- workspaceとallocator arena
- steps/shift/solverが同じ場合の時刻列、timestep embedding、AdaLN modulation
- canvas/token配置が同じ場合のposition embedding
- 同じnegative conditionのcontext
- decoder/rendererのcheckpoint依存固定table

frameごとに再計算する:

- 画像encoderが出す正condition
- 正conditionのcontext-refiner出力
- initial noiseをframeで変える設定の場合のnoise
- latent、camera、RePo、Q/K/V、attention、MLP、velocity

同じframeを複数parameterで再評価する場合だけ、入力hashとencoder設定が一致する正condition
contextを再利用できる。

## 6. cache keyとnegative test

永続cacheへ進める場合のkeyは次を含める。

- checkpoint SHA-256
- native backend ABI/versionとpacked weight format
- inputまたはnegative conditionのcontent SHA-256
- encoder/checkpoint、preprocess、canvas、dtype
- steps、shift、solver、timestep scale
- token配置とcamera有無

必須negative test:

1. 正conditionを1要素変更するとmissする。
2. stepsまたはshiftを変更するとschedule/timestep cacheがmissする。
3. dtype、backend ABI、checkpoint hashを変えると全関連cacheがmissする。
4. frame 2のlatent/Q/K/Vがframe 1と同じpointerまたは内容として再利用されていない。
5. cache entry上限を越えた場合、LRU eviction後も出力が基準と一致する。

現実装の正condition cacheは、呼び出し側がframeごとに新しいcondition dictを作る前提である。
異なるfeatureを同じdictへ上書きして内部cached keyだけ残す操作は禁止する。将来の動画runnerでは
content hashを照合してからcached contextを付与する。

## 7. 判断

process保持とworkspace再利用は実装上成立し、2 frame目に新規workspace allocationはなかった。
timestep/AdaLN cacheも厳密だが、s1で削減するのは約2.3msなので単独の速度効果は小さい。
P2の主要利益は、frameごとのmodel load、NF24 pack、negative context構築、200MB級workspace確保を
繰り返さないことにある。次の動画runnerはframe loopをmodel processの内側へ置き、正conditionだけを
frameごとに入れ替える構造にする。
