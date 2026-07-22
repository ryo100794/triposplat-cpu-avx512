# TripoSplat P1 QKV演算順・再利用監査

日付: 2026-07-22

## 1. 目的

`ref`で有効だった「全体中間を作らず、局所値を生成直後に消費する」という着眼を、
TripoSplatのNF24 QKV projectionからpacked SDPAまでへ適用した結果を記録する。
推論式、RoPE、RMSNorm、softmax、Euler更新は変更しない。

## 2. 採用した演算順

従来の後処理は、QKV全体を作った後にQ/K/Vを分離し、RoPE、RMSNorm、layout変換、
K/V packを個別に実行していた。採用候補v2は次の順序にした。

```text
QKV projection
-> 16-token tileを読む
-> Q/K: RoPE -> RMSNorm
-> Q: BHLDへ直接store
-> K/V: BHDL-paddedへ直接store
-> packed SDPA
-> BLHDへ直接store
-> output projection
```

packed K/Vの全領域を毎回0 clearせず、実際にpaddingとなる末尾最大15 tokenだけを0にする。
Q、packed K、packed V、SDPA出力は最大shapeの4 bufferを全blockで共有する。

## 3. v2後処理の実測

実形状`B=1, L=12294, Lq=8193, H=16, D=64`、4 thread、7反復中央値で測定した。

| 経路 | 中央値 | v1比 | Q/K/V |
|---|---:|---:|---:|
| v1通常後処理 | 18.333 ms | 1.0000 | 基準 |
| v2通常後処理 | 13.155 ms | 0.7176 | RMSE/max 0 |
| v2選択行後処理 | 13.144 ms | 0.7170対v1全行 | RMSE/max 0 |

`L=31, Lq=17`でもQ/K/Vは完全一致し、K/V padding末尾は通常・選択行とも0だった。

実モデルs1では、正枝23回、負枝24回、最終正枝selected-query 1回の合計48回を覆った。

- packed呼出し: 48
- selected-query呼出し: 1
- key-bias呼出し: 24
- fallback: 0
- 共有workspace: 201,506,816 byte
- latent/camera SHA-256: 47回版基準と一致

後処理累積は`0.900 s -> 0.586 s`、34.9%短縮した。一方、s1 wall timeは
`139.792 s -> 140.059 s`であり、SDPAの実行時間変動を含む全体速度差は立証できなかった。
したがってv2は厳密同値と局所短縮が成立した候補だが、wall time短縮を単独の採用理由にしない。

## 4. QKV中間を除く2方式

QKV projectionの151,068,672 byte中間tensorも除くため、同じNF24 FMA順序で2方式を評価した。

### 4.1 head-order 6KB tile

各8行について1 head分のQ/K/Vを計算し、直後に後処理した。中間は約6KBまで減ったが、
Q、K、Vの離れたweight領域をheadごとに往復した。

| 経路 | 中央値 | 基準比 | Q/K/V |
|---|---:|---:|---:|
| GEMM + v2後処理 | 231.255 ms | 1.0000 | 基準 |
| head-order直接store | 303.981 ms | 1.3145 | RMSE/max 0 |

31.4%遅いため不採用とする。ref型の局所化でも、大きなweightの走査順を壊す場合は逆効果である。

### 4.2 output-order 96KB cache tile

元GEMMと同じ出力列順で8行 x 3072列を計算し、96KBのthread-local tileをL2内で後処理した。
DRAM上の全QKV tensorは作らないが、weight streamとFMA順を維持する。

| 経路 | 中央値 | 基準比 | Q/K/V | 除去中間 |
|---|---:|---:|---:|---:|
| GEMM + v2後処理 | 229.090 ms | 1.0000 | 基準 | 0 |
| 96KB cache tile | 231.384 ms | 1.0100 | RMSE/max 0 | 151,068,672 byte |

速度は1.0%悪化したが、151MBのpeak中間を除ける。標準の速度profileには入れず、
peak RSSを下げる`strict_low_memory`候補として保持する。採用には実モデルpeak RSS測定が必要である。

## 5. 再利用の判断

厳密に再利用するもの:

- Q、packed K/V、SDPA出力の共有workspace
- packed weight、scale、bias
- 同じscheduleの時刻列とtimestep embedding
- 同じcanvas/token配置のposition embedding
- 同じnegative conditionのcontext
- 動画frame間でのmodel processとscratch arena

再利用しないもの:

- step間のQ/K/V、attention出力、latent、camera
- 異なるframeの正condition
- state依存のRePo frequency
- 前step velocityのstrict profileでの流用

cache keyはcheckpoint、backend ABI、canvas、steps、guidance、shift、dtype、negative condition、
preprocess hashを含め、hit/miss、生成時間、再利用byteをmanifestへ記録する。

## 6. 精度を使う次段候補

以下は`strict_nf24`へ混ぜず、別profileとする。

1. `bounded_approx`: packed VだけBF16、次にKだけBF16、最後にK/V BF16を評価する。
2. `bounded_approx`: MLP hiddenをBF16で保持し、AVX-512 BF16 dotへ置換する。
3. `bounded_approx`: RoPE前後でL2 normが不変な性質を使い、normだけを回転前に計算する。
4. `bounded_approx`: 感度の低いblock/layerだけNF16へ落とし、camera、先頭、末尾をNF24に保つ。
5. `fast_approx`: attention残存mass上限、MLP tile寄与上限、late condition freezeでskipする。
6. `fast_approx`: CFG枝の補間、adaptive Euler/AB2でmodel evaluation回数を減らす。

各候補は省略演算数だけでなく、latent、camera、PLY属性、公式視点renderを順にgateする。

## 7. P1の残課題

- 96KB cache tile版の実モデルpeak RSSを測り、速度profileと低メモリprofileを分ける。
- Python tensorとctypes境界を減らすunified block ABIを作る。ただしMLP weight再走査は増やさない。
- prepacked loader、正式runner、backend ABI/versionを一本化する。
- s20で厳密同値、fallback 0、peak RSS、wall timeを記録する。
- compiler出力、PGO、prefetch、layoutを評価した後にだけinline assemblyを検討する。

今回の主要な判断規則は、削除した中間byteだけでなく、追加したweight再読込byte、
weight走査順、OpenMP同期、cache tileの所属階層を同時に評価することである。
