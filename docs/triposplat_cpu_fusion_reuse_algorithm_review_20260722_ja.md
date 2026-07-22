# TripoSplat CPU 演算順序・再利用・精度許容の再精査

日付: 2026-07-22

## 1. 目的

本書は、公式TripoSplatの機能や推論結果を後処理で補正するための文書ではない。
公式Flow modelの演算をCPU低リソース向けに並べ替え、中間結果と作業領域を再利用し、
同じ結果を短時間・低peak memoryで得るための実測追補である。

初期に参照した別実装`ref`のGaussian式をFlow推論へ移植しない。参考にするのは、
`gs_minimal.py`、`gs_colmap_pipeline.py`系で確認した次の実行上の着眼である。

- 全画面・全要素の中間結果を一度に作らず、必要な局所領域だけ作る。
- 局所結果をその場で次段へ渡し、可能ならin-placeで合成する。
- 固定値、stage成果物、作業領域を再利用する。
- 早期skipは、結果へ影響しない場合と許容誤差を明示した場合だけ使う。
- 時間、容量、品質をmanifestへ残し、見た目だけで採否を決めない。

## 2. 今回の結論

今回、ref型の局所化は2つの異なる結果になった。

1. MLPを小チャンクへ分ける案は不採用となった。中間hiddenの局所化より、W1/W2の再読込と
   OpenMP同期の増加が大きかった。
2. QKV後処理をRoPE、RMSNorm、layout変換、K/V packまで一続きにする案は採用候補となった。
   大きな中間tensorの往復を除き、出力を完全一致させたままs1 forwardを5.96%短縮した。

したがって「局所化すれば速い」ではなく、次の条件を同時に満たす境界だけを融合する。

- 次段が直ちに同じ値を消費する。
- 融合しても大きなweightを何度も読み直さない。
- 中間値を削除したbyte数が、同期・分岐・再計算コストを上回る。
- 共有workspaceを層ごとに複製せず、逐次実行する層間で再利用できる。

## 3. NF24 MLP streamingの実測と不採用理由

対象は主ブロックの`1024 -> 4096 -> 1024` MLPである。

```text
Y = NF24_GEMM_2(GELU(NF24_GEMM_1(X)))
```

### 3.1 8行完全融合

8行についてW1復号、fc1、GELU、W2復号、fc2を完了してから次の8行へ進めた。
出力は基準とbit完全一致したが、実hot shape 12,294行では次の結果だった。

| 経路 | 中央値 |
|---|---:|
| 独立NF24 GEMM + GELU + NF24 GEMM | 0.796746秒 |
| 8行完全融合 | 1.060175秒 |
| 候補 / 基準 | 1.3306 |

33.1%遅くなった。8行ごとにW1とW2を切り替えるため、約12MBの量子化weightを
交互に再走査し、weight streaming localityを失ったことが主因である。

### 3.2 共有hiddenのチャンク掃引

fc1をチャンク全体へ実行し、GELU後にfc2を同じチャンクへ実行した。
7反復の実形状結果は次のとおりである。

| chunk rows | hidden workspace | 中央値 | 基準比 |
|---:|---:|---:|---:|
| 基準 | 約201MB | 0.753158秒 | 1.0000 |
| 1024 | 16MB | 0.815498秒 | 1.0828 |
| 2048 | 32MB | 0.789360秒 | 1.0481 |
| 4096 | 64MB | 0.794383秒 | 1.0547 |
| 12294 | 約201MB | 0.752986秒 | 0.9998 |

全候補でRMSE 0、最大絶対差0だった。全行workspaceを使った場合だけ基準と同速であり、
低メモリ化と短縮を両立しなかったため採用しない。

### 3.3 MLPから得た判断規則

- W1とW2の両weightがLLCへ同時常駐しない場合、細かいproducer-consumer融合を避ける。
- hidden materializeの削減byteだけで判断せず、weight再走査byteを加える。
- OpenMP parallel regionの統合だけでは、GEMM本体が支配する形状で大きな利益にならない。
- 次のMLP候補は、BF16/VNNIなどGEMM本体の演算形式を変える場合だけ再開する。

## 4. QKVからpacked SDPAまでの厳密融合

公式順序と現CPU基準は概ね次のとおりである。

```text
QKV Linear
-> q/k/v view
-> Q/K contiguous copy
-> Q/K RoPE
-> Q/K RMSNorm
-> Q/K/V [B,H,L,D] contiguous copy
-> SDPA内でK/Vを[B,H,D,Lpad]へpack
-> SDPA [B,H,L,D]
-> [B,L,H,D]へ再配置
-> output Linear
```

注意: 公式実装の順番は`RoPE -> RMSNorm`である。以前の計画文書にある
`RMSNorm -> RoPE`という記述は実装順と異なるため、実装では採用していない。

新経路はQKV Linearの出力を読み、16 token tileごとに次を行う。

```text
Q,K: RoPE -> RMSNorm
Q:   [B,H,L,D]へ直接store
K,V: [B,H,D,Lpad]へ直接store
SDPA: packed K/Vを直接読む
SDPA output: [B,L,H,D]へ直接store
```

RoPE、RMSNorm、online softmax、指数近似、積和順序は現基準と同じである。
変えたのはlayout materializationの順序だけである。

### 4.1 後処理単体

`B=2, L=6147, H=16, D=64`で7反復した。

| 経路 | 中央値 | 基準比 | Q/K/V誤差 |
|---|---:|---:|---:|
| 分離RoPE/RMSNorm/layout/pack | 0.143884秒 | 1.0000 | 基準 |
| 16-token融合direct pack | 0.018199秒 | 0.1265 | RMSE/max 0 |

後処理部分は87.4%短縮した。Qは約50.4MB、packed K/Vは各約50.5MBである。

### 4.2 packed SDPA単体

同じonline softmaxをpacked K/V入力、BLHD直接出力へ変更した。

| 経路 | 中央値 | 基準比 | 出力誤差 |
|---|---:|---:|---:|
| 標準K/V入力 + BLHD materialize | 1.312810秒 | 1.0000 | 基準 |
| packed K/V + BLHD直接出力 | 1.306186秒 | 0.9950 | RMSE/max 0 |

SDPA本体は計算支配なので単体速度はほぼ同じである。利益は前段のlayout往復削除と、
SDPA後の約50MB再配置削除にある。

### 4.3 実モデルs1

negative-condition圧縮の内部helperへ接続し、正枝23回、負枝24回、合計47回をpacked化した。
最終正枝のselected-query 1回は既存経路のままである。

直後に同じCPU割当で基準を再走したpaired比較:

| 指標 | 基準 | packed v2 | 短縮 |
|---|---:|---:|---:|
| forward | 130.853830秒 | 123.057837秒 | 5.96% |
| Attention累積 | 88.114446秒 | 81.164138秒 | 7.89% |
| QKV Linear累積 | 9.740565秒 | 9.693761秒 | 0.48% |
| MLP累積 | 35.220836秒 | 34.586372秒 | 1.80% |
| run elapsed | 150.092156秒 | 139.791629秒 | 6.86% |

latent/cameraは基準に対してRMSE 0、最大絶対差0、NaN/Inf 0だった。

### 4.4 実モデルs4

GDriveに保存済みの同条件tile8基準を再利用した。これは同時刻paired測定ではないため、
s4の速度比は参考値とし、正式な速度採用値はs1 paired比較を使う。

| 指標 | 保存済み基準 | packed v2 | 差 |
|---|---:|---:|---:|
| forward合計 | 583.994613秒 | 506.406757秒 | -13.29% |
| run elapsed | 605.281621秒 | 524.281562秒 | -13.38% |

packed呼出し188回、key-bias呼出し96回、fallback 0、workspace allocation 4回、
最大共有workspace 201,506,816 byteだった。4 step後もlatent/cameraはRMSE 0、
最大絶対差0、NaN/Inf 0である。

## 5. 結果と作業領域の再利用

### 5.1 今回実装した再利用

- Q、packed K、packed V、BLHD出力の4 bufferを全24層で共有する。
- 最大shapeへ4回だけ拡張し、その後のs1 47回、s4 188回で再利用する。
- 層ごとに約201MBを保持しない。層ごとなら約4.8GBとなるため禁止する。
- QKV、RoPE、RMSNorm、K/V packはstate依存なのでstep間cacheせず、その場で消費する。

### 5.2 次に厳密再利用する対象

| 対象 | 再利用範囲 | 条件 |
|---|---|---|
| modelとpacked weight | 全frame | 動画を1 processで逐次処理する |
| 負condition context | 全frame | 負条件入力と前処理が同じ |
| timestep schedule/embedding | 全frame | steps、shift、dtype、checkpointが同じ |
| position embedding | 全frame | canvas、token配置、dtypeが同じ |
| decoderの固定index/random table | 全frame | seed、Gaussian数、decoder設定が同じ |
| scratch arena | 全step・全frame | backend formatと最大shapeが同じ |

cache keyにはcheckpoint hash、backend ABI、canvas、steps、guidance、shift、dtype、seed、
preprocess hashを含める。cache hit/miss、再利用byte、生成時間をmanifestへ残す。

### 5.3 再利用しない対象

- 正condition contextは入力frameごとに変わる。
- latent state、Q/K/V、RePo rotary embedding、velocityはstepごとに変わる。
- 前step QKVの無条件再利用はstrict経路では行わない。
- camera推論結果はlatentとattention stateに依存するため、別frameへ流用しない。

## 6. 近似を許容する別profile

以下は`strict_nf24`へ混ぜず、`bounded_approx`として別manifest・別成果物にする。

### 6.1 K/V BF16 storage

Q、softmax最大値、指数和、最終accumulatorはFP32を維持し、packed K/VだけBF16へする。
K/V workspaceを約101MBから約50MBへ減らし、帯域も半減できる。候補はVだけ、Kだけ、K/Vの
順でs1感度を測る。gateはs1 combined RMSE 5e-5、s4 2e-4、camera 5e-5とする。

### 6.2 RoPE normの順序利用

実数上、RoPEは直交回転なので`||RoPE(q)||_2 = ||q||_2`である。RMSNormのnormだけを
回転前Q/Kから計算すれば、中間storeをさらに減らせる可能性がある。ただしFP32の加算順が変わり、
bit一致しない可能性があるためstrictには入れない。gamma乗算とRoPE自体の順番は入れ替えない。

### 6.3 NF24/NF16層別混在

先頭/末尾block、camera、final出力をNF24に保ち、s1 layer sensitivityが低いMLP/Attention Linearだけ
NF16またはBF16 dotへ移す。目的は保存量だけでなく、復号+FP32 FMAをVNNI/BF16 packed dotへ
置換することである。全層一括量子化は行わない。

### 6.4 時間方向の再利用

負枝velocityの前step値をそのまま使う、blockをskipする、adaptive stepへ変える案は効果が大きいが、
公式20-stepの演算条件を変える。velocity差、局所誤差、最終latent/camera誤差を記録する
`fast_approx`研究課題とし、strict結果として扱わない。

## 7. 次の具体的ゴール

### P1 strict fusionの完了

1. 最終正枝selected-query経路へpacked K/V APIを追加し、48/48 Attentionを覆う。
2. NF24 QKV GEMMのepilogueから16-token/head scratchへ直接渡し、約151MBのQKV全体出力を除く。
3. packed bufferの全領域`memset`をやめ、末尾padding最大15 tokenだけを0にする。
4. prepacked checkpoint loaderと正式なstrict runnerへv2経路を統合する。
5. s20 paired gateを完走し、現2747.270秒から2600秒前後を第一目標とする。
6. `/usr/bin/time -v`またはcgroup peakでruntime-pack/prepacked双方の最大RSSを記録する。

### P2 frame間cache

1. 動画frame loopをmodel processの内側へ移し、model/weight/workspaceを保持する。
2. 負condition、schedule、timestep/position embeddingのcache keyとhit/missを実装する。
3. 2 frame連続実行で、2 frame目のload/pack/cache時間とpeak RSSを測る。
4. frameごとの正conditionとlatentが誤って再利用されないnegative testを追加する。

### P3 bounded approximation

1. V BF16、K BF16、K/V BF16を別々にs1評価する。
2. layer別NF16/BF16 sensitivity表を作る。
3. s1を通過した候補だけs4、s20へ進める。
4. strict、bounded_approx、fast_approxの成果物名とmanifestを分離する。

### P4/P5

1. latent/camera gate通過後にPLY属性、公式視点render、viewerを比較する。
2. C intrinsicsでhot loopが明確に残る場合だけcompiler assemblyを監査する。
3. inline assemblyは、同じ入力でintrinsicsより5%以上速く、GCC13/Zen4限定fallbackを持つ場合だけ採用する。

## 8. 現時点の判断

CPU最適化余地はまだある。inline assemblyへ進む前に、QKV全体materialize、最終selected-query、
frameごとのmodel/weight再loadという大きいデータ移動を除く方が優先である。

今回の成功例は、refの「必要な局所値を作り、その場で次段へ渡す」という着眼を、公式TripoSplatの
演算順序を保ったままQKV/Attention境界へ適用したものである。失敗例のMLPは、局所化により
weight再走査が増える場合は逆効果になることを示した。今後も省略byte、再読込byte、同期回数、
品質差を同時に測って採否を決める。
