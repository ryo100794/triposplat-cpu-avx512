# TripoSplat CPU P4 model evaluation回数削減監査

日付: 2026-07-22

## 1. 区分

model forward回数、公式timestamp、Euler更新式を変える候補は、CPU kernelの同等置換ではない。
すべて`fast_approx`として`strict_nf24`、`bounded_approx`と別成果物にする。

## 2. AB2実装の役割

既存AB2は、最初のstepをEulerで更新し、2 step目以降を可変刻みAB2で更新する。

```text
r = dt_i / dt_(i-1)
f_eff = (1 + r/2) f_i - (r/2) f_(i-1)
x_(i+1) = x_i + dt_i f_eff
```

過去velocityを再利用するため、更新式自体に追加model forwardは要らない。ただし同じstep数で
実行すればNFEは同じであり、AB2を選ぶだけでは時間はほぼ減らない。短縮にはstep数も減らす
必要がある。

## 3. 3 NFE probe

exact Euler 4-step結果を参照し、同じnoise、condition、NF24 packed-v3 backendでEuler 3-stepと
AB2 3-stepを比較した。これは低コストの棄却probeであり、公式20-step品質の代替評価ではない。

| 候補 | elapsed | forward合計 | combined RMSE | latent RMSE | camera RMSE |
|---|---:|---:|---:|---:|---:|
| Euler 3 | 384.400秒 | 366.492秒 | 3.99290e-1 | 3.37959e-1 | 4.52381e-1 |
| AB2 3 | 380.436秒 | 363.793秒 | 5.92897e-1 | 5.58426e-1 | 6.25471e-1 |
| Euler 4参照 | 524.282秒 | 506.407秒 | 0 | 0 | 0 |

3 NFEは4 NFE参照より約27%短いが、誤差は品質gateから3桁以上離れた。AB2はこの粗いscheduleで
Euler 3-stepより悪かった。単純にstepを1つ減らす案と、同じtimestampへAB2を適用する案は
不採用とする。

## 4. なぜ単純step削減が破綻するか

step数を変えると1回だけforwardを省くのではなく、全timestampと各`dt`が変わる。各stepのstateが
変わり、その後の全velocity評価へ差が伝播する。特に少数stepでは1区間が大きく、過去velocityを
線形外挿するAB2の局所誤差も大きい。したがって、s3対s4の結果からs19対s20の誤差を比例推定しては
ならない。一方、このprobeは「AB2なら粗いstepでも自動的に補える」という仮説を棄却する。

## 5. 次に評価できるadaptive方式

次の候補はtimestampを一律に引き直さず、公式20点を基準に区間ごとの省略可否を判断する。

1. まずstrict s20で各stepの`norm(f_i - f_(i-1))`、state更新norm、camera更新normを保存する。
2. AB2とEulerの更新差を追加NFEなしの局所誤差指標として使う。
3. 指標が小さい連続区間だけを統合し、前半・後半を別thresholdにする。
4. 1区間だけ省くs19候補から開始し、省略位置を全19候補でoffline replayする。
5. s20 combined `5e-4`、camera `1e-4`、6視点worst PSNR低下0.25 dBを通過した位置だけ採用する。

offline replayには各公式stepの入力stateとvelocityが必要である。次回のs20 strict実行時に
step checkpointを同時保存し、候補ごとに高価なmodelを再実行しない。これはrefのstage成果物再利用
という着眼を、solver探索へ適用する方法である。

## 6. CFG負枝の時間再利用

負condition contextは固定でも、負枝velocityはstateとtimestepに依存するためstrict cacheできない。

```text
v_cfg = v_neg + g * (v_pos - v_neg)
```

guidance `g=3`では負枝誤差の係数は`1-g=-2`となり、再利用誤差が増幅される。次の順で評価する。

1. strict s20で正枝・負枝velocityをstep別に保存する。
2. `v_neg`のstep差と最終出力に対するoffline置換誤差を測る。
3. 前値保持、線形補間、低rank delta予測を比較する。
4. 省略した負枝1回あたり約半forwardの短縮が得られるか実測する。
5. adaptive stepより厳しいcamera/6視点gateを通す。

現行runnerには負枝velocityの記録・再投入APIがないため未実装である。推測だけで前step値を使う
実装は行わない。

## 7. block/branch予測skip

residual blockの出力normが小さくても、後段attentionで増幅される可能性がある。blockを丸ごと
skipする前に、各block residualと最終latent/cameraの感度をstep別に測る。P3のMLP感度では
block 18-23が高感度だったため、末尾blockからの無条件skipは優先しない。

## 8. P4判定

- AB2 3-step: 品質不合格。
- Euler 3-step: 品質不合格。
- CFG負枝再利用: state依存のためstrict不可、記録/replay API未実装。
- adaptive s19: s20 step checkpointを使うoffline探索を先に実装する。
- 現行runtimeへの`fast_approx`採用: なし。

raw成果物はGDriveへ移し、8ファイルのchecksum差分0を確認後、共有先から削除した。
