# TripoSplat CPU 高メモリ速度モード監査 2026-07-23

## 1. 目的と結論

低リソース既定の全206 Linear NF24 int16を維持しつつ、メモリ上限を緩めた場合に
CPU速度を上げられる箇所を調べた。結論は次の通りである。

- 28個のself-Attentionにある`qkv: 1024 -> 3072`と`out: 1024 -> 1024`、計56 Linearだけを、
  NF24の量子化値からFP32 `[K,N]`へ一度展開して常駐させる。
- 展開後もPyTorch/MKLへは渡さず、既存のnative FP32 AVX-512/FMA kernelを使う。
- MLP `fc1`と`fc2`を含む残り150 Linearは、従来どおりNF24をGEMM内で復号する。
- packed SDPAのquery tile 8/key tile 512は維持する。key tile 1024以上は遅かった。
- s1出力は現行NF24 packed-v3とlatent/cameraともbit完全一致、fallback 0だった。

この構成はopt-inの高メモリ速度候補として実装した。共有CPUのrun間変動が大きいため、
既定の低リソースs20を置換する正式採用は、s4/s20の複数回paired測定後とする。

## 2. なぜメモリを速度へ交換できるか

現行NF24 kernelは、重みを24 bitで保持し、GEMMのたびに次を実行する。

```text
q = 256 * int32(code01) + int32(q2)
w = float32(q) * scale
y += x * w
```

高メモリモードでは`w`を起動時に一度だけ作り、各stepでは`y += x * w`だけを実行する。
モデルの量子化値は変わらない。変わるのは重みの保存形式とkernel内の復号有無だけである。

選択した56 Linearのweight数は117,440,512個である。

| 保存形式 | weight payload |
|---|---:|
| NF24 | 352,321,536 byte |
| 展開FP32 | 469,762,048 byte |
| 常駐差分 | +117,440,512 byte（約112 MiB） |

runtime量子化中は元weight、全NF24 code、選択FP32 weightが重なるため、観測peak RSSの増加は
常駐差分より大きい。実測候補は3.41から3.96 GiB、同時刻基準は2.95から3.07 GiBだった。
事前pack直接loadを使えば、公式weightを最初に全展開するpeakを避けられる。

## 3. 実装

`native_linear_nf24_prepacked.py`へ次を追加した。

1. `decode_nf24_i16_weight_t()`がNF24 codeをnative GEMMの`[in,out]`へ直接展開する。
2. `materialize_nf24_i16_linears()`がregexで選んだ層だけFP32化し、元packed bufferを解放する。
3. FP32 `weight_t`を一度も転置コピーせず、`gemm_f32_avx512`へ同じstorageで渡す。
4. selected-row Attention用のoutput-range APIを維持する。

専用runnerは`run_triposplat_nf24_materialized_packed_v3_param_batch.py`である。事前pack済み
checkpointとruntime packの両方を受け付ける。再現用入口は次である。

```bash
STEPS=20 \
TRIPOSPLAT_RNF8_PREPACKED_DIR=/path/to/triposplat_nf24_i16_v1 \
REFERENCE_NPZ=/path/to/reference_s20/base_latent.npz \
bash scripts/run_nf24_materialized_speed_probe.sh
```

## 4. GEMM同一run比較

AMD EPYC 9654、4 CPU、GCC13 Zen 4 build、`M=12294`で交互測定した。

| shape `(M,K,N)` | NF24 | native FP32 | FP32/NF24 | 判定 |
|---|---:|---:|---:|---|
| 12294,1024,4096（fc1） | 0.371208秒 | 0.380528秒 | 1.0251 | NF24維持 |
| 12294,4096,1024（fc2） | 0.417800秒 | 0.508094秒 | 1.2161 | NF24維持 |
| 12294,1024,3072（QKV） | 0.331979秒 | 0.218777秒 | 0.6590 | FP32展開 |
| 12294,1024,1024（attn out） | 0.074678秒 | 0.071998秒 | 0.9641 | FP32展開 |

4 shapeともNF24 kernelとの差はRMSE 0、最大絶対差0だった。native FP32 kernelはNF24が
復号後に使うのと同じFMA順序を使うためである。MKL版も試したが、MKL OpenMPと既存の
GCC/libgomp kernelを同一processで交互に使うとSDPAまで遅くなったため不採用とした。

## 5. s1 paired結果

共有ホスト負荷により、同一構成でもs1 `elapsed_sec`が148から208秒まで変動した。
したがって歴史値との単純比較ではなく、連続する基準と候補を比較した。

高速状態の隣接pairは次の通りだった。

| 項目 | packed-v3基準 | 高メモリ候補 | 変化 |
|---|---:|---:|---:|
| sampler elapsed | 148.163秒 | 143.534秒 | 3.12%短縮 |
| forward total | 126.792秒 | 125.870秒 | 0.73%短縮 |
| QKV projection | 10.083秒 | 9.214秒 | 8.62%短縮 |
| Attention out | 3.329秒 | 3.058秒 | 8.16%短縮 |
| 全Linear | 50.782秒 | 49.703秒 | 2.12%短縮 |
| process total | 202.884秒 | 196.388秒 | 3.20%短縮 |
| peak aggregate RSS | 3,163,529,216 byte | 4,254,904,320 byte | +1,091,375,104 byte |
| combined RMSE | 0 | 0 | bit完全一致 |

低速状態の隣接pairでも、QKVは8.32%、Attention outは5.23%、forwardは1.91%短縮した。
sampler elapsedは0.23%短縮で、共有CPU変動が局所kernel改善を上回った。したがって確実に
主張できるのはQKV/outと全Linearの短縮であり、s20全体の率は未確定である。

QKV/outの低負荷pair差を単純に20 stepへ積むと約22.8秒である。現行2747.270秒に対する
保守的な予測は約2724秒だが、正式値ではない。s20を実測するまで20から60秒程度の短縮候補とする。

## 6. SDPAへ追加メモリを与えた結果

key tileを大きくするとonline-softmax merge回数は減るが、score scratchがL1/L2へ圧力を
かける。tile 512を基準にした結果は次の通りだった。

| key tile | L=8194 | L=12294 | 判定 |
|---:|---:|---:|---|
| 1024 | 1.0769倍 | 1.0570倍 | 不採用 |
| 2048 | 1.4270倍 | 1.3845倍 | 不採用 |
| 4096 | 1.5304倍 | 1.3301倍 | 不採用 |

差はRMSE `3.6e-9`から`6.9e-9`でfiniteだったが、速度が悪化した。SDPAはkey tile 512を維持する。

## 7. 採否と次のgate

高メモリ候補としてQKV/outのnative FP32常駐を残す。低リソース既定は全Linear NF24のままにする。
正式採用には次を要求する。

1. 事前pack直接loadでruntime変換peakを除いたRSSを測る。
2. s4で基準/候補を各3回交互実行し、QKV/out、全Linear、forward中央値を比較する。
3. s20を1回以上完走し、combined RMSE `<= 5e-4`、NaN/Inf/fallback 0を確認する。
4. s20が20秒以上または1%以上短縮した場合、高メモリprofileとして採用する。

現時点では、追加メモリを無制限に使うほど速くなるわけではない。速度へ変換できたのは、
復号コストが大きく、FP32 weight帯域増加を上回ったQKVとAttention outだけである。
