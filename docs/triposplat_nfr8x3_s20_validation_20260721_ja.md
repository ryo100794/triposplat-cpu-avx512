# TripoSplat NFR8x3 CPU s20 検証報告 2026-07-21

## 1. 結論

TripoSplat Flow modelの全206 Linearを、float32 weightを保持しない24 bit/weightの
NFR8x3へ置換し、CPU-only、1024 canvas、20 steps、guidance 3.0、shift 3.0を
完走した。NFR8x3は「1段の非線形NF8」と「2段の符号付き対称int8 residual」を
合成するweight-only量子化である。

| 項目 | 結果 | gate | 判定 |
|---|---:|---:|---|
| wall time | 4640.813秒 | 7200秒未満 | 合格 |
| 1時間tier | 4640.813秒 | 3600秒未満 | 不合格 |
| combined RMSE | 2.31568e-5 | 5.0e-4以下 | 合格 |
| latent RMSE | 3.25672e-5 | 参考値 | 合格 |
| camera RMSE | 3.44202e-6 | 1.0e-4以下 | 合格 |
| Flow NaN / Inf | 0 / 0 | 0 / 0 | 合格 |
| Linear fallback | 0 / 7252 calls | 0 | 合格 |
| 6視点render PSNR | 平均76.24 dB、最低69.33 dB | strict float32比 | 合格 |

元のCPU float32 10856.388秒に対して2.34倍速い。一方、量子化しないstrict native
AVX-512の3322.886秒より39.7%遅い。したがって、結果同等性とweight削減は達成したが、
最速CPU backendとしての置換にはなっていない。

## 2. NFR8x3の量子化式

Linearの入力を `x`、weightを `W`、biasを `b`、出力を `y` とする。

```text
y = x W^T + b
```

`n` を出力channel、`k` を入力channelとする。まず各出力channelのscale `s0_n` を
求め、正規化weight `W_nk / s0_n` に最も近いNF8 codebook値を選ぶ。NF8 codebook
`C` は正規分布の分位点を基にした256要素の非等間隔表である。

```text
q0_nk = argmin_q |W_nk / s0_n - C[q]|
W0_nk = s0_n C[q0_nk]
R1_nk = W_nk - W0_nk
```

`q0_nk` は1 byteのindexである。小さい値が密な非線形codebookを使うため、単純な
一様int8より0付近のweightを細かく表現できる。

2段目と3段目は、前段残差 `Rj` を出力channel単位のsigned int8で量子化する。

```text
sj_n = max_k |Rj_nk| / 127
qj_nk = clip(round(Rj_nk / sj_n), -127, 127)
Wj_nk = sj_n qj_nk
R(j+1)_nk = Rj_nk - Wj_nk
```

最終近似weightは次式になる。

```text
W_hat_nk = s0_n C[q0_nk] + s1_n q1_nk + s2_n q2_nk
```

したがってGEMMはfloat32 weightを復元して保存せず、積和の直前に3 streamを合成する。

```text
y_mn = b_n + sum_k x_mk *
       (s0_n C[q0_nk] + s1_n q1_nk + s2_n q2_nk)
```

1段目だけ非線形lookupを行い、残り2段はint8 sign extensionとfloat変換で済む。
純粋な3段NF8よりcodebook gatherを2回減らし、残差表現精度も改善する。

## 3. packed data構造

各Linearは次を保持する。

- 1段目NF8 index: `uint8`、weightごとに1 byte
- 2段目residual: `int8`、weightごとに1 byte
- 3段目residual: `int8`、weightごとに1 byte
- 各段のoutput-channel scale: float32
- bias: float32

全206 Linearの元weightは1,480,996,180 byte、packed表現は1,114,596,688 byteで、
比率は75.2599%である。量子化後のLinear moduleはfloat32 weight tensorを保持しない。

ただし、現実装は公式float32 checkpointを一度loadしてからruntimeでpackする。ここで
示す削減率はpack後のLinear weight常駐量であり、起動時peakやcheckpoint file自体を
24 bit化した値ではない。起動時peakも削減するには、事前pack済みweight fileの保存と
直接loadが別途必要である。model/checkpointは公開repositoryには含めない。

## 4. AVX-512実行

kernelは16出力channelを1本のZMMで扱い、16入力rowを並行する16x16 tileを使う。
各 `k` でNF8 indexをgatherし、2本のsigned int8 residualをfloat32へ変換し、scaleを
掛けてweight vectorを作る。そのvectorを16個の入力scalarでbroadcast-FMAする。

16-row tileは8-row版と同一SHAのs1 latent/cameraを生成し、全体時間を507.736秒から
398.512秒へ短縮した。24-row候補は445.657秒、4-thread候補は490.705秒だったため、
16-row、8 threadsを採用した。libraryはrow tileとresidual modeをABIから返し、Python
側は要求modeとの不一致を例外にする。

## 5. 段階gate

| 構成 | steps | wall time | Linear time | combined RMSE | 判定 |
|---|---:|---:|---:|---:|---|
| NF8 1段 | 1 | 295.746秒 | 134.349秒 | 2.51245e-2 | 不合格 |
| RNF8x2 | 1 | 414.846秒 | 241.226秒 | 4.50681e-4 | 不合格 |
| RNF8x3、純NF8、8-row | 1 | 507.736秒 | 332.770秒 | 8.12530e-6 | 合格 |
| RNF8x3、純NF8、16-row | 1 | 398.512秒 | 232.616秒 | 8.12530e-6 | 合格 |
| RNF8x3、純NF8、8-row | 4 | 1924.940秒 | 1301.402秒 | 1.16460e-4 | 合格 |
| NFR8x3、16-row | 1 | 314.608秒 | 150.021秒 | 5.62961e-6 | 合格 |
| NFR8x3、16-row | 4 | 982.154秒 | 490.539秒 | 1.64431e-5 | 合格 |
| NFR8x3、16-row | 20 | 4640.813秒 | 2365.633秒 | 2.31568e-5 | 合格 |

NFR8x3 s20のaggregate weight RMSEは5.14587e-8、最大絶対誤差は5.27652e-7だった。
Linear runtimeは全体の約51.0%で、SDPAは1959.760秒だった。今後の速度改善は3段を
別々にdecodeするoverheadだけでなく、依然として大きいSDPAも同時に見る必要がある。

## 6. Gaussianとrenderer評価

同じdecoder random NPZを使い、candidate latentから262,144 GaussiansをCPUで生成した。
decoder/export内部時間は67.635秒、監視できたPython peak RSSは約2.01 GiBだった。
PLY、SPLAT、front_x初期視点の単一WebGL HTMLを生成した。

PLYのrecord単位比較は完全一致しない。latentの微小差がoctree境界をまたぐ箇所では、
一部Gaussianの生成分岐、opacity、rotation、scaleに大きな外れ値が生じる。SPLATは
座標sort後のrecord順も変わるため、同じindex同士のbyte比較はsemantic比較にならない。

そこでstrict native float32 GaussianとNFR8x3 Gaussianを同じCPU rendererで6方向から
比較した。

| view | PSNR |
|---|---:|
| front_z | 76.81 dB |
| back_z | 69.33 dB |
| front_x | 71.16 dB |
| back_x | 80.64 dB |
| front_y | 83.36 dB |
| back_y | 76.16 dB |

平均は76.24 dB、最大MAEは2.16017e-5である。公式初期姿勢の1024px比較はPSNR
74.27 dB、MAE 7.38879e-6だった。表示結果は全視点でstrict float32と同等だが、
Gaussian parameterのbit一致を達成したという意味ではない。

## 7. ref実装を使う位置

TripoSplatのDINO/VAE条件生成、Flow model、camera推論、Gaussian decoderは公式構造を
維持している。refのGaussian最適化や色合わせで出力を補正していない。

refから採用した考え方は、低メモリのchunk export、巨大な中間配列を避けるrenderer、
段階ごとの比較とmanifest記録である。つまりrefはdecoder後のexport・評価設計の参考で、
NFR8x3の推論式や見えない部分の生成を置き換えるものではない。

## 8. 判定と残件

達成済み:

- 全206 Linearのruntime非線形/residual量子化
- float32 Linear weight非保持
- packed AVX-512直接GEMM、fallback 0
- s1、s4、s20品質gate
- CPU Gaussian、PLY、SPLAT、単一WebGL viewer
- strict float32との6視点表示同等性

残件:

- 4640.813秒から3600秒未満への短縮
- 事前pack済みweight fileの直接loadによる起動時peak削減
- NFR8x3をraw画像からviewerまでの単一入口へ統合
- AVX-512非対応CPU向けbackend

現時点のNFR8x3は品質を保つ低容量weight backendとして成立するが、速度優先なら
3322.886秒のstrict float32 AVX-512 backendを選ぶべきである。
