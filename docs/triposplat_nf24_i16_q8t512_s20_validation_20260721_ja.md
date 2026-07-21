# TripoSplat NF24 int16 + SDPA tile512 CPU検証報告 2026-07-21

## 1. 目的

TripoSplatの機能や推論式を追加・変更せず、公式Flow modelの出力をCPU-onlyの
低リソース実装で再現する。従来のNFR8x3は品質gateを通過したが、s20が
4640.813秒で1時間tierを超えていた。本検証では次を同時に満たすことを目標にした。

- canvas 1024、20 steps、guidance 3.0、shift 3.0
- 全206 Linearを24-bit packed weightから直接AVX-512実行
- Linear fallback 0、float32 Linear weight保持0
- CPU float32 s20比combined RMSE `<= 5.0e-4`
- camera RMSE `<= 1.0e-4`
- samplerを含む記録時間 `< 3600秒`

背景除去、condition encoder、Flow、GS decoder、renderer、viewerというTripoSplatの
役割分担は変えない。ここで置換するのはFlow model内部のLinear weight表現と、
意味を維持したexact SDPAのCPU実行方法である。

## 2. NF24 int16表現

出力channelごとのfloat32 weightを `w`、そのchannelの最大絶対値を `m` とする。
基本scaleを

```text
s = max(m, tiny) / 127
```

とする。1段目は線形int8ではなく、NF8 codebookを整数化した不等間隔集合

```text
A = unique(round(127 * NF8))
```

から最近傍値を選ぶ。

```text
q0 = argmin(a in A) |w / s - a|
r1 = w - q0 * s
```

2段目と3段目は共有scaleに対するresidual量子化である。

```text
q1 = clamp(round(r1 / (s / 4)), -128, 127)
r2 = r1 - q1 * (s / 4)
q2 = clamp(round(r2 / (s / 1024)), -128, 127)
```

保存時は `q0` と `q1` を1本のsigned int16へまとめる。

```text
q01 = 4 * q0 + q1
scale_final = s / 1024
w_hat = scale_final * (256 * q01 + q2)
```

したがって1 weightは `q01:int16 + q2:int8 = 24 bit` である。runtime decoderは
16 weightごとに次のAVX-512処理を行う。

1. 16個のint16を16個のint32へ符号拡張
2. 8 bit左shiftして `256 * q01` を作る
3. 16個のint8 `q2` をint32へ符号拡張して加算
4. 16個をfloat32へ変換し、channel scaleを乗算
5. 16-row x 16-outputのFMAへそのまま渡す

3 streamを別々にdecodeするNFR8x3と異なり、整数合成をfloat変換の前に終える。
代表shape `M=20488, K=1024, N=4096`、8 threadsでは、NFR8x3の中央値
1.752063秒に対しNF24 int16は0.931842秒、比率0.531854だった。

全modelのweight量子化誤差はRMSE `3.33420e-7`、最大絶対誤差
`8.95141e-6`、packed/original比 `75.1087%` である。

## 3. exact SDPAのtile拡大

query blockは8のまま、online softmaxのkey tileを128から512へ拡大した。
attentionのkey削減、近似、top-k、low-rank化は行っていない。tile内max/sumと
tile間のonline softmax統合式は従来と同じである。

| 候補 | L=8194 median | L=12294 median | 判定 |
|---|---:|---:|---|
| query block 8 | 1.198794秒 | 2.802159秒 | 基準 |
| query block 12 | 1.235927秒 | 2.995563秒 | 不採用 |
| query block 16 | 1.265095秒 | 2.949185秒 | 不採用 |

key tile比較ではtile512が最速だった。別時間帯の測定でhost負荷の揺れがあるため、
最終採否はmicrobenchmarkだけでなく同条件s1/s4のmodel全体時間で決めた。
PyTorch SDPA比のRMSEはL=8194で `5.99e-9`、L=12294で `5.11e-9` だった。

## 4. s1、s4、s20結果

同じprepared image、condition NPZ、external noise、seed、guidance、shiftと、元の
CPU float32 reference NPZを使った。時間は各manifestの `elapsed_sec` で比較する。

| steps | elapsed | Linear | SDPA | combined RMSE | camera RMSE | fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 189.345秒 | 77.682秒 | 82.286秒 | 2.22759e-5 | 2.22542e-5 | 0 |
| 4 | 707.630秒 | 304.530秒 | 327.992秒 | 9.85188e-5 | 2.05602e-5 | 0 |
| 20 | 3471.330秒 | 1519.392秒 | 1645.894秒 | 9.37666e-5 | 8.93055e-6 | 0 |

s1は同じNF24 int16 + tile128の213.369秒から11.3%、s4は781.399秒から9.4%
短縮した。s1/s4の線形外挿はs20を約3472秒と予測した。

## 5. 事前pack済みcheckpoint

runtime量子化だけでは、起動時に公式checkpointをfloat32 Flow model全体へloadした
後で24-bitへ変換するため、起動時peakを解消できない。そこで次のshard形式を実装した。

- 各Linear: `code01_t:int16`、`q2_t:int8`、`scale:float32`、`bias:float32`
- 非Linear state: 独立したsafetensors shard
- manifest: source SHA256、module名、shape、量子化誤差、shard SHA256、byte数
- 各Linear shardを書いた直後にmanifestをatomic更新
- 再開時はbyte数とSHA256が一致するshardを再量子化しない

loaderは `nn.Linear.__init__` の間だけparameterをmeta deviceへ置く。LayerNorm、buffer、
Sobol positionなどは通常のCPU上で構築し、非Linear shardから復元する。その後、
各Linearのmeta weightを空parameterへ置換し、packed bufferとAVX-512 forwardをattachする。
公式Flow checkpointをloadする関数は呼ばない。

小型2層往復testではRMSE `1.88904e-6`、最大絶対誤差 `5.87106e-6`、fallback 0、
checksum再開成功、`source_checkpoint_loaded=false` を確認した。

公式206 Linearでの直接load検証:

- packed bytes: 1,113,368,944 byte
- runtime-pack peak RSS: 3,437,973,504 byte（約3.20 GiB）
- prepacked peak RSS: 2,551,123,968 byte（約2.38 GiB、25.8%削減）
- prepacked s1 elapsed: manifest 207.980秒、checksum検証・比較込み262.797秒
- runtime-pack s1との直接比較: latent/cameraともbit完全一致（最大絶対差0）

事前packデータは公式weightの派生物なのでpublic repositoryへ含めない。repositoryには
形式、converter、loader、smoke testだけを置く。

## 6. raw画像からviewerまでの単一入口

`scripts/run_cpu_low_resource_nf24.sh` を追加した。既存の再開可能なend-to-end
pipelineを使い、Flow runnerだけをNF24 int16 + tile512へ切り替える。

```bash
INPUT=/path/to/source.png \
REFERENCE_NPZ=/path/to/float32_s20/base_latent.npz \
TRIPOSPLAT_RNF8_PREPACKED_DIR=/path/to/nf24_i16_prepacked \
bash scripts/run_cpu_low_resource_nf24.sh
```

出力stageはprepared image、condition、noise、Flow latent/camera、Gaussian、PLY/SPLAT、
reference render、単一WebGL viewerである。`RESUME=1` で完了済みstageを再利用する。

## 7. 結論

NF24 int16とSDPA key tile 512の組合せは、TripoSplatの推論式を変えず、s20を3471.330秒まで短縮して1時間未満のgateを通過した。combined RMSEは9.37666e-5、camera RMSEは8.93055e-6、全206 Linearのfallbackは0である。事前pack済みcheckpointの直接loadも公式checkpointをruntimeで展開せず同一出力を再現し、起動から終了までのpeak RSSを25.8%削減した。checksum検証を有効にした直接loadは今回runtime-packより総時間が長く、速度改善ではなくdisk/memory peak削減として採用する。
