# TripoSplat CPU P5 assembly着手条件監査

日付: 2026-07-22

## 1. 目的

P5はinline assemblyを書くこと自体を目的にしない。GCC13/Zen4、C intrinsic、tile/layout/fusionを
適用した後でも、特定の内側loopを置き換えることでend-to-end 5%以上の短縮が見込めるかを判定する。

## 2. 逆アセンブル結果

### 2.1 NF24 GEMM

hot OpenMP functionは次の処理をZMMで実行していた。

- `vpmovsxwd`と`vpmovsxbd`でint16/int8 codeを展開する。
- shift、add、`vcvtdq2ps`、scale乗算で16 weightをFP32へ復号する。
- 復号した1本のZMM weightを8 input rowへbroadcast FMAする。
- 8 accumulatorを`vfmadd231ps`で更新し、16 output列をまとめて保存する。

hot loop内に関数call、scalar FMA、gatherはなく、C intrinsicの意図どおり8-row x 16-columnへ
展開されている。compilerがvector化に失敗している状態ではない。

### 2.2 packed SDPA

packed SDPAも8 queryを並列に処理し、QKとPVをZMM FMAへ展開していた。softmaxのexp近似は
`vrndscaleps`、FMA、integer exponent生成までinline化され、hot loop内に`expf` callはない。
一方、query block 8/key tile 512のscore、max、sum、accumulatorによりhot functionは大きい。

| function | baseline code size | unroll code size | 増加 |
|---|---:|---:|---:|
| NF24 GEMM range OpenMP | 840 byte | 1,144 byte | 36.2% |
| packed SDPA OpenMP | 5,636 byte | 7,641 byte | 35.6% |

## 3. compiler unroll probe

同じGCC13/Zen4 buildへ`-funroll-loops`だけを追加し、4 CPU affinity、7回paired中央値で測った。
全候補の出力はbit一致した。

### 3.1 NF24 GEMM

| shape `(M,K,N)` | baseline | unroll | 比率 | 短縮 |
|---|---:|---:|---:|---:|
| 12294,1024,4096 | 0.386852秒 | 0.365880秒 | 0.9458 | 5.42% |
| 12294,4096,1024 | 0.508618秒 | 0.459653秒 | 0.9037 | 9.63% |
| 12294,1024,3072 | 0.217183秒 | 0.208910秒 | 0.9619 | 3.81% |
| 12294,1024,1024 | 0.070597秒 | 0.069016秒 | 0.9776 | 2.24% |

unrollはshape依存だが有効だった。これはassemblyの前にcompiler flagとshape別dispatchを詰める
余地が残っていることを示す。

### 3.2 packed SDPA

| sequence長 | baseline | unroll | 比率 | 判定 |
|---:|---:|---:|---:|---|
| 8194 | 0.845691秒 | 0.855593秒 | 1.0117 | 不採用 |
| 12294 | 1.927841秒 | 1.932225秒 | 1.0023 | 不採用 |

SDPAではcode size増加に対する利益がなく、明示unrollを採用しない。

## 4. 統合s1 probe

NF24 unroll backendをpacked-v3へ接続した1回のs1 probeは次の結果だった。

| 指標 | 保存済みbaseline | unroll probe |
|---|---:|---:|
| elapsed | 140.059秒 | 137.752秒 |
| Linear | 49.113秒 | 47.674秒 |
| SDPA | 67.116秒 | 65.123秒 |
| fallback | 0 | 0 |
| latent/camera差 | 基準 | RMSE/max 0 |

wall timeは1.65%、Linearは2.93%短かったが、同時刻paired測定ではなくSDPAも約3%変動している。
正式採用にはbaseline/unrollを交互に3回以上実行した中央値が必要である。

## 5. Amdahl判定

s20 2747.270秒ではLinear 1008.472秒、SDPA 1432.319秒だった。Linearを実shapeごとに分類し、
上のmicrobenchmark比率を掛けると推定短縮は次になる。

| 系統 | s20時間 | 推定短縮 |
|---|---:|---:|
| QKV 1024->3072 | 214.262秒 | 8.16秒 |
| attention out 1024->1024 | 71.703秒 | 1.61秒 |
| MLP fc1 1024->4096 | 343.777秒 | 18.64秒 |
| MLP fc2 4096->1024 | 358.420秒 | 34.50秒 |
| 合計 | 988.162秒 | 約62.9秒 |

約62.9秒はs20全体の約2.29%であり、P5の5%条件を満たさない。5%短縮にはLinear全体を
少なくとも13.6%、またはSDPA全体を9.6%短縮する必要がある。現時点でその差を示すpaired
microbenchmarkはない。

## 6. assemblyを見送る理由

1. GEMMとSDPAはすでにZMM/FMAへ展開され、scalar fallbackやhot loop callがない。
2. compiler unrollで確認できたend-to-end上限は約2.3%である。
3. SDPA unrollはcode sizeを35.6%増やして遅くなった。
4. hardware counterが許可されず、front-end、port、cache、帯域stallを特定できていない。
5. inline assemblyはregister allocatorを広く拘束し、GCC/CPU差への保守負担を増やす。

したがって`.S` microkernelもinline assemblyも現時点では実装しない。

## 7. 次の着手条件

assemblyを再検討する条件は次のすべてである。

1. `perf`または同等counterで、特定の10〜30命令loopが全体stallの原因と分かる。
2. C intrinsic、PGO、shape dispatch、compiler flagで同じ改善を出せない。
3. standalone `.S`候補がintrinsicより5%以上速く、出力bit一致する。
4. Amdahl換算でend-to-end 5%以上を見込める。
5. GCC13/Zen4以外はC intrinsicへ戻るruntime dispatchを持つ。

直近のexact候補はassemblyではなく、GEMM unrollをshape別に有効化して3回paired s1、s4、s20を
通すことである。packed SDPAは現buildを維持する。

## 8. P5判定

- assembly着手: 条件未達のため見送り。
- NF24 GEMM unroll: exact候補、正式採用前のpaired評価待ち。
- packed SDPA unroll: microbenchmarkで不採用。
- 現行runtimeへ追加したassembly: なし。

raw成果物はGDriveへ移し、10ファイルのchecksum差分0を確認後、共有先から削除した。
