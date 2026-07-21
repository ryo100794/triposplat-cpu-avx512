# TripoSplat CPU低リソース完全実装マイルストーン 2026-07-19

## 完全実装の定義

ここでの完全実装は、TripoSplatの機能を増やすことではない。GPU基準 `gpu_encoded_rmbg_1024_s20_g3_seed0_n262k_ref_renderer` と同じ入力に対して、CPU-onlyの低リソース実装が同等の推論結果、3DGS出力、viewer/renderer比較を再現できる状態を指す。

現状は部分実装である。`blocks.5.attn.out` の mixed pc8 / AVX2 GEMMはs1 gateを通したが、s4/s20では誤差累積が残る。モデル全体の非線形量子化、activation量子化、attention/MLPを含む完全SIMD置換は未完了。

## 最終ゴール

- CPU-onlyでs20/g3を実行し、GPU基準およびCPU float32 baselineに対して比較可能な3DGS成果物を自動生成する。
- `blocks.5.attn.out` 起点のweight-only pc8を、channel/group単位のmixed precisionから、モデル全体の非線形量子化候補へ拡張する。
- SIMD backendは一部GEMMだけでなく、主要Linear/MLP/attention周辺のhot pathをカバーする。
- 低リソースCPU環境で、品質を壊さずに現行CPU float32 baselineより実用的に速いことを示す。
- 成果物はquotaを管理した作業領域へ保存し、検証後にchecksum付きarchiveへ移す。

## 品質ゲート

| ゲート | 必須条件 | 理由 |
|---|---:|---|
| s1 latent | combined RMSE `<= 0.0002` | 既に達成済みの最低条件。退行検知に使う。 |
| s4 latent | combined RMSE `<= 0.0002` を目標、暫定許容 `<= 0.0003` | s1だけでは誤差累積を見逃すため。 |
| s20 latent | CPU float32 s20比 combined RMSE `<= 0.0005` を目標 | 完全置換候補としての主判定。 |
| s20 camera | camera RMSE `<= 0.0001` | 視点破綻がrendererへ伝播するため。 |
| renderer | 6視点平均PSNRで現行CPU候補以上、worst viewを悪化させない | 見た目の破綻検知。 |
| viewer | WebGL viewerで初期視点、ドラッグ操作、表示解像感が破綻しない | 最終確認用。 |
| 再現性 | seed固定で同じmanifest/compare JSONを再生成できる | 評価実験ではなく実装候補にするため。 |

## 速度ゲート

現行基準:

- オリジナルCPU float32 s20: `10856.39 sec`
- 現行主候補 `keep095`: `7555.97 sec`
- GPU基準s20: `29.98 sec`

段階目標:

| 段階 | 20step目標時間 | 判定 |
|---|---:|---|
| 実用候補入口 | `< 7200 sec` | 2時間未満。現行keep095から小幅改善。 |
| 低リソース候補 | `< 3600 sec` | 1時間未満。PyTorch hot path削減が必要。 |
| 実用目標 | `< 1800 sec` | 30分未満。SIMD backendの面展開が必要。 |
| 長期目標 | `< 600 sec` | 10分未満。attention/MLP/decoder/export全体の設計変更が必要。 |

## マイルストーン

### M0 現状固定

状態: 完了。

成果物:

- `docs/triposplat_cpu_s20_gpu_reference_current_status_20260719_ja.md`
- `docs/triposplat_goal_completion_audit_20260719_ja.md`
- `artifacts/audits/triposplat_cpu_s20_gpu_reference_current_status_20260719_summary.json`
- `artifacts/audits/triposplat_goal_completion_audit_20260719_summary.json`

ゴール:

- GPU基準、CPU候補、renderer/viewer、mask、quant_flow証跡を固定する。
- 以後の比較はこの状態を退行検知の基準にする。

### M1 s4を通るb05 mixed precision

期限目標: 2026-07-22。

ゴール:

- `blocks.5.attn.out` 単体でs4 combined RMSE `<= 0.0002` を達成する。
- late-weighted keep095のs1成功を維持する。
- hot_replace storage ratioを `<= 1.0` に戻す。

実装内容:

- final-state residual weighted maskを再設計する。
- step 1-4だけでなく、s8/s20の後半誤差をrow/group scoreへ入れる。
- row単位保護ではなく、channel/group単位の重要度で保護する。

終了条件:

- s1/s4 compare JSON。
- mask npzとmask summary JSON。
- s4 gate結果を日本語docsに記録。

### M2 非線形weight量子化の導入

期限目標: 2026-07-24。

ゴール:

- 線形pc8だけでなく、NF4、log-scale、k-means/codebookの候補を同じ評価器で比較する。
- 最初の対象は `blocks.5.attn.out`、次に `noise_refiner.*.attn.out`。

実装内容:

- groupごとにcodebookを持つweight-only非線形量子化を追加する。
- scale/zero-point固定の線形量子化と、codebook lookup型の非線形量子化を同じruntime APIで切り替える。
- calibrationは入力分布、出力残差、最終latent残差の3種類を保持する。

終了条件:

- `linear pc8`、`log pc8`、`nf4/codebook` のs1/s4比較表。
- 非線形量子化でs4が線形pc8を上回ること、または上回らない理由の確定。

### M3 SIMD backendの実装範囲拡張

期限目標: 2026-07-26。

ゴール:

- mixed pc8 backendを1つのGEMM実験から、主要Linearの差し替え候補へ拡張する。
- AVX2を最低ラインにし、AVX512が使える環境では自動選択する。

実装内容:

- packed weight layoutを事前生成する。
- n8 tile hot/cold mixed precisionを複数layerで使えるABIへ整理する。
- kernelごとに `call_count`、`kernel_sec`、`fallback_count`、`storage_ratio` をmanifestに必ず出す。
- fallbackが発生した場合は品質評価から除外できるようにする。

終了条件:

- b05、noise_refiner、MLP候補のkernel timing表。
- PyTorch側のforward時間とSIMD kernel時間を分離したprofile。
- fallbackなしでs1/s4が走る。

### M4 activation/attention低リソース化

期限目標: 2026-07-28。

ゴール:

- 現在のwall time支配要因であるPyTorch CPU forward、attention、メモリ転送を削る。
- weight GEMMだけの最適化から脱却する。

実装内容:

- attention backend別の時間とメモリ使用量を固定フォーマットで記録する。
- activationの一時バッファを再利用する。
- attention中間値の低精度化、chunk化、cache化を比較する。
- 精度が落ちる近似はTripoSplat機能追加ではなく、同等実装の範囲内でのみ採用する。

終了条件:

- s4で品質ゲートを満たしつつ、20step推定時間が現行keep095より明確に短い。
- memory peakとdisk usageを監査JSONに残す。

### M5 s20/g3 end-to-end昇格試験

期限目標: 2026-07-30。

ゴール:

- CPU-only s20/g3を完全に走らせ、PLY/SPLAT/viewer/ref rendererを生成する。
- GPU基準とCPU float32 baselineの両方に対する比較を自動生成する。

実装内容:

- s20 manifest、latent compare、PLY diff、renderer image compare、contact sheetを一括生成する。
- exportはlowmem streamingを維持する。
- local/remote同期を自動化する。

終了条件:

- s20 combined RMSE `<= 0.0005` 目標。
- renderer worst viewが現行候補より悪化しない。
- 20step時間 `< 7200 sec` を最低ラインとして通す。

### M6 実用候補パッケージ化

期限目標: 2026-07-31。

ゴール:

- 評価スクリプト群ではなく、CPU低リソース実装候補として再実行できる形にする。

実装内容:

- `run_cpu_low_resource_s20_g3.sh` 相当の単一入口を作る。
- 入力画像、mask/codebook、backend選択、renderer/viewer生成、local syncを1つのmanifestにまとめる。
- 失敗時に中間ファイルから再開できるようにする。

終了条件:

- clean workspaceで1frame再実行できる。
- 出力先は環境変数で指定し、repositoryへmodel・入力・生成物を混在させない。
- 日本語runbookと監査JSONを生成する。

## 作業優先順位

1. s4を通す。s1だけでは先に進めない。
2. 非線形weight量子化はb05単体で評価する。いきなり全layerへ広げない。
3. SIMD backendはfallbackなし、timing可視化ありを必須にする。
4. 速度改善はGEMM単体ではなく、attention/MLP/メモリ転送を含むforward全体を見る。
5. s20昇格は最後に行う。s4未達のままs20を増やしても計算時間を浪費する。

## 現時点の次アクション

- M1から開始する。
- late-weighted keep095 maskを基準に、s4 final-state weighted maskを作り直す。
- s4 combined RMSE `<= 0.0002` を第一ゴールにする。
- その後、同じ評価器で非線形codebook量子化を比較する。
