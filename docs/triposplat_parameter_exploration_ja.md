# TripoSplat低リソース検証で探索しているパラメータ

## 目的と前提

この文書は、単眼1-frame TripoSplat / 3DGS検証で探索したパラメータの種類、機能、品質・時間・メモリへの影響を整理するものです。

現在の主目的は、TripoSplatに機能を足すことではありません。目的は、本家TripoSplatと同じ raw Gaussian 出力、つまり `output.ply` / `output.splat` を低リソースに生成できるよう、実装境界を置き換えていくことです。

したがって、以下を区別します。

- **主結果**: DINO/VAE/flow/decoderから出た raw TripoSplat PLY/SPLAT。
- **低リソース実装パラメータ**: 同じ結果を維持しながら、メモリ配置、streaming、chunk size、dtype、モデル寿命を変えるもの。
- **評価パラメータ**: raw PLYをどうレンダリングし、どう比較するかを変えるもの。raw結果は変えない。
- **診断・非主結果パラメータ**: LS色補正、opacity bake、2.5D proxyなど。見た目確認には有用だが、TripoSplat同等化の証拠ではない。

軽量ref実装は、TripoSplat推論モデルの代替ではなく、低リソース実装上の参考です。小パッチ処理、streaming、bounded memory、局所的なGaussian合成、評価rendererの作り方を参照しています。

## 入力・前処理パラメータ

### 入力動画・フレーム

検証では動画から抽出した単一フレームを使います。公開repositoryには元動画、抽出フレーム、個別のlocal pathを含めません。

機能:

- TripoSplatの単眼入力になる画像を決める。
- object-centricな単体物体画像に近いほど、本家TripoSplatの学習分布に近くなりやすい。
- TV番組全景のような複数人物・机・背景を含む画像は、単眼3D推定として難しく、カメラ方向や奥行きが破綻しやすい。

探索状況:

- 全景フレーム。
- `right_person_medium` などの手動crop候補。
- 本家サンプル `building_stone_house.webp`。

### `--canvas-size`

TripoSplatへ入れる正方形画像の解像度です。

主な値:

- `512`: CPU検証の基本設定。時間・メモリを抑えやすい。
- `1024`: 本家サンプルやGPU基準に近い高解像設定。CPUではflow sampler時間が大きく伸びる。

機能:

- 前処理後の入力画像サイズを決める。
- DINO/VAE encoderの入力情報量に影響する。
- Gaussian数とstepsが同じでも、入力detailと推論時間が変わる。

観測:

- 512/8/32kはCPUで約1700-1800秒級。
- 1024/12/131kはCPUで約3878秒。
- 1024/16/131kはCPUで完走。elapsed 5212.77秒、最大RSSは4,483,644KB（約4.48GB）。標準cameraでは best `front_x`, PSNR 14.4012 dB / MAE 0.079861。camera sweep bestは `front_x`, FOV 42, distance 1.6, PSNR 15.22595 dB / MAE 0.073392。
- 1024/20/262kは `lowmem_sampler + separate_cfg + lowmem_export` でCPU完走。elapsed 8706.30秒、標準camera best `front_x`, PSNR 14.8854 dB / MAE 0.075476。run中の `oom` / `oom_kill` 増加はなし。

### 背景除去・準備画像

使用スクリプト:

- `triposplat_cpu_prepare_image.py`

主な出力:

- `prepared_rgba.png`: alpha付き準備画像。staged pathで使用。
- `prepared_rgb.webp`: 背景除去済みRGB。direct encode pathで使用。

関連パラメータ:

- `--erode-radius`: alphaを少し縮める半径。
- alpha有無: 実alphaがある場合、BiRefNet再実行を避けやすい。

機能:

- BiRefNet背景除去を本体推論から切り離し、CPUメモリと失敗要因を減らす。
- staged pathでは、前処理済みRGBAを使って本体側の追加背景除去を避ける。

注意:

- 前処理の違いはraw TripoSplat出力に影響するため、同等性比較では同じ準備画像を使う必要がある。

## TripoSplat推論パラメータ

### `--num-gaussians`

出力Gaussian数です。

探索値:

- `32768`: 512 CPU smoke / 低負荷検証。
- `131072`: 1024 CPUの中間設定。
- `262144`: GPU raw基準、本家高品質寄り設定。今回のCPU最高設定でも採用。

機能:

- Decoder token数と最終PLY/SPLATサイズを決める。
- 本家設定では `gaussians_per_point=32` なので、`num_gaussians / 32` がdecoder token数になる。
- viewer・rendererの描画量にも直接効く。

影響:

- 多いほど細部表現の余地は増える。
- CPUではdecode/export/viewer負荷も増えるが、今回の最大ボトルネックはflow samplerであり、exportはlowmem化でかなり抑えられている。

### `--steps`

Flow matching samplerのEuler積分step数です。

探索値:

- `1`: 最初の完走確認。品質は低い。
- `4`, `8`, `12`, `16`: CPU品質と時間の探索。
- `20`: GPU基準、公式品質寄り。CPUでは非常に重い。

機能:

- latent/cameraをnoiseから生成結果へ近づける積分回数。
- `guidance_scale > 1` の場合、通常は各stepでconditional/unconditionalの2forwardが必要。
- `steps=20`, `guidance=3.0` は、実質40回のflow model forward相当になる。

観測:

- 512/8/32k seed0 raw: local renderer PSNR 14.6818 dB。
- 512/12/32k seed0 raw: PSNR 14.9606 dB。
- 512/16/32k seed0 raw: PSNR 15.0998 dB、elapsed 3381.64 sec。
- 1024/12/131k seed0 raw: standard camera PSNR 14.4785 dB、camera sweep best 15.2659 dB、elapsed 3878.16 sec。
- 1024/16/131k seed0 raw: 完走。elapsed 5212.77 sec、max RSS 4,483,644KB、標準camera best `front_x`, PSNR 14.4012 dB、MAE 0.079861。camera sweep best `front_x`, FOV 42, distance 1.6, PSNR 15.22595 dB、MAE 0.073392。

解釈:

- step数増加で品質は上がる傾向だが、CPUではほぼ線形に時間が増える。
- 低リソース実装の本丸は、stepsを減らしてごまかすことではなく、同じstepsで低メモリ・低オーバーヘッドにすること。

### `--guidance-scale`

Classifier-free guidance (CFG) の強さです。

主な値:

- `1.0`: guidanceなしに近い。最初の完走確認で使用。
- `3.0`: 本検証の標準。

機能:

```text
pred = guidance_scale * pred_cond - (guidance_scale - 1) * pred_uncond
```

- 条件画像に沿わせる強さを決める。
- 値を変えるとraw TripoSplat結果が変わる。

注意:

- guidanceを下げれば計算が軽くなるわけではない。CFGを使う限りcond/uncond forwardが必要。
- guidance値を変えて品質・速度を比較することは有用だが、同じ結果を低リソース化した証拠にはならない。

### `--shift`

Flow samplerのtime scheduleを歪ませる係数です。

現在の標準値:

- `3.0`

schedule:

```python
t_seq = shift * linspace(1, 0, steps + 1) / (1 + (shift - 1) * linspace(1, 0, steps + 1))
```

機能:

- どの時間領域に積分stepを多く割くかを変える。
- latentのdenoising軌道が変わるため、raw出力も変わる。

同等性上の注意:

- `shift` はアルゴリズム設定であり、低リソース置換では固定する。

### `--seed`

乱数seedです。

探索値:

- CPU: 主に `0`, 以前は `42`。
- GPU: `0`, `1`, `2`, `3`, `4`, `42`。

機能:

- VAE latent samplingやflow noiseの初期値を決める。
- 同じ入力・同じ設定でもseedでraw PLYが変わる。

観測:

- GPU 1024/20/262k rawでは seed0 がlocal renderer上もっとも良かった。
- CPU 512/8/32k stagedでも seed0 が seed42 より良かった。

注意:

- seed探索はTripoSplat設定探索であり、低リソース同等化そのものではない。
- 低リソース実装を検証するときは、reference/candidateでseedを必ず固定する。

### `--model-dtype`

モデルをCPU上でどのdtypeで持つかです。

探索値:

- `bfloat16`: 現在の標準。CPUメモリを抑える。
- `float32`: 基準としては安全だが、8GB級CPU環境ではOOMしやすい。

機能:

- モデル重み・中間activationのメモリ量に効く。
- 数値結果に差が出る可能性があるため、厳密同等性ではdtypeも固定する。

現在の運用:

- CPU推論モデルは `bfloat16`。
- export時は必要に応じてfloat32/numpyへ戻す。

## 低リソース実装パラメータ

### `--lowmem-export`

本家 `Gaussian.save_ply()` / `save_splat()` の代わりに、streaming exportを使うフラグです。

使用スクリプト:

- `triposplat_lowmem_export.py`

機能:

- PLY/SPLATの最終巨大bufferを一括で作らず、chunk単位で書き出す。
- TripoSplat推論結果であるGaussian属性は変更しない。

検証:

- 合成Gaussianで本家PLY/SPLATとbyte完全一致。
- これは現在、受け入れ済みの低リソース置換境界です。

### `--export-chunk-size`

streaming exportのchunkサイズです。

使用値:

- `8192`: 512/16/32k run。
- `16384`: 1024/12・1024/16/131k run。
- `32768`: 262k検証など。

機能:

- 一度にnumpy化・packするGaussian数を決める。
- 小さすぎるとI/O loop回数が増え、大きすぎると一時メモリが増える。

同等性:

- 正しく実装されていれば、chunk sizeはPLY/SPLAT内容を変えない。
- 変わる場合は実装バグとして扱う。

### `--lowmem-sampler`

本家samplerと同じ数式を維持しつつ、CFG計算などの実装を低リソース化する候補です。

関連スクリプト:

- `triposplat_lowmem_sampler.py`
- `test_lowmem_sampler_equivalence.py`

機能:

- conditional/unconditionalのforwardをbatch=2化する、または公式と同じ2callを明示する。
- sampler wrapperのメモリ・呼び出し構造を調整する。

検証状況:

- 合成モデルでは本家samplerと最大差0で一致。
- 実モデルでは `1024 / 20 steps / 262144 gaussians / seed0` を `--lowmem-sampler --separate-cfg --lowmem-export` で完走確認済み。これは公式samplerとのraw同等性証明ではなく、8GB quota下で同じTripoSplat設定を実行できることの確認。次は同設定のGPU rawまたは公式sampler CPU小設定とのPLY属性比較が必要。

注意:

- sampler内部はflow model計算そのものに近いので、ここを置き換える場合はlatentまたはraw PLY比較が必須。

### `--separate-cfg`

`--lowmem-sampler` 使用時に、CFGをbatch=2化せず、本家と同じcond/uncond分離forwardで実行するためのフラグです。

機能:

- 低リソースsampler wrapperの中でも、本家に近い呼び出し形を保つ。
- batch=2による微小な数値差やメモリ増加を切り分ける。

使いどころ:

- `lowmem_sampler` の結果差が出た場合、batch化が原因かどうかを切り分ける。

## 評価rendererパラメータ

### `--views`

raw PLYをどの方向から見るかのプリセットです。

標準:

```text
front_z, back_z, front_x, back_x, front_y, back_y
```

機能:

- PLY座標系と入力画像視点の対応を探る。
- local rendererのbest viewを選ぶ。

注意:

- best viewが変わってもraw TripoSplat結果は変わらない。
- GPU/CPU比較でviewが違う場合、renderer/camera解釈の差も疑う必要がある。

### `--fov-deg`

評価rendererのカメラ視野角です。

探索値:

- 標準: `38`
- sweep: `24, 30, 36, 42, 50, 60, 70` など。

機能:

- 投影の拡大率と見え方を変える。
- official SparkJS viewerの見え方に近づけるための診断軸。

注意:

- FOV tuningはraw PLYを変更しない。
- PSNR向上は評価カメラ適合であり、TripoSplat推論品質改善とは別。

### `--distance-scale`

評価rendererのカメラ距離係数です。

探索値:

- 標準: `2.8`
- sweep: `1.2, 1.6, 2.0, 2.4, 2.8, 3.4, 4.2, 5.0` など。

機能:

- Gaussian cloud全体が画面へどう収まるかを変える。
- FOVと組み合わせて、入力画像とraw PLYの見え方を合わせる。

### `--max-radius-px`, `--min-radius-px`, `--radius-clip`

ref-style rendererで、各Gaussianを画面上に何px程度の小パッチとして描くかを制御します。

機能:

- `Sigma_2D = J Sigma_3D J^T` で得た楕円Gaussianの画面半径をclipする。
- 大きいほどぼかし・被覆が増え、CPU描画時間も増える。

使いどころ:

- 評価rendererが点状すぎる、または広がりすぎる場合の診断。
- LS補正では局所patchの範囲にも影響する。

注意:

- renderer radius tuningはraw PLYを変えない。
- LS後処理と組み合わせた場合は、主結果ではなく診断・補助結果として扱う。

### `--alpha-scale`

rendererまたは後処理でopacityに掛ける係数です。

機能:

- 同一視点での密度・透過合成の強さを調整する。
- opacity bakeするとPLY内のopacity値自体を書き換える。

注意:

- raw TripoSplat評価では `alpha_scale=1.0` が基本。
- `alpha_scale` tuningやopacity bakeは、post-export補正であり、低リソース同等化の証拠ではない。

## LS色補正・opacity bake系パラメータ

これらは過去に有用な診断として試しましたが、現在の主目的である raw TripoSplat同等化では主結果にしません。

### `--iterations`

固定geometryのまま、Gaussian色を座標降下LSで更新する反復回数です。

- 少ないほど速い。
- 多いほど同一視点PSNRは上がりやすいが、過学習しやすい。

### `--damping`

LS更新量の緩和係数です。

- 大きいほど更新が強い。
- 不安定な場合は小さくする。

### `--ridge`

閉形式LS更新時の正則化です。

- 小さすぎると局所patchで不安定化しやすい。
- 大きすぎると色補正が弱くなる。

### `--init`

LS開始時の色初期化です。

- `current`: TripoSplat出力の現在色から開始。
- `projected`: Gaussian中心を入力画像へ投影して、その色から開始。

位置づけ:

- これらは見た目の補正であり、TripoSplat推論モデルの同等化ではありません。

## Viewerパラメータ

### `--point-scale`

単一HTML viewerで、各Gaussian/pointをどの大きさで描くかの係数です。

- 小さいほどシャープだが点穴が出やすい。
- 大きいほど連続的だが低解像・ぼやけに見えやすい。

WebGL版では現在 `pointScale=10.5` 付近を代表設定にしています。

### `--default-view`

HTMLを開いたときの初期カメラ方向です。

- 入力画像視点に近い方向を選ぶ。
- 直近の1024/12/131kでは `front_x` がlocal renderer上のbest view。

### HiDPI / DPR

WebGL版viewerではdevice pixel ratioを上げ、Canvas版より低解像に見えないようにしています。

- static: 最低DPR 2.0、最大4.0。
- dragging: 最低DPR 1.5、最大2.5。

### Depth sort / depth test

- depth sort: blended Gaussianを静止時に奥行き順へ並べる。チラつき低減。
- depth test: occlusionは強くなるが、半透明Gaussianでは見え方が硬くなる場合がある。

現在のWebGL viewerでは、ドラッグ中は直前のsort順を使い、停止後に再sortします。

## 評価指標

### PSNR

入力画像とレンダリング結果の二乗誤差から計算する指標です。

- 高いほど同一視点再現が近い。
- camera/FOV/renderer設定に強く依存する。
- raw PLY品質の補助指標であり、本家viewer一致の絶対証拠ではない。

### MAE / MSE

- MAE: 平均絶対誤差。色の平均的なズレを見る。
- MSE: 平均二乗誤差。PSNRの元になる。

### raw PLY/SPLAT比較

低リソース同等化の主ゲートです。

使用スクリプト:

- `compare_triposplat_ply.py`
- `compare_triposplat_splat.py`

比較対象:

- `xyz`
- `f_dc_*`
- `opacity`
- `scale_*`
- `rot_*`
- その他PLY属性

同等化では、見た目ではなく raw属性比較を合格条件にします。

## 現在の探索状況まとめ

GPU raw基準:

- `gpu_encoded_rmbg_1024_s20_g3_seed0_n262k`
- 1024 / 20 steps / 262144 Gaussians / seed0
- elapsed 29.98 sec
- local renderer best: `front_x`, PSNR 15.3869 dB

CPU raw代表:

- `cpu_staged_rmbg_512_s16_g3_seed0_lowmem_export`
- 512 / 16 steps / 32768 Gaussians / seed0
- elapsed 3381.64 sec
- best: `back_z`, PSNR 15.0998 dB

CPU 1024 raw代表:

- `cpu_staged_rmbg_1024_s20_g3_seed0_n262k_lowmem_separatecfg_export`
- 1024 / 20 steps / 262144 Gaussians / seed0
- elapsed 8706.30 sec
- run中の `oom` / `oom_kill` 増加なし。ただしcgroup peakは8GB上限到達済み。
- standard camera best: `front_x`, PSNR 14.8854 dB, MAE 0.075476

比較用CPU 1024 raw:

- `cpu_staged_rmbg_1024_s12_g3_seed0_n131k_lowmem_export`: elapsed 3878.16 sec、standard camera best `front_x`, PSNR 14.4785 dB、camera sweep best 15.2659 dB。
- `cpu_staged_rmbg_1024_s16_g3_seed0_n131k_lowmem_export`: elapsed 5212.77 sec、max RSS 4,483,644KB、standard camera best `front_x`, PSNR 14.4012 dB、camera sweep best 15.22595 dB。

## 次に優先する探索

1. `1024/20/262k` CPU rawは完走済みなので、同条件GPU rawとのPLY属性差分、または小設定で公式sampler/lowmem samplerのraw一致を確認する。
2. camera sweepは評価補助として実施するが、採否はraw PLY/SPLAT比較を優先する。
3. 8GB quotaでは余裕が少ないため、次の実装置換は sampler/decoder/export のメモリ寿命短縮とrun単位のcgroup監査を継続する。
4. LS補正やviewer見た目は診断に留め、TripoSplat同等化の判定には使わない。

## 2026-07-05 CPU direct-encoded 1024/20/262k 追記

`cpu_encoded_rmbg_1024_s20_g3_seed0_n262k_lowmem_separatecfg_export` を追加実行した。これは `prepared_rgb.webp` を直接 `encode_image()` に渡す経路で、GPU seed0 runと `preprocessed_image.webp` がbyte同一になった。

結果:

- elapsed: `8313.14 sec`
- run中の `oom` / `oom_kill` 増加: なし
- standard ref renderer best: `front_z`, PSNR `13.2965 dB`, MAE `0.111181`
- quick camera sweep best: `front_z`, FOV `60`, distance_scale `1.6`, PSNR `13.5985 dB`, MAE `0.106621`
- GPU seed0との差: 入力画像は完全一致するが、raw PLY属性は大きく異なる。

これにより、前回の `cpu_staged_rmbg_1024_s20_g3_seed0_n262k_lowmem_separatecfg_export` は8GB quotaで完走した実績としては有効だが、`prepared_rgba.png` を `pipe.run()` に渡したため前処理が二重に走り、GPU基準と同じ入力条件の品質比較には使わない。

次の焦点は、CPU/CUDAで同じ `seed=0` でも初期noiseが一致しない問題を解消すること。低リソース同等化の検証には、外部noise注入またはdevice-independent noise生成を導入し、同一noise・同一入力・同一samplerでraw差分を見る必要がある。
