# TripoSplatモデル構造・データ構造・役割

## 目的

この文書は、TripoSplat本家実装の推論モデル構造、主要データ構造、各段の役割を整理する。目的は、CPU低リソース実装を作るときに「何を同等に保つ必要があるか」と「どこを実装上置き換えてよいか」を明確にすること。

ここでの「同等」は、見た目を後処理で合わせることではない。同じ入力、同じcheckpoint、同じseed、同じsampler設定から、同じraw Gaussian出力、つまり `output.ply` / `output.splat` を得ることを指す。

## 生成の流れを読み物として見る

この節では、TripoSplatが「1枚の画像から、なぜ3D Gaussianの集合を出せるのか」を、実装の順番に沿って説明する。厳密な式は別文書 `triposplat_model_pipeline_math_analysis_ja.md` に分け、この文書では各段が出力へ与える効果を追う。

最初に、この節で使う記号を明示する。

| 記号 | 読み方 | 意味 |
| --- | --- | --- |
| `I` | アイ | 入力画像。今回なら動画から切り出した1枚のフレームや本家サンプル画像。 |
| `I_p` | アイ・ピー | 前処理後の画像。背景除去、crop、resize、黒背景合成を済ませ、encoderが実際に見る正方形RGB画像。 |
| `C` | シー | 画像条件特徴。TripoSplatが「この画像は何を写しているか」を保持する情報の束。 |
| `F_1` | エフ・ワン | `C` の一部。DINOv3が出す高次視覚特徴。形、部位、構造の手がかりを持つ。 |
| `F_2` | エフ・ツー | `C` の一部。VAE encoderが出す画像latent系特徴。局所的な色、質感、配置の手がかりを持つ。 |
| `X` | エックス | flow samplerが更新する生成途中の状態。latent tokenとcamera tokenを含む。 |
| `Z` | ゼット | `X` のうち、3D復号へ渡すlatent token列。 |
| `M` | エム | `X` のうち、camera token。生成中に推定されるカメラ関連の小さな状態。 |
| `K` | ケー | sampler step数。`steps=16` なら `K=16`。 |
| `t` | ティー | flow samplerの時刻。1から0へ向かって進む生成の進行度。 |
| `v` | ブイ | flow modelが予測するvelocity。現在の生成状態をどちらへ動かすべきかを表す更新方向。 |
| `s_cfg` | エス・シーエフジー | guidance scale。画像条件にどれだけ強く従わせるかを決める係数。 |
| `N_g` | エヌ・ジー | 出力Gaussian数。例: 32768、131072、262144。 |
| `N_p` | エヌ・ピー | decoderが内部で扱うoctree point数。TripoSplatでは概ね `N_g / 32`。 |
| `G` | ジー | 最終的な3D Gaussian集合。PLY/SPLATへ保存される実体。 |
| `g_i` | ジー・アイ | `G` に含まれるi番目のGaussian。位置、色、不透明度、スケール、回転を持つ。 |
| `x_i` | エックス・アイ | `g_i` の3D位置。 |
| `f_dc_i` | エフ・ディーシー・アイ | `g_i` の色を表すSH DC係数。通常のRGBへ戻す元になる値。 |
| `alpha_i` | アルファ・アイ | `g_i` の不透明度。画面へ描いたときの濃さや透け方に効く。 |
| `s_i` | エス・アイ | `g_i` の3軸スケール。Gaussianの広がり方に効く。 |
| `q_i` | キュー・アイ | `g_i` の回転quaternion。異方性Gaussianの向きを決める。 |

### 1. 入力画像 `I` を、推論しやすい画像 `I_p` に整える

TripoSplatは任意の写真をそのまま3Dへ変換するのではなく、まず入力画像 `I` を「対象物が中央にあり、背景が整理された正方形画像」へ変換する。この前処理後の画像を `I_p` と呼ぶ。`I_p` の `p` は prepared、つまり準備済みという意味で使っている。

この段の効果は大きい。TripoSplatは単眼モデルなので、画像内のどれを3D化すべきかを前処理に強く依存する。背景が残り、複数人物や机、壁が同時に入ると、モデルは「1つの物体」を復元するのではなく、画面全体からもっともありそうな3D Gaussian配置を作ろうとする。その結果、正面はそれらしく見えても、角度を変えると奥行きが破綻しやすい。

ここでの `I_p` は、後段encoderが実際に見た画像である。したがって低リソース版を比較するときは、元の `I` だけでなく `I_p` が同じかどうかも重要になる。`I_p` が変われば、以降の全ての特徴 `C`、latent `Z`、Gaussian集合 `G` が変わる。

### 2. `I_p` から条件特徴 `C` を作る

次に、画像 `I_p` を2種類のencoderへ通す。出力される条件特徴の束を `C` と呼ぶ。`C` は condition の意味で、flow modelに「この画像に沿って生成しなさい」と伝える情報である。

`C` は2つの要素を持つ。

- `F_1`: DINOv3の特徴。これは画像の意味的・構造的な読み取りに強い。輪郭、部位、物体らしさ、全体の配置など、3D形状を推定するための高次の手がかりになる。
- `F_2`: VAE encoderの特徴。これは画像の局所的な見た目に近い情報を持つ。色、明暗、テクスチャ、細かな配置など、最終Gaussianのappearanceに効く手がかりになる。

この2つを合わせた `C = {F_1, F_2}` が、TripoSplatにとっての「入力画像の理解結果」になる。重要なのは、ここではまだ3D Gaussianは出ていないという点である。`C` はあくまで「画像を条件として圧縮した情報」であり、3D構造そのものではない。

低リソース実装では、この `C` を一度保存してencoderを解放することができる。これは推論の意味を変えない。`C` が同じなら、その後のflow samplerへ渡る条件も同じだからである。

### 3. 初期状態 `X_0` は乱数から始まる

TripoSplatは、条件特徴 `C` から直接Gaussianを一発で出すのではない。まず乱数から生成途中状態 `X_0` を作る。`X_0` の添字 `0` は、samplerの最初の状態という意味である。

`X_0` は2つの成分を持つ。

- `Z_0`: latent token列。`Z` は最終的にdecoderへ渡される潜在表現で、8192個のtokenを持つ。
- `M_0`: camera token。`M` はcameraに関係する小さな状態で、5次元のtokenとして扱われる。

この初期乱数は `seed` で決まる。つまり同じ `I_p` と同じ設定でも、`seed=0` と `seed=42` では `X_0` が変わり、その後の `Z` と `G` も変わる。今回の検証でseed0がよく見えたのは、後処理の問題ではなく、この初期状態からたどる生成経路が違ったためである。

### 4. Flow modelは `X` を画像条件 `C` に合う方向へ少しずつ動かす

生成本体は `LatentSeqMMFlowModel` である。このモデルは、現在の状態 `X` と条件特徴 `C` を見て、「今の `X` をどちらへ動かせば画像に合うlatentへ近づくか」をvelocity `v` として予測する。

ここで `v` は速度というより、生成状態の更新方向である。たとえば `v_Z` はlatent token `Z` の更新方向、`v_M` はcamera token `M` の更新方向を表す。

samplerは時刻 `t` を1から0へ進めながら、この更新を `K` 回繰り返す。`K` はstep数で、`steps=16` なら16回、`steps=20` なら20回である。stepが多いほど細かく更新できるが、CPUではほぼその分だけ時間が伸びる。

1回の更新は概念的に次のように読む。

```text
現在の状態 X_i を見る
画像条件 C を見る
flow modelが更新方向 v_i を予測する
X_i を v_i の方向に少し動かし、次の状態 X_{i+1} にする
```

ここで `X_i` の `i` はi番目のstepの状態、`v_i` はそのstepで予測された更新方向、`X_{i+1}` は更新後の状態である。

### 5. Guidance `s_cfg` は「画像条件へ従わせる強さ」を調整する

TripoSplatのsamplerはClassifier-Free Guidanceを使う。ここで使う `s_cfg` は guidance scale のことで、今回の標準設定では `s_cfg=3.0` である。

考え方は次の通りである。

- 条件ありの予測は、画像特徴 `C` を見た更新方向である。
- 条件なしの予測は、空の条件、つまり `C` をゼロにしたときの更新方向である。
- `s_cfg` を使って、条件あり方向を強め、条件なし方向を差し引く。

読み物として言えば、`s_cfg` は「この画像らしさをどれだけ強く押し出すか」を決めるつまみである。高くすると入力画像に沿いやすくなる一方、色や形が強く出すぎる可能性もある。低くすると計算結果は穏やかになるが、入力画像への追従が弱くなる。

CPU時間の面では、`s_cfg > 1` のとき、各stepで条件ありと条件なしの2回のflow model forwardが必要になる。したがって `steps=20` かつ `guidance_scale=3.0` は、実質40回の重いTransformer forwardに相当する。

### 6. 最終latent `Z_K` をdecoderへ渡す

`K` 回のflow更新が終わると、最終状態 `X_K` が得られる。`X_K` の中のlatent成分を `Z_K` と呼ぶ。`K` は最後のstep番号であり、`Z_K` は「flow samplerが画像条件に合わせて作った最終latent」である。

この `Z_K` はまだGaussianではない。`Z_K` は3D Gaussianを作るための圧縮表現であり、次のdecoderがこれを具体的な3D点とGaussian属性へ展開する。

### 7. Decoderは `Z_K` からoctree pointを出し、その周りにGaussianを配置する

`OctreeGaussianDecoder` は2段構成である。

最初の段は `OctreeProbabilityFixedlenDecoder` で、最終latent `Z_K` から3D空間内のpoint候補を出す。このpoint数を `N_p` と呼ぶ。`N_p` は number of points の意味である。

次の段は `ElasticGaussianFixedlenDecoder` で、各pointの周りに複数のGaussianを置く。TripoSplat本家設定では1 pointあたり32個のGaussianを作る。したがって、出力Gaussian数 `N_g` と内部point数 `N_p` の関係はおおむね次の通りである。

```text
N_p = N_g / 32
```

ここで `N_g` は number of gaussians の意味である。例として、`N_g=131072` なら `N_p=4096` になる。

この設計の効果は、latentから直接全Gaussianを独立に出すのではなく、まず粗い3D point構造を作り、その各pointの周囲へ32個の小さなGaussianを配置することにある。これにより、形状の骨格と局所的な表面表現を分けて扱える。

### 8. 1つのGaussian `g_i` が持つもの

最終出力 `G` は、Gaussian `g_i` の集合である。`G` は全体の集合、`g_i` はi番目のGaussianを意味する。

各 `g_i` は次の属性を持つ。

- `x_i`: 3D位置。Gaussianが空間のどこにあるかを決める。
- `f_dc_i`: 色の係数。SH DC項で、通常のRGBへ戻す元になる。
- `alpha_i`: 不透明度。rendererで描くときの濃さと透け方を決める。
- `s_i`: 3方向のスケール。Gaussianがどれだけ広がるかを決める。
- `q_i`: 回転quaternion。`s_i` の3方向の向きを決める。

この5種類の属性が揃うと、1つの小さな楕円体状の色付きGaussianとして描ける。大量の `g_i` を重ねて描くことで、画像から推定された3D形状と見た目が表現される。

PLYでは、これらの属性が `x,y,z`、`f_dc_*`、`opacity`、`scale_*`、`rot_*` として保存される。SPLATでは、viewer向けにposition、scale、RGBA、rotationを32 bytes/recordでpackする。


### 9. 入力画像に写っていない部分はどう補われるのか

単眼のTripoSplatは、入力画像 `I_p` に写っていない背面や側面を、画像の外から直接見ているわけではない。見えない部分を作る仕組みは、モデルの重みの中に入っている3D priorと、入力画像から得た条件特徴 `C` の整合で説明できる。

まず、DINOv3特徴 `F_1` は、画像の中にある形や部位の関係を読む。たとえば、輪郭、左右対称らしさ、頭部や胴体のような部位、机や箱のような面構造などである。ここで重要なのは、`F_1` が単なる画素の色の集まりではなく、「この見え方なら、どのような立体カテゴリや部品配置に近いか」という高次の手がかりを持つ点である。

次に、VAE特徴 `F_2` は、見えている面の色、明暗、局所的な質感、輪郭付近の細部を条件として渡す。`F_2` は「入力画像に実際に写っているもの」を最終出力へ貼り戻すための手がかりになる。つまり、`F_1` が形の読み取りを支え、`F_2` が見た目の細部を支える。

flow samplerは、この `C = {F_1, F_2}` を見ながら、乱数から始まった状態 `X_0` を、学習済みの3D latent分布の中で「この画像に合いそうな状態」へ動かしていく。ここでいう学習済み分布とは、過去の訓練でモデルが覚えた「こういう正面なら、側面や背面はこう続くことが多い」という統計的な規則である。したがって、不可視部は入力画像の画素から復元されるというより、入力画像が示す手がかりに合う3D候補を、モデルの内部priorから選ぶことで補われる。

この補完は、1回の処理で突然起きるわけではない。各sampler stepでは、現在のlatent `X_i` が「画像条件に合う方向」へ少しずつ動く。最初は乱数なので、形も見た目も定まっていない。stepを重ねると、見えている輪郭や色に合う方向へ前面が固まり、それと同時に、学習済みpriorが許す範囲で側面や背面も一貫した形になるようにlatent全体が調整される。

decoderは、その最終latent `Z_K` からoctree pointとGaussian属性を出す。このとき出るGaussianは、入力画像に写った前面だけに置かれるのではない。`Z_K` が3D空間全体の圧縮表現なので、decoderは見えている面、輪郭の奥、側面らしい場所、背面らしい場所にもGaussianを配置する。これが、単眼入力から視点移動可能な3D表現が出る理由である。

ただし、これは観測ではなく推論である。入力画像が本家サンプルのように、単体物体が中央にあり、背景が整理され、学習分布に近い場合は、モデルのpriorが働きやすい。見えない背面や側面も「よくある続き方」として比較的一貫して補われる。

一方、今回の番組フレームのように、複数人物、机、背景、画面内の別要素が同時に入る場合、条件特徴 `C` は1つの物体を表すきれいな手がかりになりにくい。モデルはそれでも1つのGaussian集合 `G` を作るため、正面からは入力に似ていても、側面へ回すと奥行きや遮蔽関係が破綻しやすい。これはrendererだけの問題ではなく、不可視部を決めるpriorが曖昧な条件から複数の可能性を1つに畳み込んでいるためである。

seedの影響もここに出る。同じ画像条件 `C` でも、初期noise `X_0` が違うと、samplerがたどる経路が変わる。見えている前面は条件で強く縛られるが、見えていない裏側や奥行きは条件が弱いため、seedによる差が出やすい。GPU検証でseedごとの見え方が変わったのは、この不可視部の選び方が揺れているためと読める。

低リソース実装で重要なのは、この不可視部推論の仕組みを変えないことである。`steps`、`guidance_scale`、`seed`、`num_gaussians`、encoder特徴、flow model、decoderを変えると、見えない部分の補われ方も変わる。したがって、低リソース化では、見えない部分を別手法で補うのではなく、同じ `C -> X_K -> G` の写像を、メモリの少ない実装で再現する必要がある。

### 10. なぜ正面は良くても角度を変えると破綻するのか

TripoSplatは単眼モデルなので、入力画像 `I_p` に写っていない背面や側面を直接観測していない。モデルは学習済みpriorに基づいて「この画像なら裏側や奥行きはこうなっていそうだ」と推定する。

この推定がうまく働くのは、入力が学習分布に近い場合である。たとえば本家サンプルのように、単体物体が中央にあり、背景がきれいに抜けている画像では、`C` が対象物の形状を比較的一貫して表しやすい。

一方、評価対象の複雑なフレームのように、複数人物、机、背景、番組画面全体が含まれると、`C` は「1つの物体」ではなく複数の要素を含んだ条件になる。その状態から `Z_K` と `G` を作るため、正面viewの一部は入力に近くても、横や斜めから見たときに奥行きや遮蔽関係が破綻しやすい。

これは単にrendererの問題ではない。renderer cameraやFOVで見え方は変わるが、raw Gaussian集合 `G` に十分な奥行き整合性が無ければ、大きな視点変更には耐えない。

### 11. どこが低リソース化でき、どこを変えると別物になるか

低リソース化で触ってよいのは、原則として `G` を変えない実装境界である。

安全性が高い例:

- `G` をPLY/SPLATへ書くexportをstreaming化する。
- encoder、flow、decoderを段階ロードして、使い終わったモデルを解放する。
- 同じ入力 `I_p`、同じ条件 `C`、同じseed、同じsampler設定を保ったまま、中間結果を保存する。

危険な例:

- `steps` を減らして同等と主張する。
- `guidance_scale` を変えて同等と主張する。
- `num_gaussians` を変えて同等と主張する。
- LS色補正やopacity bakeで見た目を合わせ、それをraw TripoSplat品質として扱う。
- 2.5D image-plane proxyをTripoSplatの代替として扱う。

`ref` は、このうち安全性が高い実装技法の参考になる。たとえば、小パッチ処理、streaming、bounded memory、評価rendererの構成は参考になる。しかし、`ref` の最適化式でTripoSplatの `C -> X_K -> G` という推論写像を置き換えると、目的が別物になる。

### 12. 数式編との関係

この節は、TripoSplatがどのような効果の連鎖でoutputを作るかを読むための説明である。各変数の厳密なshape、更新式、Gaussian属性のactivation、PLY/SPLAT保存時の変換は、補助資料 `triposplat_model_pipeline_math_analysis_ja.md` にまとめる。

読み物としては、次の流れを押さえればよい。

```text
入力画像 I
  -> 前処理後画像 I_p
  -> 条件特徴 C = {F_1, F_2}
  -> flow samplerで更新された最終latent Z_K
  -> decoderが作るGaussian集合 G
  -> raw output.ply / output.splat
```

低リソース実装で守るべき対象は、この流れの最後に出る `G` と、それを保存したraw出力である。

## 全体パイプライン

本家の推論順は `TripoSplatPipeline.run()` に集約されている。

```text
input image
  -> preprocess_image()
  -> encode_image()
  -> sample_latent()
  -> decode_latent()
  -> Gaussian
  -> save_ply() / save_splat()
```

主要コンポーネントは以下。

| 段 | 実装 | 入力 | 出力 | 役割 |
| --- | --- | --- | --- | --- |
| 背景処理 | `BiRefNet` | RGB/RGBA画像 | RGB正方形画像 | 背景除去、crop、resize |
| 画像エンコード1 | `DinoV3ViT` | 正規化RGB | `feature1` | 画像の高次特徴 |
| 画像エンコード2 | `Flux2VAEEncoder` | RGB [-1,1] | `feature2` | VAE latent系の画像特徴 |
| 生成本体 | `LatentSeqMMFlowModel` | noise + condition | latent + camera | flow matching denoising |
| 3D復号 | `OctreeGaussianDecoder` | latent | Gaussian | octree点とGaussian属性を生成 |
| 出力 | `Gaussian.save_ply/save_splat` | Gaussian | PLY/SPLAT | raw 3DGSファイル化 |

## Component Loaders

本家は各モデルを以下のloaderで生成する。

```text
load_dinov3()      -> DinoV3ViT
load_vae_encoder() -> Flux2VAEEncoder
load_rmbg()        -> BiRefNet
load_flow_model()  -> LatentSeqMMFlowModel
load_decoder()     -> OctreeGaussianDecoder
```

公式pipelineでは、これらを一度にdeviceへ載せる。

```text
dinov3      : bfloat16
vae_encoder : bfloat16
rmbg        : float16
flow_model  : float16
decoder     : float16
```

CPU低リソース実装では、同じcheckpointと同じ演算を使ったまま、モデルを段階ロードしてよい。

```text
load encoder -> encode -> save cond -> unload encoder
load flow    -> sample -> save latent -> unload flow
load decoder -> decode -> export -> unload decoder
```

この分割は演算結果を変えないため、低リソース化の有力な境界である。

## Preprocess

`preprocess_image()` は画像を正方形canvasへ整える。

処理:

1. 入力をPIL imageへ変換
2. 短辺がcanvas sizeになるようresize
3. 実alphaが無ければ `BiRefNet.remove_background()` を実行
4. alphaを必要に応じてerode
5. alpha bboxから対象中心をcrop
6. canvas sizeへresize
7. 黒背景RGBへcomposite

今回の低リソース実験では、前処理済み画像を使うことでBiRefNetの再実行を避けている。

## Image Encoding Data

`encode_image()` の出力はdict。

```python
{
  "feature1": dinov3_feat,
  "feature2": vae_feat,
}
```

`feature1`:

- 由来: `DinoV3ViT`
- 入力: ImageNet mean/stdで正規化したRGB
- 役割: 条件画像の高次視覚特徴
- 後段flow modelでは `cond_channels=1280` として扱われる

`feature2`:

- 由来: `Flux2VAEEncoder`
- 入力: RGBを `[-1, 1]` へ変換
- 役割: VAE由来の補助条件
- 先頭に5 zero tokenをpadして、DINO側のtoken長に合わせる
- 後段flow modelでは `cond2_channels=128` として扱われる

低リソース化では、この `cond` を保存してencoderを解放できる。

## Flow Matching Model

生成本体は `LatentSeqMMFlowModel`。

主な設定:

```text
q_token_length       = 8192
in_channels          = 16
cam_channels         = 5
out_channels         = 16
model_channels       = 1024
cond_channels        = 1280
cond2_channels       = 128
num_refiner_blocks   = 2
num_blocks           = 24
num_heads            = 16
mlp_ratio            = 4
qk_rms_norm          = true
share_mod            = true
use_shift_table      = true
```

入力state:

```python
noise = {
  "latent": torch.randn(1, 8192, 16),
  "camera": torch.randn(1, 1, 5),
}
```

condition:

```python
cond = {
  "feature1": ...,
  "feature2": ...,
}
```

内部の大まかな流れ:

1. `latent` を `input_layer` で1024次元へproject
2. `feature1` を `cond_embedder` で1024次元へproject
3. `feature2` を `cond_embedder2` で1024次元へprojectし、feature1側に加算
4. timestepを `TimestepEmbedder` で埋め込み
5. noise側を `noise_refiner` で処理
6. condition側を `context_refiner` で処理
7. camera tokenを `cam_refiner` で処理
8. latent token、condition token、camera tokenを結合
9. 24個の `UnifiedTransformerBlock` でfull attention処理
10. latent部分を `out_layer` へ通して velocityを出す
11. camera部分を `cam_out_layer` へ通してcamera velocityを出す

ここがCPU実行時間の主因である。

## Euler CFG Sampler

`FlowEulerCfgSampler.sample()` はflow modelをEuler積分で回す。

schedule:

```python
t_seq = shift * linspace(1, 0, steps + 1) / (1 + (shift - 1) * linspace(1, 0, steps + 1))
```

更新:

```python
sample = sample - pred_v * dt
```

guidance `> 1` の場合、各stepで2つの推論を行う。

```python
pred_cond = model(x_t, t, cond)
pred_uncond = model(x_t, t, neg_cond)
pred = guidance_scale * pred_cond - (guidance_scale - 1) * pred_uncond
```

今回の設定 `guidance_scale=3.0` では、20 stepsなら実質40回のflow model forward相当になる。

低リソース候補:

- conditional/unconditionalをbatch=2にまとめる
- modelを変えずにsampler wrapperだけ置換する
- 合成モデルでは `FlowEulerCfgSamplerLowmem` が本家samplerと最大差0で一致済み

注意:

- batch=2化は数式上は同じだが、実モデルで完全一致を主張するには小設定でraw比較が必要
- guidanceを下げると速くなるが、これはアルゴリズム設定変更であり、同等実装ではない

## Decoder

`OctreeGaussianDecoder` は2段構成。

```text
OctreeGaussianDecoder
  -> OctreeProbabilityFixedlenDecoder
  -> ElasticGaussianFixedlenDecoder
```

### OctreeProbabilityFixedlenDecoder

役割:

- latentからoctree/point候補をsampleする
- `num_gaussians` から必要なdecoder token数を決める

関係:

```python
num_decoder_tokens = num_gaussians // gaussians_per_point
```

`gaussians_per_point` はGS decoderの `rep_config["num_gaussians"]` で、今回の本家設定では32。

例:

```text
32768 gaussians  -> 1024 decoder tokens
262144 gaussians -> 8192 decoder tokens
```

### ElasticGaussianFixedlenDecoder

役割:

- pointごとに複数Gaussianを生成する
- 各Gaussianの属性を出力する

出力layout:

```text
_xyz         : (ng, 3)
_features_dc : (ng, 1, 3)
_scaling     : (ng, 3)
_rotation    : (ng, 4)
_opacity     : (ng, 1)
_offset_scale: (ng, 1)  # learned offset scale使用時
```

本家設定:

```text
model_channels = 1024
cond_channels  = 16
num_blocks     = 16
num_heads      = 16
attn_mode      = full
num_gaussians per point = 32
scaling_activation = softplus
opacity_bias = 0.1
scaling_bias = 0.004
```

低リソース化の次候補はここである。

理由:

- decoder tokenごとにGaussianを生成するためchunk化しやすい可能性がある
- ただしfull attention blockを持つため、単純にtokenを分割すると結果が変わる可能性がある
- 結果同等にするなら、attention依存範囲を保ったchunkingか、出力projection以降だけのstreaming化から始めるべき

## Gaussian Data Structure

`Gaussian` は最終的な3DGS属性を保持する。

主な保存値:

```text
xyz          : world position
features_dc : SH DC color
opacity     : sigmoid済みopacity
scaling     : activation済みscale
rotation    : quaternion
```

PLY保存時の属性:

```text
x, y, z
nx, ny, nz
f_dc_0, f_dc_1, f_dc_2
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
```

注意:

- PLY内の `opacity` はinverse sigmoid後の値
- PLY内の `scale_*` は `log(get_scaling)`
- `.splat` はposition、scale、RGBA、rotationを32 bytes/recordでpackする
- 本家 `save_ply/save_splat` はGPU tensorを `detach().cpu().numpy()` へ落として保存する

## 低リソース化で触ってよい境界

同等性を保ちやすい順:

1. **export boundary**
   - 実施済み
   - `triposplat_lowmem_export.py`
   - 合成Gaussianで本家PLY/SPLATとbyte完全一致

2. **pipeline staging**
   - encoder、flow、decoderを段階ロード/解放
   - `cond` と `latent` を保存して中間比較可能
   - 演算は変えないため安全

3. **sampler wrapper**
   - CFGの2forwardをbatch=2にまとめる
   - 合成モデルでは同等性確認済み
   - 実モデルでは小設定raw比較が必要

4. **decoder output streaming**
   - Gaussian出力をchunk単位で書く
   - full attention部分を分割すると結果が変わるので注意

5. **flow model内部のchunking**
   - 一番難しい
   - full attentionがあるため、単純なtoken分割は同等にならない
   - attention計算を数学的に同じままblockwise化する必要がある

## 触ると同等性が崩れるもの

以下は低リソース化ではなく、別アルゴリズムになる。

- checkpointを変える
- DINO/VAE/flow/decoderの層を削る
- token数を変える
- guidanceやstepsを変えて同等と主張する
- quantizationして許容差を決めずに同等扱いする
- ref rendererやLS color fitで見た目を合わせる
- 2.5D image-plane proxyをTripoSplat結果扱いする

## 軽量ref実装の位置づけ

`ref` はTripoSplat推論モデルの代替ではない。

参考にできるのは実装技法:

- 大きな全体テンソルを作らない
- touched patchだけ処理する
- chunk/streamingで書く
- 中間結果を保存して段階実行する
- 評価用rendererを軽量にする

参考にしてはいけないもの:

- refのGaussian最適化式でTripoSplat推論を置き換える
- refitした色やopacityをTripoSplat raw品質として扱う

## 現在の実装状況

2026-07-21時点で追加・検証済み:

- `triposplat_lowmem_export.py`
- `triposplat_lowmem_sampler.py`
- `compare_triposplat_ply.py`
- `compare_triposplat_splat.py`
- 全206 Linearのfloat32 AVX-512 backend
- exact dense/key-bias/final-cross SDPA
- GELU、SiLU、LayerNorm、RMSNorm、RoPE、RePo、CFG/Eulerのnative backend
- raw画像から262,144 Gaussian、PLY/SPLAT、renderer、viewerまでのCPU end-to-end
- 全206 LinearのNF8、residual NF8、NFR8x3 packed AVX-512 backend

strict float32 AVX-512 s20は3322.886秒、combined RMSE 2.06857e-5である。旧NFR8x3
s20は4640.813秒だった。現在の採用構成NF24 int16 + SDPA key tile 512は
3471.330秒、combined RMSE 9.37666e-5、camera RMSE 8.93055e-6で、全206 Linearを
24-bit packed weightからfallbackなしで実行する。

NF24 int16はNF8由来の不等間隔code `q0` とresidual code `q1` を
`q01 = 4*q0 + q1` としてint16へまとめ、`q2` をint8で保持する。kernelは
`(scale/1024) * (256*q01 + q2)` をSIMD register内で復号する。これはモデルの式や
学習済み機能を変えず、weight表現とGEMM実装だけを置換する。

事前pack済みcheckpointのconverterと直接loaderも実装済みである。直接loaderは
公式Flow checkpointをruntimeでloadせず、meta Linearへpacked bufferをattachする。
runtime-pack版とlatent/cameraがbit完全一致し、process-tree peak RSSを25.8%削減した。

## まとめ

TripoSplatの本体は、画像特徴を条件にしたflow matching transformerでlatent/cameraを生成し、それをoctree + Gaussian decoderで3D Gaussianへ変換する構造である。

低リソース版を同等にするには、モデルを簡略化するのではなく、同じモデルを段階実行・
streaming export・同等sampler・中間比較可能な形へ分解する必要がある。現在は
sampler、主要Flow演算、decoder/exportを含むCPU end-to-end、NF24 int16 s20、
事前pack済みweightの直接loadまで検証済みである。3600秒未満と起動時memory削減は
達成した。次の性能目標は1800秒未満だが、これは今回の同等低リソース実装の完了条件
より先の最適化である。
