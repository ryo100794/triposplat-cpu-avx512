# TripoSplatモデル構造・パイプライン数式解析

## 目的

この文書は、主文書 `triposplat_model_structure_ja.md` の「生成の流れを読み物として見る」を補完する数式編です。主文書では各段が出力へ与える効果を文章で追い、この文書では同じ流れをtensor shape、更新式、Gaussian属性のactivation、export変換として整理します。

ここでの主張は次の通りです。

- TripoSplatの主結果は、単眼画像から生成されるraw 3D Gaussian集合 `output.ply` / `output.splat` である。
- `ref` 由来のGaussian rendererやLS refitは、評価・診断・低リソース実装技法の参考であり、TripoSplat推論モデルそのものではない。
- 低リソース化で保存すべき同等性は、同じ入力、同じcheckpoint、同じseed、同じsteps/guidance/shift/gaussian countから、同じGaussian属性を得ることである。


## この数式編の読み方

先に `triposplat_model_structure_ja.md` の読み物節を読むことを前提にしています。そこで出てくる `I`、`I_p`、`C`、`F_1`、`F_2`、`X`、`Z`、`G` などの記号を、ここではより厳密に扱います。

- `I` は入力画像です。
- `I_p` は前処理後にencoderが見る画像です。
- `C` は条件特徴の束です。
- `F_1` はDINO特徴、`F_2` はVAE特徴です。
- `X` はsampler中の状態、`Z` はそのうちdecoderへ渡すlatentです。
- `G` は最終Gaussian集合です。

この文書の式は、実装を置き換えるときに「どの値が同じなら同等と言えるか」を判断するための補助資料です。


## 推論stepごとの処理内容

ここでは、TripoSplatの1回の推論を実装上のstepとして分解します。目的は、CPU低リソース版で「どのstepを置き換えてよいか」「どのstepの値が変わるとraw Gaussianが変わるか」を追えるようにすることです。

### Step 0. 入力とrun設定を固定する

入力は画像 `I`、run設定は `seed`、`canvas_size`、`steps=K`、`guidance_scale=s_cfg`、`shift=lambda`、`num_gaussians=N_g` です。

```text
run_config = (I, seed, S, K, s_cfg, lambda, N_g)
```

このstepではまだモデル計算は行いません。ただし、ここで固定した値が以降の全出力を決めます。低リソース化の比較では、まずこの設定が完全に同じであることを確認します。

### Step 1. 前処理画像 `I_p` を作る

`preprocess_image()` は入力画像 `I` をencoderが読む正方形RGB画像 `I_p` へ変換します。

処理内容:

1. 入力をRGB/RGBAとして読み込む。
2. 短辺が `canvas_size=S` になるようresizeする。
3. alphaがなければ背景除去モデルでalphaを推定する。
4. alpha bboxから対象物中心を決める。
5. 対象物が中央に来るようcropし、`S x S` へresizeする。
6. 黒背景へalpha compositeする。

出力:

```text
I_p = P(I)
I_p in R^{S x S x 3}
```

`I_p` が変わると以降の条件特徴 `C` が変わります。低リソース版で前処理済み画像を再利用するのは安全ですが、その画像が本家と同じ `I_p` である必要があります。

### Step 2. 画像条件特徴 `C` を作る

前処理画像 `I_p` をDINOv3とVAE encoderへ通し、2種類の条件特徴を作ります。

処理内容:

1. `I_p` をDINO用にImageNet mean/stdで正規化する。
2. DINOv3 ViTへ通し、高次視覚特徴 `F_1` を得る。
3. `I_p` をVAE用に `[-1,1]` へ変換する。
4. Flux2 VAE encoderへ通し、VAE特徴 `F_2` を得る。
5. DINO側とtoken長が揃うよう、VAE側の先頭へzero tokenをpadする。

出力:

```text
C = {F_1, F_2}
F_1 in R^{1 x N x 1280}
F_2 in R^{1 x N x 128}
N = 5 + (S/16)^2
```

このstepは「画像をどう理解したか」を作る段です。ここではまだ3D点やGaussianは出ません。低リソース実装では、`C` を保存してencoderを解放できます。`C` が同じなら、後段へ渡る条件は同じです。

### Step 3. 初期noise状態 `X_0` を作る

指定seedから、flow samplerの初期状態を作ります。

処理内容:

1. seedで乱数generatorを固定する。
2. latent noise `Z_0` を生成する。
3. camera noise `M_0` を生成する。
4. 2つをdict状態 `X_0` にまとめる。

出力:

```text
Z_0 ~ N(0, I),  Z_0 in R^{1 x 8192 x 16}
M_0 ~ N(0, I),  M_0 in R^{1 x 1 x 5}
X_0 = {latent: Z_0, camera: M_0}
```

`seed` が変わると `X_0` が変わるため、最終Gaussian `G` も変わります。

### Step 4. flow samplerの時刻列を作る

`steps=K` と `shift=lambda` から、Euler updateで使う時刻列を作ります。

処理内容:

1. `K+1` 個の一様な値 `u_i` を1から0まで作る。
2. shift付きscheduleへ変換する。
3. 隣り合う `(t_i, t_{i+1})` をstepごとの更新区間として使う。

式:

```text
u_i = 1 - i/K
t_i = lambda * u_i / (1 + (lambda - 1) * u_i)
```

この時刻列は、各stepでどれだけ状態を動かすかを決めます。

### Step 5. sampler stepを `K` 回繰り返す

ここがTripoSplat推論の最も重い部分です。各step `i` では、現在状態 `X_i` を画像条件 `C` に合う方向へ更新します。

1 step内の処理内容:

1. 現在状態 `X_i` を取り出す。
2. `X_i` をcloneし、flow modelへ渡す入力 `x_t` を作る。
3. 条件ありforwardを実行する。
4. `guidance_scale > 1` の場合、条件なしforwardも実行する。
5. CFG式で最終velocity `V_hat_i` を作る。
6. `dt_i = t_i - t_{i+1}` を計算する。
7. Euler式で `X_{i+1}` を作る。

式:

```text
V_cond_i   = f_theta(X_i, 1000 t_i, C)
V_uncond_i = f_theta(X_i, 1000 t_i, C_empty)
V_hat_i    = s_cfg * V_cond_i - (s_cfg - 1) * V_uncond_i
dt_i       = t_i - t_{i+1}
X_{i+1}    = X_i - V_hat_i * dt_i
```

`guidance_scale <= 1` の場合は `V_uncond_i` を使わず、`V_hat_i = V_cond_i` になります。今回の `steps=20, guidance_scale=3.0` では、sampler stepは20回ですが、flow model forwardは条件あり/なしで実質40回です。CPU時間の大半はこのstepに使われます。

### Step 6. 最終latent `Z_K` を取り出す

`K` 回の更新後、最終状態 `X_K` が得られます。decoderへ渡すのは主にlatent成分 `Z_K` です。

処理内容:

1. `X_K["latent"]` を取り出す。
2. 必要ならdtypeをdecoderへ合わせる。
3. camera tokenはflow状態には含まれるが、Gaussian decoderの主入力はlatentです。

出力:

```text
Z_K = X_K.latent
Z_K in R^{1 x 8192 x 16}
```

この `Z_K` が、画像条件と学習済み3D priorを統合した圧縮表現です。

### Step 7. octree pointをsampleする

decoderはまず、`Z_K` から3D点候補を作ります。

処理内容:

1. `num_gaussians=N_g` からdecoder point数 `N_p` を決める。
2. `OctreeProbabilityFixedlenDecoder` で `N_p` 個のpointをsampleする。
3. 各pointはGaussianを置く中心候補になります。

式:

```text
N_p = N_g / 32
P = {p_j}_{j=1}^{N_p}
```

例:

```text
131072 gaussians -> 4096 points
262144 gaussians -> 8192 points
```

### Step 8. pointごとにGaussian属性を生成する

`ElasticGaussianFixedlenDecoder` は、各pointの周りに32個のGaussianを生成します。

処理内容:

1. point `P` を1024次元特徴へprojectする。
2. point positional embeddingを加える。
3. latent `Z_K` へcross attentionし、各pointに必要な3D情報を取り込む。
4. Transformer blockを通す。
5. 各pointから32個分のGaussian属性を出す。

出力:

```text
G = {g_i}_{i=1}^{N_g}
g_i = (x_i, f_dc_i, alpha_i, s_i, q_i)
```

ここで初めて、描画可能な3D Gaussian集合になります。

### Step 9. Gaussian属性をactivationする

decoderが出すraw値は、そのままPLYの属性ではありません。位置、opacity、scale、rotationへ変換します。

処理内容:

1. point座標とoffsetからGaussian中心 `x_i` を作る。
2. 色raw値をSH DC係数 `f_dc_i` として扱う。
3. opacity raw値へbiasを足し、sigmoidで `alpha_i` を作る。
4. scale raw値へbiasを足し、softplusで正のscale `s_i` を作る。
5. rotation raw値へidentity quaternion biasを足し、quaternion `q_i` を作る。

このstepは数値変換であり、同等実装では同じactivation式を保つ必要があります。

### Step 10. PLY/SPLATへ保存する

最後にGaussian集合 `G` をraw 3DGSファイルとして保存します。

処理内容:

1. PLY用に座標系transformを適用する。
2. PLYでは `opacity` をlogit空間、`scale` をlog空間で書く。
3. SPLATでは1 Gaussianを32 bytes recordへpackする。
4. SPLAT recordは `-alpha * prod(scale)` でsortして保存する。

出力:

```text
output.ply
output.splat
manifest.json
```

低リソース実装で最も安全に置き換えられるのはこのstepです。streaming exportは、Gaussian属性 `G` を変えずに、最終的な保存バッファだけをchunk化します。

## 全体を合成写像として見る

入力画像を `I`、前処理を `P`、画像encoderを `E`、flow samplerを `S`、Gaussian decoderを `D`、exportを `W` と置くと、本家TripoSplat推論は概念的に次の合成です。

```text
I -> P(I) -> E(P(I)) -> S(E(P(I)); seed, steps, guidance, shift) -> D(latent; N_g) -> Gaussian -> W(Gaussian)
```

数式風に書くと、raw出力 `Y` は以下です。

```text
I_p = P(I)
C   = E(I_p) = {F_1, F_2}
X_K = S_theta(C, epsilon; K, s_cfg, lambda)
G   = D_phi(X_K.latent; N_g)
Y   = W(G)
```

記号:

- `I`: 入力画像。
- `I_p`: TripoSplat encoderが実際に見る正方形RGB画像。
- `C`: 条件特徴。DINO特徴 `F_1` とVAE特徴 `F_2` のdict。
- `epsilon`: seedから生成される初期noise。
- `K`: sampler steps。
- `s_cfg`: guidance scale。
- `lambda`: flow schedule shift。
- `N_g`: 出力Gaussian数。
- `G`: 3D Gaussian属性集合。
- `Y`: PLY/SPLAT raw file。

低リソース同等化では、`Y_candidate ~= Y_reference` を満たすように、実装だけを置き換えます。

## 1. 前処理 `P`

本家 `preprocess_image()` は、入力画像をcanvas size `S=1024` の正方形RGBへ変換します。CPU runnerでは `--canvas-size 512/1024` を使うため、以下では一般の `S` とします。

### resize

元画像サイズを `(W,H)` とし、短辺が `S` になるscaleを

```text
a = S / min(W, H)
```

と置くと、resize後サイズは

```text
W' = round(a W),  H' = round(a H)
```

です。

### alpha / background removal

RGBA入力に実alphaがある場合はそのalphaを使います。alphaが無い場合はBiRefNetでalpha matteを推定します。

```text
A = alpha(I')                 if real alpha exists
A = RMBG(I'_RGB)              otherwise
```

`erode_radius = r` のとき、alphaへ最小フィルタをかけます。

```text
A_e(x,y) = min_{|u-x|<=r, |v-y|<=r} A(u,v)
```

これは境界の背景混入を抑えるための処理です。

### crop and compose

alpha非ゼロ画素のbboxを

```text
B = [x_min, y_min, x_max, y_max]
```

中心を

```text
c = ((x_min + x_max)/2, (y_min + y_max)/2)
```

crop半径を

```text
h = 0.6 * max(x_max - x_min, y_max - y_min)
```

として、正方形cropを取り、`S x S` へresizeします。最後に黒背景へalpha compositeします。

```text
I_p(x,y) = A_e(x,y) * I_crop(x,y) + (1 - A_e(x,y)) * 0
```

注意:

- `P` の違いはraw Gaussian出力を変える。
- staged CPU runでは前処理済みRGBAを使い、BiRefNet再実行を避けています。これは実装上のstagingであり、同じ `I_p` を使う限り推論後段の比較が可能です。

## 2. 画像encoder `E`

TripoSplatの条件特徴は2系統です。

```text
C = {F_1, F_2}
```

- `F_1`: DINOv3 ViT特徴。
- `F_2`: Flux2 VAE encoder特徴。

### 2.1 DINOv3 ViT特徴 `F_1`

入力RGBをImageNet mean/stdで正規化します。

```text
I_d = (I_p - mu) / sigma
```

patch size `p=16` のConv patch embeddingでtoken化します。

```text
X_patch = Conv_{p x p, stride=p}(I_d)
```

canvas sizeを `S` とすると、patch token数は

```text
N_patch = (S / 16)^2
```

です。DINOv3では `cls token` 1個と `register token` 4個を加えるため、token長は

```text
N = 5 + (S / 16)^2
```

です。

本家の主設定では hidden size `d_1 = 1280` なので、

```text
F_1 in R^{1 x N x 1280}
```

です。

各Transformer blockは概念的に以下です。

```text
Q = LN(X) W_Q
K = LN(X) W_K
V = LN(X) W_V
Attn(X) = softmax(Q K^T / sqrt(d_h) + pos) V W_O
X' = X + gamma_1 * Attn(X)
X_next = X' + gamma_2 * MLP(LN(X'))
```

DINOv3ではpatch token側へ2D rotary embeddingが入ります。公式コード上は、patch側の `q,k` に

```text
q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

を適用します。

最終的に `layer_norm` したDINO特徴が `F_1` になります。

### 2.2 Flux2 VAE encoder特徴 `F_2`

VAE encoderにはRGBを `[-1,1]` へ変換して入れます。

```text
I_v = 2 I_p - 1
```

encoderとquant convからmomentを得ます。

```text
[M, L] = QuantConv(Encoder(I_v))
```

ここで `M` はmean、`L` はlog varianceです。本家 `encode_image()` では `deterministic=False` なので、seed付きgeneratorからnoiseを引き、

```text
Z_v = M + exp(0.5 L) * eps_v
```

を作ります。

その後、空間2x2をchannel側へ畳み込むpixel-unshuffle型の変換をします。

```text
Z_v: (B, C, H, W)
  -> (B, 4C, H/2, W/2)
```

BatchNorm running statsで正規化し、flattenしてtoken列にします。

```text
F_2_raw in R^{1 x N_patch x 128}
```

DINO側は `cls + 4 register` の5 tokenを持つため、VAE側にも先頭へ5個のzero tokenをpadします。

```text
F_2 = concat(0_{1 x 5 x 128}, F_2_raw)
F_2 in R^{1 x N x 128}
```

### 2.3 条件特徴の意味

flow model内では、2つの特徴は同じtoken長へ揃えられ、それぞれ1024次元へprojectされて加算されます。

```text
H_c = F_1 W_c1 + F_2 W_c2
H_c in R^{1 x N x 1024}
```

この `H_c` が、単眼画像から3Dを推定するための条件文脈になります。

## 3. Flow matching sampler `S`

生成本体は `LatentSeqMMFlowModel` です。これは、条件特徴 `C` を見ながら、latent token列とcamera tokenをnoiseから生成結果へEuler積分で移動させるvelocity modelです。

### 3.1 state

初期状態はseedから作るnoiseです。

```text
Z_0 ~ N(0, I),  Z_0 in R^{1 x 8192 x 16}
M_0 ~ N(0, I),  M_0 in R^{1 x 1 x 5}
X_0 = {latent: Z_0, camera: M_0}
```

`Z` は8192個のlatent token、各16次元です。`M` はcamera tokenで、5次元です。

### 3.2 time schedule

stepsを `K`、shiftを `lambda` とします。公式コードでは

```python
t_seq = lambda * linspace(1, 0, K + 1) / (1 + (lambda - 1) * linspace(1, 0, K + 1))
```

です。数式では、

```text
u_i = 1 - i/K,  i = 0,...,K

t_i = lambda * u_i / (1 + (lambda - 1) * u_i)
```

となります。

`lambda=1` なら一様scheduleです。`lambda>1` では高noise側へstep配分が寄ります。

### 3.3 model velocity

flow modelは現在状態 `X_i`、時刻 `t_i`、条件 `C` からvelocityを返します。

```text
V_i = f_theta(X_i, 1000 t_i, C)
V_i = {v_Z, v_M}
```

内部構造を大まかに書くと以下です。

```text
H_x = Z_i W_x + PE_sobol
H_c = F_1 W_c1 + F_2 W_c2
E_t = TimestepEmbed(1000 t_i)
A_t = AdaLN(E_t)
```

noise latent側は `num_refiner_blocks=2` でrefineされます。

```text
H_x <- NoiseRefiner(H_x; A_t, rotary)
```

condition側も2 blockでrefineされます。

```text
H_c <- ContextRefiner(H_c; rotary)
```

camera tokenがある場合はMLPで1024次元へprojectされます。

```text
H_m = MLP(M_i)
```

その後、latent token、condition token、camera tokenを結合します。

```text
H = concat(H_x, H_c, H_m)
```

これを `num_blocks=24` のfull attention transformerで処理します。

```text
H <- Block_24(...Block_2(Block_1(H; A_t, rotary)))
```

最後にlatent部分とcamera部分を取り出し、velocityへprojectします。

```text
v_Z = LN(H_latent) W_out
v_M = LN(H_camera) W_cam
```

`use_shift_table=True` のため、出力直前にtime embedding由来のshift/scaleも入ります。

```text
H_latent = H_latent * (1 + scale_t) + shift_t
H_camera = H_camera * (1 + scale_t) + shift_t
```

### 3.4 CFG

unconditional conditionはzero特徴です。

```text
C_empty = {0_like(F_1), 0_like(F_2)}
```

conditional velocityとunconditional velocityを

```text
V_cond = f_theta(X_i, 1000 t_i, C)
V_uncond = f_theta(X_i, 1000 t_i, C_empty)
```

とすると、guidance scale `s_cfg` のCFG velocityは

```text
V_hat = s_cfg * V_cond - (s_cfg - 1) * V_uncond
```

です。`s_cfg <= 1` の場合はunconditional passを実行せず、conditionalのみになります。今回の標準 `guidance_scale=3.0` では、各stepでcond/uncondの2回forwardが必要です。

### 3.5 Euler update

`t_i > t_{i+1}` として、

```text
dt_i = t_i - t_{i+1}
```

公式updateは

```text
X_{i+1} = X_i - V_hat_i * dt_i
```

です。これを `K` 回繰り返して、最終状態 `X_K` を得ます。

```text
X_K = S_theta(C, epsilon; K, s_cfg, lambda)
```

CPU実行時間の支配要因はこのflow model forwardです。特に `K=20, s_cfg=3.0` では、実質40回の大きなTransformer forwardになります。

## 4. Gaussian decoder `D`

flow samplerの出力 `X_K.latent` を、3D Gaussian集合へ変換します。

```text
Z = X_K.latent in R^{1 x 8192 x 16}
```

decoderは2段です。

```text
OctreeGaussianDecoder
  -> OctreeProbabilityFixedlenDecoder
  -> ElasticGaussianFixedlenDecoder
```

### 4.1 Gaussian数とdecoder token数

本家GS decoderでは、1つのoctree pointから `G_pp=32` 個のGaussianを作ります。

出力Gaussian数を `N_g` とすると、decoder point数は

```text
N_p = max(1, N_g / 32)
```

です。

例:

```text
32768 gaussians  -> 1024 points
131072 gaussians -> 4096 points
262144 gaussians -> 8192 points
```

### 4.2 Octree point sampling

Octree decoderはlatent `Z` からpoint集合をsampleします。

```text
P = O_omega(Z; N_p, level=8, temperature=1.0, algo=systematic)
P = {p_j}_{j=1}^{N_p},  p_j in R^3
```

これは単純な画像平面proxyではなく、学習済みdecoderがlatentから3D point候補を出す段です。

### 4.3 Elastic Gaussian decoder

Elastic decoderは、point `P` とlatent `Z` を条件に、各pointから32個のGaussian属性を生成します。

```text
H = T_rho(P, Z)
H in R^{1 x N_p x C_out}
```

layoutは以下です。各pointごとに32個分を持ちます。

```text
_xyz         : (32, 3)
_features_dc : (32, 1, 3)
_scaling     : (32, 3)
_rotation    : (32, 4)
_opacity     : (32, 1)
_offset_scale: (32, 1)  # learned offset scale使用時
```

`ElasticGaussianFixedlenDecoder` は、point座標をinput projectionし、point positional embeddingを加え、latent `Z` にcross attentionするTransformerです。

```text
H_0 = W_in P + PE(P)
H_l = CrossBlock_l(H_{l-1}, Z),  l=1,...,16
H = W_out(LN(H_16))
```

ここは `attn_mode="full"` です。低リソース化で安易にtoken分割するとattention結果が変わる可能性が高いため、同等化では注意が必要です。

### 4.4 Gaussian中心

各point `p_j` の周りに32個のoffsetを生成します。raw offsetを `r_{j,k}`、Hammersley perturbationを `q_k`、learned offset scaleを `a_{j,k}` とします。

公式実装の流れは概念的に以下です。

```text
a_{j,k} = softplus(h_offset_scale_{j,k} + b_offset)

Delta_{j,k}
  = tanh(lr_xyz * r_{j,k} + q_k)
    * 0.5 * perturbe_size
    * a_{j,k}
```

最終的な正規化座標は

```text
x_tilde_{j,k} = p_j + Delta_{j,k}
```

`Gaussian` objectではAABB `[-0.5,-0.5,-0.5] + [1,1,1]` に写します。今回のAABBでは実質

```text
x_{j,k} = x_tilde_{j,k} - 0.5
```

です。

### 4.5 色

色はspherical harmonicsのDC項だけを使います。decoderから出るraw値を `h_c` とし、learning rate係数 `lr_c=1.0` を掛けます。

```text
f_dc_{j,k} = lr_c * h_c
```

viewer/SPLATでRGBへ戻すときは、SH DC定数

```text
C0 = 0.28209479177387814 = 1 / (2 sqrt(pi))
```

を使います。

```text
RGB = clip(C0 * f_dc + 0.5, 0, 1)
```

### 4.6 opacity

decoder raw opacityを `h_o`、opacity biasを `o_b=0.1` とします。実装ではbiasをlogit空間へ変換してから足します。

```text
logit(o_b) = log(o_b / (1 - o_b))

alpha = sigmoid(h_o + logit(0.1))
```

PLYへ保存するときは、activated opacityではなくinverse sigmoid後の値を書きます。

```text
opacity_ply = logit(alpha)
```

### 4.7 scale

decoder raw scalingを `h_s`、scaling biasを `s_b=0.004`、minimum kernel sizeを `m=0.0009` とします。scaling activationはsoftplusです。

```text
softplus^{-1}(s_b) = s_b + log(-expm1(-s_b))

s_raw = softplus(h_s + softplus^{-1}(0.004))
s = sqrt(s_raw^2 + m^2)
```

PLYへ保存するときはlog scaleを書きます。

```text
scale_ply = log(s)
```

### 4.8 rotation

decoder raw rotationを `h_r in R^4` とし、rotation learning rateは `0.1`、identity quaternion biasは `[1,0,0,0]` です。

```text
q_raw = 0.1 * h_r + [1, 0, 0, 0]
```

SPLAT packingやtransform適用時には正規化されたquaternionとして扱います。

```text
q = q_raw / ||q_raw||
```

PLY/SPLAT exportではデフォルトtransform

```text
T = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
```

を適用します。座標と回転は

```text
x_export = T x
R_export = T R(q)
q_export = quat(R_export)
```

となります。

## 5. Gaussian表現 `G`

最終的なGaussian集合は

```text
G = {g_i}_{i=1}^{N_g}
```

で、各Gaussianは以下を持ちます。

```text
g_i = (x_i, f_dc_i, alpha_i, s_i, q_i)
```

- `x_i in R^3`: 位置。
- `f_dc_i in R^3`: SH DC色。
- `alpha_i in (0,1)`: opacity。
- `s_i in R_+^3`: anisotropic scale。
- `q_i in R^4`: rotation quaternion。

PLYの属性は以下です。

```text
x, y, z
nx, ny, nz
f_dc_0, f_dc_1, f_dc_2
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
```

注意:

- PLYの `opacity` はlogit値。
- PLYの `scale_*` はlog scale。
- PLYの `rot_*` はtransform適用後のquaternion。
- SPLATは `xyz(12 bytes) + scale(12 bytes) + rgba(4 bytes) + rotation(4 bytes)` の32 bytes/record。
- SPLAT record orderは `-alpha * prod(scale)` の降順です。

## 6. 評価rendererの数式

以下はTripoSplat推論ではなく、出力PLYを評価・表示するためのGaussian splatting式です。軽量ref実装を参考にした箇所は主にここです。

### 6.1 3D covariance

scale `s_i=(s_x,s_y,s_z)` と回転 `R_i` から3D covarianceを作ります。

```text
Sigma_3D,i = R_i diag(s_i^2) R_i^T
```

### 6.2 projection covariance

カメラ座標の点を `y_i`、projection関数を `pi` とします。

```text
u_i = pi(y_i)
```

projectionのJacobianを

```text
J_i = d pi / d y |_{y_i}
```

とすると、画面上covarianceは一次近似で

```text
Sigma_2D,i = J_i Sigma_3D,i J_i^T
```

です。

### 6.3 screen weight

画面pixel `p` に対するGaussian weightは

```text
w_i(p) = exp(-0.5 * (p - u_i)^T Sigma_2D,i^{-1} (p - u_i))
```

です。

### 6.4 alpha compositing

奥行き順に並べ、透過率 `T` と色 `C` を更新します。

```text
C(p) <- C(p) + T(p) * alpha_i * w_i(p) * color_i
T(p) <- T(p) * (1 - alpha_i * w_i(p))
```

このrendererのFOV、distance、radius clip、alpha scaleを変えても、raw TripoSplat `G` は変わりません。したがって、renderer tuningでPSNRが上がっても、TripoSplat推論の同等性証明にはなりません。

## 7. 低リソース同等化で保存すべき不変量

低リソース実装候補を `A'`、本家実装を `A` とします。同じ設定 `Omega` を

```text
Omega = (I_p, checkpoint, seed, K, s_cfg, lambda, N_g, dtype/backend condition)
```

と置くと、同等化の条件は

```text
A'(Omega).PLY_fields ~= A(Omega).PLY_fields
A'(Omega).SPLAT_records ~= A(Omega).SPLAT_records
```

です。

比較対象:

```text
x, y, z
f_dc_0, f_dc_1, f_dc_2
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
record order
```

現在安全に置換できている境界はexportです。

### export streamingの同等性

本家exportは概念的に、全Gaussian recordを一度配列化してから書きます。

```text
B = concat(record_1, record_2, ..., record_N)
write(B)
```

lowmem exportはchunkで書きます。

```text
write(concat(record_1, ..., record_m))
write(concat(record_{m+1}, ..., record_{2m}))
...
```

record内容と順序が同じなら、最終byte列は同じです。

```text
concat(chunks(records)) = concat(records)
```

このため、export boundaryは同等性を保ったまま低リソース化しやすい境界です。

## 8. 低リソース化が難しい箇所

### sampler

samplerは式としては単純です。

```text
X_{i+1} = X_i - V_hat_i * dt_i
```

しかし `V_hat_i` を出すflow modelが巨大なfull attention transformerです。ここを変えるとraw latentが変わり、decoder後のGaussianも変わります。

許容できる候補:

- CFGのcond/uncondをbatch=2にまとめる。ただし実モデルでraw比較が必要。
- model load/unloadやactivation lifetimeを変える。
- dtype/backendを固定した上で、同じ演算順に近い形でmemoryを削る。

危険な候補:

- attention tokenを単純chunk分割する。
- steps/guidance/shiftを変えて同等と主張する。
- quantization後に許容差を決めず同等扱いする。

### decoder

Elastic Gaussian decoderもfull attention/cross attentionを持ちます。point tokenを分割すると、attentionの相互作用が消えて結果が変わる可能性があります。

安全寄りの候補:

- out projection以降のattribute packing / export streaming。
- full attentionを数学的に同じblockwise attentionとして実装する。

危険な候補:

- point tokenを独立chunkとしてdecoder forwardする。
- `num_gaussians` を変えて同等と扱う。

## 9. 今回検証のパラメータとの対応

主な探索パラメータは、上の式のどこに入るかで整理できます。

| パラメータ | 式中の位置 | 変えるとraw出力は変わるか | 役割 |
| --- | --- | --- | --- |
| `canvas_size` | `P(I)` とencoder token数 | 変わる | 入力情報量とtoken数 |
| `seed` | `eps_v`, `Z_0`, `M_0` | 変わる | VAE samplingとflow初期noise |
| `steps` | `K` | 変わる | Euler積分回数 |
| `guidance_scale` | `s_cfg` | 変わる | CFG強度 |
| `shift` | `lambda` | 変わる | time schedule |
| `num_gaussians` | `N_g`, `N_p=N_g/32` | 変わる | decoder出力密度 |
| `model_dtype` | model演算dtype | 変わる可能性あり | メモリと数値誤差 |
| `lowmem_export` | `W(G)` | 正しければ変わらない | serialization memory削減 |
| `export_chunk_size` | `W(G)` chunk幅 | 正しければ変わらない | 一時メモリとI/O loop |
| renderer FOV/distance/view | 評価renderer | 変わらない | 見え方・比較camera |
| LS色refit/opacity bake | `G` export後の改変 | 変わる | 診断・postprocess、主結果ではない |

## 10. まとめ

TripoSplatは、入力画像をDINO/VAE特徴へ変換し、その条件特徴でflow matching transformerをEuler積分し、得られたlatentをoctree + elastic Gaussian decoderで3D Gaussian集合へ復号する単眼3D生成モデルです。

重要な数式上の核は以下です。

```text
C = {F_1, F_2}
V_hat = s_cfg f_theta(X_t,t,C) - (s_cfg - 1) f_theta(X_t,t,0)
X_{i+1} = X_i - V_hat_i * (t_i - t_{i+1})
N_p = N_g / 32
G = D_phi(X_K.latent; N_g)
```

Gaussian属性はdecoder raw値にbias/activationを通して得ます。

```text
alpha = sigmoid(h_o + logit(0.1))
s = sqrt(softplus(h_s + softplus^{-1}(0.004))^2 + 0.0009^2)
q = normalize(0.1 h_r + [1,0,0,0])
RGB = C0 f_dc + 0.5
```

低リソース実装で守るべきことは、この写像の意味を変えないことです。`ref` はこの写像を置き換えるものではなく、chunking、streaming、小パッチ処理、bounded memory評価器を設計するための参考です。
