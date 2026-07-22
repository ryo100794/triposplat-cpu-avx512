#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
  HEAD_DIM = 64,
  SIMD_WIDTH = 16,
  TOKEN_TILE = 16
};

static inline __m512 rope16(__m512 x, __m512 frequency) {
  const __m512 x_re = _mm512_moveldup_ps(x);
  const __m512 x_im = _mm512_movehdup_ps(x);
  const __m512 frequency_swap = _mm512_permute_ps(frequency, 0xb1);
  return _mm512_fmaddsub_ps(x_re, frequency, _mm512_mul_ps(x_im, frequency_swap));
}

int triposplat_qkv_rope_rmsnorm_pack_f32_avx512(
    const float* qkv,
    const float* frequencies,
    const float* q_gamma,
    const float* k_gamma,
    float* q_bhld,
    float* packed_k,
    float* packed_v,
    int B,
    int L,
    int H,
    int D,
    int L_padded,
    int frequency_batches,
    int threads) {
  if (qkv == 0 || frequencies == 0 || q_gamma == 0 || k_gamma == 0 ||
      q_bhld == 0 || packed_k == 0 || packed_v == 0) return -1;
  if (B <= 0 || L <= 0 || H <= 0 || D != HEAD_DIM || L_padded < L ||
      (L_padded & 15) != 0 || (frequency_batches != 1 && frequency_batches != B)) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const int64_t bh_count = (int64_t)B * H;
  const int64_t packed_count = bh_count * D * (int64_t)L_padded;
  memset(packed_k, 0, (size_t)packed_count * sizeof(float));
  memset(packed_v, 0, (size_t)packed_count * sizeof(float));
  const int token_tiles = (L + TOKEN_TILE - 1) / TOKEN_TILE;

#pragma omp parallel for schedule(static)
  for (int64_t work = 0; work < bh_count * token_tiles; ++work) {
    const int tile = (int)(work % token_tiles);
    const int64_t bh = work / token_tiles;
    const int b = (int)(bh / H);
    const int h = (int)(bh % H);
    const int first_token = tile * TOKEN_TILE;
    const int rows = first_token + TOKEN_TILE <= L ? TOKEN_TILE : L - first_token;
    float k_tile[TOKEN_TILE][HEAD_DIM] __attribute__((aligned(64)));
    float v_tile[TOKEN_TILE][HEAD_DIM] __attribute__((aligned(64)));

    for (int row = 0; row < rows; ++row) {
      const int token = first_token + row;
      const int64_t qkv_row = (int64_t)b * L + token;
      const float* q_src = qkv + (qkv_row * 3 * H + h) * D;
      const float* k_src = q_src + (int64_t)H * D;
      const float* v_src = k_src + (int64_t)H * D;
      const int frequency_b = frequency_batches == 1 ? 0 : b;
      const float* frequency = frequencies + (((int64_t)frequency_b * L + token) * H + h) * D;
      float q_rotated[HEAD_DIM] __attribute__((aligned(64)));
      float k_rotated[HEAD_DIM] __attribute__((aligned(64)));
      __m512 q_sumsq = _mm512_setzero_ps();
      __m512 k_sumsq = _mm512_setzero_ps();

      for (int d = 0; d < D; d += SIMD_WIDTH) {
        const __m512 f = _mm512_loadu_ps(frequency + d);
        const __m512 qr = rope16(_mm512_loadu_ps(q_src + d), f);
        const __m512 kr = rope16(_mm512_loadu_ps(k_src + d), f);
        _mm512_store_ps(q_rotated + d, qr);
        _mm512_store_ps(k_rotated + d, kr);
        q_sumsq = _mm512_fmadd_ps(qr, qr, q_sumsq);
        k_sumsq = _mm512_fmadd_ps(kr, kr, k_sumsq);
      }

      const float q_norm = sqrtf(_mm512_reduce_add_ps(q_sumsq));
      const float k_norm = sqrtf(_mm512_reduce_add_ps(k_sumsq));
      const __m512 q_factor = _mm512_set1_ps(8.0f / fmaxf(q_norm, 1.0e-12f));
      const __m512 k_factor = _mm512_set1_ps(8.0f / fmaxf(k_norm, 1.0e-12f));
      float* q_dst = q_bhld + (bh * L + token) * D;
      for (int d = 0; d < D; d += SIMD_WIDTH) {
        const __m512 q_value = _mm512_mul_ps(
            _mm512_mul_ps(_mm512_load_ps(q_rotated + d), q_factor),
            _mm512_loadu_ps(q_gamma + (int64_t)h * D + d));
        const __m512 k_value = _mm512_mul_ps(
            _mm512_mul_ps(_mm512_load_ps(k_rotated + d), k_factor),
            _mm512_loadu_ps(k_gamma + (int64_t)h * D + d));
        _mm512_storeu_ps(q_dst + d, q_value);
        _mm512_store_ps(k_tile[row] + d, k_value);
        _mm512_store_ps(v_tile[row] + d, _mm512_loadu_ps(v_src + d));
      }
    }

    float* packed_k_bh = packed_k + bh * D * (int64_t)L_padded;
    float* packed_v_bh = packed_v + bh * D * (int64_t)L_padded;
    for (int d = 0; d < D; ++d) {
      float* k_dst = packed_k_bh + (int64_t)d * L_padded + first_token;
      float* v_dst = packed_v_bh + (int64_t)d * L_padded + first_token;
      for (int row = 0; row < rows; ++row) {
        k_dst[row] = k_tile[row][d];
        v_dst[row] = v_tile[row][d];
      }
    }
  }
  return 0;
}

int triposplat_qkv_postprocess_token_tile(void) {
  return TOKEN_TILE;
}
