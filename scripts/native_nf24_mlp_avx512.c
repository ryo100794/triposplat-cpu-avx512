#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
  ROW_TILE = 8,
  INPUT_FEATURES = 1024,
  HIDDEN_FEATURES = 4096,
  OUTPUT_FEATURES = 1024,
  SIMD_WIDTH = 16
};

static inline __m512i load16_i16(const uint8_t* codes) {
  const __m256i packed = _mm256_loadu_si256((const __m256i*)codes);
  return _mm512_cvtepi16_epi32(packed);
}

static inline __m512i load16_i8(const uint8_t* codes) {
  const __m128i packed = _mm_loadu_si128((const __m128i*)codes);
  return _mm512_cvtepi8_epi32(packed);
}

static inline __m512 decode16(
    const uint8_t* code01,
    const uint8_t* code2,
    __m512 scale) {
  const __m512i q01 = _mm512_slli_epi32(load16_i16(code01), 8);
  const __m512i q = _mm512_add_epi32(q01, load16_i8(code2));
  return _mm512_mul_ps(_mm512_cvtepi32_ps(q), scale);
}

static inline void gemm_8x16(
    const float* x,
    const uint8_t* code01_t,
    const uint8_t* code2_t,
    const float* scales,
    const float* bias,
    float* out,
    int K,
    int N,
    int stride_x,
    int stride_out,
    int output_n) {
  const __m512 scale = _mm512_loadu_ps(scales + output_n);
  __m512 c0 = _mm512_loadu_ps(bias + output_n);
  __m512 c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (int k = 0; k < K; ++k) {
    const int64_t offset = (int64_t)k * N + output_n;
    const __m512 w = decode16(code01_t + offset * 2, code2_t + offset, scale);
    c0 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)0 * stride_x + k]), w, c0);
    c1 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)1 * stride_x + k]), w, c1);
    c2 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)2 * stride_x + k]), w, c2);
    c3 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)3 * stride_x + k]), w, c3);
    c4 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)4 * stride_x + k]), w, c4);
    c5 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)5 * stride_x + k]), w, c5);
    c6 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)6 * stride_x + k]), w, c6);
    c7 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)7 * stride_x + k]), w, c7);
  }
  _mm512_storeu_ps(out + (int64_t)0 * stride_out + output_n, c0);
  _mm512_storeu_ps(out + (int64_t)1 * stride_out + output_n, c1);
  _mm512_storeu_ps(out + (int64_t)2 * stride_out + output_n, c2);
  _mm512_storeu_ps(out + (int64_t)3 * stride_out + output_n, c3);
  _mm512_storeu_ps(out + (int64_t)4 * stride_out + output_n, c4);
  _mm512_storeu_ps(out + (int64_t)5 * stride_out + output_n, c5);
  _mm512_storeu_ps(out + (int64_t)6 * stride_out + output_n, c6);
  _mm512_storeu_ps(out + (int64_t)7 * stride_out + output_n, c7);
}

static inline void gemm_1x16(
    const float* x,
    const uint8_t* code01_t,
    const uint8_t* code2_t,
    const float* scales,
    const float* bias,
    float* out,
    int K,
    int N,
    int output_n) {
  const __m512 scale = _mm512_loadu_ps(scales + output_n);
  __m512 c = _mm512_loadu_ps(bias + output_n);
  for (int k = 0; k < K; ++k) {
    const int64_t offset = (int64_t)k * N + output_n;
    const __m512 w = decode16(code01_t + offset * 2, code2_t + offset, scale);
    c = _mm512_fmadd_ps(_mm512_set1_ps(x[k]), w, c);
  }
  _mm512_storeu_ps(out + output_n, c);
}

static inline __m512 exp512_ps(__m512 x) {
  const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
  const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
  const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 one = _mm512_set1_ps(1.0f);
  x = _mm512_min_ps(x, exp_hi);
  x = _mm512_max_ps(x, exp_lo);
  __m512 fx = _mm512_floor_ps(_mm512_fmadd_ps(x, log2ef, half));
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
  const __m512 z = _mm512_mul_ps(x, x);
  __m512 y = _mm512_set1_ps(1.9875691500e-4f);
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
  y = _mm512_add_ps(_mm512_fmadd_ps(y, z, x), one);
  __m512i emm = _mm512_add_epi32(_mm512_cvttps_epi32(fx), _mm512_set1_epi32(0x7f));
  emm = _mm512_slli_epi32(emm, 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(emm));
}

static inline __m512 tanh512_ps(__m512 x) {
  const __m512 zero = _mm512_setzero_ps();
  const __m512 one = _mm512_set1_ps(1.0f);
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  const __m512 ax = _mm512_andnot_ps(sign_mask, x);
  const __m512 e = exp512_ps(_mm512_mul_ps(_mm512_set1_ps(2.0f), ax));
  __m512 y = _mm512_sub_ps(one, _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, one)));
  y = _mm512_or_ps(y, _mm512_and_ps(x, sign_mask));
  return _mm512_mask_mov_ps(y, _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ), zero);
}

static inline void gelu_tanh_inplace(float* values, int count) {
  for (int offset = 0; offset < count; offset += SIMD_WIDTH) {
    const __m512 x = _mm512_loadu_ps(values + offset);
    const __m512 x2 = _mm512_mul_ps(x, x);
    const __m512 x3_term = _mm512_mul_ps(
        _mm512_set1_ps(0.044715f), _mm512_mul_ps(x2, x));
    const __m512 inner = _mm512_mul_ps(
        _mm512_set1_ps(0.7978845608028654f), _mm512_add_ps(x, x3_term));
    const __m512 y = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_set1_ps(0.5f), x),
        _mm512_add_ps(_mm512_set1_ps(1.0f), tanh512_ps(inner)));
    _mm512_storeu_ps(values + offset, y);
  }
}

int triposplat_mlp_nf24_gelu_f32_avx512(
    const float* x,
    const uint8_t* w1_code01_t,
    const uint8_t* w1_code2_t,
    const float* w1_scales,
    const float* w1_bias,
    const uint8_t* w2_code01_t,
    const uint8_t* w2_code2_t,
    const float* w2_scales,
    const float* w2_bias,
    float* out,
    int M,
    int K,
    int H,
    int N,
    int stride_x,
    int stride_out,
    int threads) {
  if (x == 0 || w1_code01_t == 0 || w1_code2_t == 0 || w1_scales == 0 ||
      w1_bias == 0 || w2_code01_t == 0 || w2_code2_t == 0 || w2_scales == 0 ||
      w2_bias == 0 || out == 0) return -1;
  if (M <= 0 || K != INPUT_FEATURES || H != HIDDEN_FEATURES || N != OUTPUT_FEATURES ||
      stride_x < K || stride_out < N) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  const int m_full = (M / ROW_TILE) * ROW_TILE;

#pragma omp parallel
  {
    float hidden[ROW_TILE * HIDDEN_FEATURES] __attribute__((aligned(64)));
#pragma omp for schedule(static)
    for (int i = 0; i < m_full; i += ROW_TILE) {
      const float* x_tile = x + (int64_t)i * stride_x;
      float* out_tile = out + (int64_t)i * stride_out;
      for (int hn = 0; hn < H; hn += SIMD_WIDTH) {
        gemm_8x16(x_tile, w1_code01_t, w1_code2_t, w1_scales, w1_bias,
                  hidden, K, H, stride_x, H, hn);
      }
      gelu_tanh_inplace(hidden, ROW_TILE * H);
      for (int n = 0; n < N; n += SIMD_WIDTH) {
        gemm_8x16(hidden, w2_code01_t, w2_code2_t, w2_scales, w2_bias,
                  out_tile, H, N, H, stride_out, n);
      }
    }

#pragma omp for schedule(static)
    for (int i = m_full; i < M; ++i) {
      const float* x_row = x + (int64_t)i * stride_x;
      float* out_row = out + (int64_t)i * stride_out;
      for (int hn = 0; hn < H; hn += SIMD_WIDTH) {
        gemm_1x16(x_row, w1_code01_t, w1_code2_t, w1_scales, w1_bias,
                  hidden, K, H, hn);
      }
      gelu_tanh_inplace(hidden, H);
      for (int n = 0; n < N; n += SIMD_WIDTH) {
        gemm_1x16(hidden, w2_code01_t, w2_code2_t, w2_scales, w2_bias,
                  out_row, H, N, n);
      }
    }
  }
  return 0;
}

int triposplat_mlp_nf24_row_tile(void) {
  return ROW_TILE;
}

int triposplat_mlp_nf24_hidden_workspace_bytes_per_thread(void) {
  return ROW_TILE * HIDDEN_FEATURES * (int)sizeof(float);
}
