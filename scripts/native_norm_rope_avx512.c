#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline __mmask16 mask16_for(int remaining) {
  return remaining >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << remaining) - 1u);
}

int triposplat_layernorm_f32_avx512(
    const float* input, const float* weight, const float* bias, float* output,
    int64_t rows, int cols, float eps, int has_affine, int threads) {
  if (input == 0 || output == 0) return -1;
  if (rows <= 0 || cols <= 0 || eps < 0.0f) return -2;
  if (has_affine && (weight == 0 || bias == 0)) return -3;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    const float* src = input + row * (int64_t)cols;
    float* dst = output + row * (int64_t)cols;
    __m512 sum_v = _mm512_setzero_ps();
    for (int col = 0; col < cols; col += 16) {
      sum_v = _mm512_add_ps(sum_v, _mm512_maskz_loadu_ps(mask16_for(cols - col), src + col));
    }
    const float mean = _mm512_reduce_add_ps(sum_v) / (float)cols;
    const __m512 mean_v = _mm512_set1_ps(mean);
    __m512 var_v = _mm512_setzero_ps();
    for (int col = 0; col < cols; col += 16) {
      const __m512 delta = _mm512_sub_ps(_mm512_maskz_loadu_ps(mask16_for(cols - col), src + col), mean_v);
      var_v = _mm512_fmadd_ps(delta, delta, var_v);
    }
    const float inv_std = 1.0f / sqrtf(_mm512_reduce_add_ps(var_v) / (float)cols + eps);
    const __m512 inv_v = _mm512_set1_ps(inv_std);
    for (int col = 0; col < cols; col += 16) {
      const __mmask16 mask = mask16_for(cols - col);
      __m512 y = _mm512_mul_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(mask, src + col), mean_v), inv_v);
      if (has_affine) {
        y = _mm512_fmadd_ps(y, _mm512_maskz_loadu_ps(mask, weight + col), _mm512_maskz_loadu_ps(mask, bias + col));
      }
      _mm512_mask_storeu_ps(dst + col, mask, y);
    }
  }
  return 0;
}

int triposplat_multihead_rmsnorm_f32_avx512(
    const float* input, const float* gamma, float* output,
    int64_t outer, int heads, int dim, float eps, int threads) {
  if (input == 0 || gamma == 0 || output == 0) return -1;
  if (outer <= 0 || heads <= 0 || dim <= 0 || eps < 0.0f) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  const int64_t vectors = outer * heads;
#pragma omp parallel for schedule(static)
  for (int64_t vector = 0; vector < vectors; ++vector) {
    const int head = (int)(vector % heads);
    const float* src = input + vector * dim;
    const float* scale = gamma + (int64_t)head * dim;
    float* dst = output + vector * dim;
    __m512 sumsq_v = _mm512_setzero_ps();
    for (int d = 0; d < dim; d += 16) {
      const __m512 x = _mm512_maskz_loadu_ps(mask16_for(dim - d), src + d);
      sumsq_v = _mm512_fmadd_ps(x, x, sumsq_v);
    }
    const float norm = sqrtf(_mm512_reduce_add_ps(sumsq_v));
    const float factor = sqrtf((float)dim) / fmaxf(norm, eps);
    const __m512 factor_v = _mm512_set1_ps(factor);
    for (int d = 0; d < dim; d += 16) {
      const __mmask16 mask = mask16_for(dim - d);
      const __m512 y = _mm512_mul_ps(
          _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, src + d), factor_v),
          _mm512_maskz_loadu_ps(mask, scale + d));
      _mm512_mask_storeu_ps(dst + d, mask, y);
    }
  }
  return 0;
}

int triposplat_rope_complex_f32_avx512(
    const float* input, const float* freqs_interleaved, float* output,
    int64_t vectors, int dim, int threads) {
  if (input == 0 || freqs_interleaved == 0 || output == 0) return -1;
  if (vectors <= 0 || dim <= 0 || (dim & 1) != 0) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel for schedule(static)
  for (int64_t vector = 0; vector < vectors; ++vector) {
    const float* src = input + vector * dim;
    const float* freq = freqs_interleaved + vector * dim;
    float* dst = output + vector * dim;
    for (int d = 0; d < dim; d += 16) {
      const __mmask16 mask = mask16_for(dim - d);
      const __m512 x = _mm512_maskz_loadu_ps(mask, src + d);
      const __m512 f = _mm512_maskz_loadu_ps(mask, freq + d);
      const __m512 x_re = _mm512_moveldup_ps(x);
      const __m512 x_im = _mm512_movehdup_ps(x);
      const __m512 f_swap = _mm512_permute_ps(f, 0xb1);
      const __m512 y = _mm512_fmaddsub_ps(x_re, f, _mm512_mul_ps(x_im, f_swap));
      _mm512_mask_storeu_ps(dst + d, mask, y);
    }
  }
  return 0;
}
