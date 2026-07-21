#include <immintrin.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline __mmask16 mask16_for(int remaining) {
  return remaining >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << remaining) - 1u);
}

static inline void sincos_ps(__m512 x, __m512* sin_out, __m512* cos_out) {
  const __m512 inv_pio2 = _mm512_set1_ps(0.63661977236758134308f);
  const __m512 pio2_hi = _mm512_set1_ps(1.57079625129699707031f);
  const __m512 pio2_lo = _mm512_set1_ps(7.54978941586159635335e-8f);
  const __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(x, inv_pio2));
  const __m512 qf = _mm512_cvtepi32_ps(q);
  __m512 r = _mm512_fnmadd_ps(qf, pio2_hi, x);
  r = _mm512_fnmadd_ps(qf, pio2_lo, r);
  const __m512 z = _mm512_mul_ps(r, r);

  __m512 sp = _mm512_set1_ps(-1.0f / 6227020800.0f);
  sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(1.0f / 39916800.0f));
  sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(-1.0f / 362880.0f));
  sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(1.0f / 5040.0f));
  sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(-1.0f / 120.0f));
  sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(1.0f / 6.0f));
  const __m512 sin_r = _mm512_sub_ps(r, _mm512_mul_ps(_mm512_mul_ps(r, z), sp));

  __m512 cp = _mm512_set1_ps(-1.0f / 87178291200.0f);
  cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(1.0f / 479001600.0f));
  cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(-1.0f / 3628800.0f));
  cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(1.0f / 40320.0f));
  cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(-1.0f / 720.0f));
  cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(1.0f / 24.0f));
  cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(-0.5f));
  const __m512 cos_r = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(1.0f));

  const __m512 sign = _mm512_set1_ps(-0.0f);
  const __m512 neg_sin_r = _mm512_xor_ps(sin_r, sign);
  const __m512 neg_cos_r = _mm512_xor_ps(cos_r, sign);
  const __m512i quadrant = _mm512_and_epi32(q, _mm512_set1_epi32(3));
  const __mmask16 q1 = _mm512_cmpeq_epi32_mask(quadrant, _mm512_set1_epi32(1));
  const __mmask16 q2 = _mm512_cmpeq_epi32_mask(quadrant, _mm512_set1_epi32(2));
  const __mmask16 q3 = _mm512_cmpeq_epi32_mask(quadrant, _mm512_set1_epi32(3));
  __m512 s = sin_r;
  __m512 c = cos_r;
  s = _mm512_mask_mov_ps(s, q1, cos_r);
  s = _mm512_mask_mov_ps(s, q2, neg_sin_r);
  s = _mm512_mask_mov_ps(s, q3, neg_cos_r);
  c = _mm512_mask_mov_ps(c, q1, neg_sin_r);
  c = _mm512_mask_mov_ps(c, q2, neg_cos_r);
  c = _mm512_mask_mov_ps(c, q3, sin_r);
  *sin_out = s;
  *cos_out = c;
}

static inline void store_interleaved(float* output, __m512 real, __m512 imag, int count) {
  const __m512i lo_idx = _mm512_setr_epi32(
      0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
  const __m512i hi_idx = _mm512_setr_epi32(
      8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
  const __m512 lo = _mm512_permutex2var_ps(real, lo_idx, imag);
  const __m512 hi = _mm512_permutex2var_ps(real, hi_idx, imag);
  const int first = count < 8 ? count : 8;
  _mm512_mask_storeu_ps(output, mask16_for(first * 2), lo);
  if (count > 8) {
    _mm512_mask_storeu_ps(output + 16, mask16_for((count - 8) * 2), hi);
  }
}

int triposplat_mul_inplace_f32_avx512(float* lhs, const float* rhs, int64_t elements, int threads) {
  if (lhs == 0 || rhs == 0) return -1;
  if (elements <= 0) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < elements; i += 16) {
    const __mmask16 mask = mask16_for((int)(elements - i));
    const __m512 a = _mm512_maskz_loadu_ps(mask, lhs + i);
    const __m512 b = _mm512_maskz_loadu_ps(mask, rhs + i);
    _mm512_mask_storeu_ps(lhs + i, mask, _mm512_mul_ps(a, b));
  }
  return 0;
}

int triposplat_repo_phasor_f32_avx512(
    const float* delta,
    const float* freq_tanh_0, const float* freq_residual_0, int freq_dim_0,
    const float* freq_tanh_1, const float* freq_residual_1, int freq_dim_1,
    const float* freq_tanh_2, const float* freq_residual_2, int freq_dim_2,
    float* output_interleaved, int64_t vectors, int threads) {
  if (delta == 0 || output_interleaved == 0) return -1;
  if (freq_tanh_0 == 0 || freq_residual_0 == 0 ||
      freq_tanh_1 == 0 || freq_residual_1 == 0 ||
      freq_tanh_2 == 0 || freq_residual_2 == 0) return -2;
  if (vectors <= 0 || freq_dim_0 <= 0 || freq_dim_1 <= 0 || freq_dim_2 <= 0) return -3;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  const float* freq_tanh[3] = {freq_tanh_0, freq_tanh_1, freq_tanh_2};
  const float* freq_residual[3] = {freq_residual_0, freq_residual_1, freq_residual_2};
  const int dims[3] = {freq_dim_0, freq_dim_1, freq_dim_2};
  const int total_dim = freq_dim_0 + freq_dim_1 + freq_dim_2;
  const __m512 pi = _mm512_set1_ps(3.14159274101257324219f);
#pragma omp parallel for schedule(static)
  for (int64_t vector = 0; vector < vectors; ++vector) {
    float* dst = output_interleaved + vector * (int64_t)(2 * total_dim);
    int dst_complex = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const __m512 x = _mm512_set1_ps(delta[vector * 3 + axis]);
      for (int base = 0; base < dims[axis]; base += 16) {
        const int count = dims[axis] - base < 16 ? dims[axis] - base : 16;
        const __mmask16 mask = mask16_for(count);
        const __m512 ft = _mm512_maskz_loadu_ps(mask, freq_tanh[axis] + base);
        const __m512 fr = _mm512_maskz_loadu_ps(mask, freq_residual[axis] + base);
        const __m512 scaled = _mm512_add_ps(_mm512_mul_ps(x, ft), _mm512_mul_ps(x, fr));
        const __m512 angle = _mm512_mul_ps(scaled, pi);
        __m512 sin_angle;
        __m512 cos_angle;
        sincos_ps(angle, &sin_angle, &cos_angle);
        store_interleaved(dst + 2 * dst_complex, cos_angle, sin_angle, count);
        dst_complex += count;
      }
    }
  }
  return 0;
}
