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

int triposplat_pcd_position_embedding_f32_avx512(
    const float* positions, const float* frequencies, float* output,
    int64_t rows, int input_dims, int frequency_dim, int channels, int double_pi, int threads) {
  if (positions == 0 || frequencies == 0 || output == 0) return -1;
  if (rows <= 0 || input_dims <= 0 || frequency_dim <= 0 ||
      channels < input_dims * frequency_dim * 2) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  const __m512 pi = _mm512_set1_ps(3.14159274101257324219f);
#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    float* dst = output + row * (int64_t)channels;
    for (int axis = 0; axis < input_dims; ++axis) {
      const __m512 x = _mm512_set1_ps(positions[row * input_dims + axis]);
      float* sin_dst = dst + axis * (2 * frequency_dim);
      float* cos_dst = sin_dst + frequency_dim;
      for (int base = 0; base < frequency_dim; base += 16) {
        const int count = frequency_dim - base < 16 ? frequency_dim - base : 16;
        const __mmask16 mask = mask16_for(count);
        const __m512 frequency = _mm512_maskz_loadu_ps(mask, frequencies + base);
        __m512 angle = _mm512_mul_ps(x, frequency);
        if (double_pi) angle = _mm512_add_ps(angle, angle);
        angle = _mm512_mul_ps(angle, pi);
        __m512 sin_angle;
        __m512 cos_angle;
        sincos_ps(angle, &sin_angle, &cos_angle);
        _mm512_mask_storeu_ps(sin_dst + base, mask, sin_angle);
        _mm512_mask_storeu_ps(cos_dst + base, mask, cos_angle);
      }
    }
    const int used = input_dims * frequency_dim * 2;
    for (int base = used; base < channels; base += 16) {
      _mm512_mask_storeu_ps(dst + base, mask16_for(channels - base), _mm512_setzero_ps());
    }
  }
  return 0;
}

int triposplat_timestep_embedding_f32_avx512(
    const float* timesteps, const float* frequencies, float* output,
    int batch, int half_dim, int output_dim, int threads) {
  if (timesteps == 0 || frequencies == 0 || output == 0) return -1;
  if (batch <= 0 || half_dim <= 0 || output_dim < half_dim * 2) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel for schedule(static)
  for (int sample = 0; sample < batch; ++sample) {
    const __m512 timestep = _mm512_set1_ps(timesteps[sample]);
    float* cos_dst = output + sample * (int64_t)output_dim;
    float* sin_dst = cos_dst + half_dim;
    for (int base = 0; base < half_dim; base += 16) {
      const int count = half_dim - base < 16 ? half_dim - base : 16;
      const __mmask16 mask = mask16_for(count);
      const __m512 angle = _mm512_mul_ps(
          timestep, _mm512_maskz_loadu_ps(mask, frequencies + base));
      __m512 sin_angle;
      __m512 cos_angle;
      sincos_ps(angle, &sin_angle, &cos_angle);
      _mm512_mask_storeu_ps(cos_dst + base, mask, cos_angle);
      _mm512_mask_storeu_ps(sin_dst + base, mask, sin_angle);
    }
    for (int base = half_dim * 2; base < output_dim; base += 16) {
      _mm512_mask_storeu_ps(cos_dst + base, mask16_for(output_dim - base), _mm512_setzero_ps());
    }
  }
  return 0;
}
