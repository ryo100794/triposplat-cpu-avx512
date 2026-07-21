#include <immintrin.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
  const __m512 two = _mm512_set1_ps(2.0f);
  const __m512 sign_mask = _mm512_set1_ps(-0.0f);
  const __m512 ax = _mm512_andnot_ps(sign_mask, x);
  const __m512 e = exp512_ps(_mm512_mul_ps(two, ax));
  __m512 y = _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(e, one)));
  y = _mm512_or_ps(y, _mm512_and_ps(x, sign_mask));
  return _mm512_mask_mov_ps(y, _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ), zero);
}

int triposplat_gelu_tanh_f32_avx512(
    const float* input, float* output, int64_t count, int threads) {
  if (input == 0 || output == 0) return -1;
  if (count <= 0) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  const int64_t chunks = (count + 15) / 16;
#pragma omp parallel for schedule(static)
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    const int64_t offset = chunk * 16;
    const int remaining = (int)(count - offset);
    const __mmask16 mask = remaining >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << remaining) - 1u);
    const __m512 x = _mm512_maskz_loadu_ps(mask, input + offset);
    const __m512 x2 = _mm512_mul_ps(x, x);
    const __m512 x3_term = _mm512_mul_ps(_mm512_set1_ps(0.044715f), _mm512_mul_ps(x2, x));
    const __m512 inner = _mm512_mul_ps(
        _mm512_set1_ps(0.7978845608028654f), _mm512_add_ps(x, x3_term));
    const __m512 y = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_set1_ps(0.5f), x),
        _mm512_add_ps(_mm512_set1_ps(1.0f), tanh512_ps(inner)));
    _mm512_mask_storeu_ps(output + offset, mask, y);
  }
  return 0;
}
