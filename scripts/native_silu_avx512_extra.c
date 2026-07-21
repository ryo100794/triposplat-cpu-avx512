#include <immintrin.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline __m512 silu_exp512_ps(__m512 x) {
  const __m512 one = _mm512_set1_ps(1.0f);
  x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
  x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
  __m512 fx = _mm512_floor_ps(_mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f)));
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

int triposplat_silu_f32_avx512(const float* input, float* output, int64_t count, int threads) {
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
    const __m512 denominator = _mm512_add_ps(_mm512_set1_ps(1.0f), silu_exp512_ps(_mm512_sub_ps(_mm512_setzero_ps(), x)));
    _mm512_mask_storeu_ps(output + offset, mask, _mm512_div_ps(x, denominator));
  }
  return 0;
}
