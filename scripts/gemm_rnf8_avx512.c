#include <immintrin.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline __m512 lookup16(const uint8_t* codes, const float* codebook) {
  const __m128i packed = _mm_loadu_si128((const __m128i*)codes);
  return _mm512_i32gather_ps(_mm512_cvtepu8_epi32(packed), codebook, 4);
}

static inline __m512 decode16(
    const uint8_t* code0,
    const uint8_t* code1,
    const uint8_t* code2,
    const float* codebook,
    __m512 scale0,
    __m512 scale1,
    __m512 scale2,
    int stages) {
  __m512 value = _mm512_mul_ps(lookup16(code0, codebook), scale0);
  if (stages >= 2) value = _mm512_fmadd_ps(lookup16(code1, codebook), scale1, value);
  if (stages >= 3) value = _mm512_fmadd_ps(lookup16(code2, codebook), scale2, value);
  return value;
}

static inline void kernel_8x16(
    const float* x,
    const uint8_t* codes0_t,
    const uint8_t* codes1_t,
    const uint8_t* codes2_t,
    const float* scales0,
    const float* scales1,
    const float* scales2,
    const float* codebook,
    const float* bias,
    float* out,
    int K,
    int N,
    int stride_x,
    int stride_out,
    int source_n,
    int output_n,
    int stages) {
  const __m512 scale0 = _mm512_loadu_ps(scales0 + source_n);
  const __m512 scale1 = _mm512_loadu_ps(scales1 + source_n);
  const __m512 scale2 = _mm512_loadu_ps(scales2 + source_n);
  __m512 c0 = _mm512_loadu_ps(bias + source_n);
  __m512 c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  for (int k = 0; k < K; ++k) {
    const int64_t offset = (int64_t)k * N + source_n;
    const __m512 w = decode16(codes0_t + offset, codes1_t + offset, codes2_t + offset,
                              codebook, scale0, scale1, scale2, stages);
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

static inline void kernel_1x16(
    const float* x,
    const uint8_t* codes0_t,
    const uint8_t* codes1_t,
    const uint8_t* codes2_t,
    const float* scales0,
    const float* scales1,
    const float* scales2,
    const float* codebook,
    const float* bias,
    float* out,
    int K,
    int N,
    int source_n,
    int output_n,
    int stages) {
  const __m512 scale0 = _mm512_loadu_ps(scales0 + source_n);
  const __m512 scale1 = _mm512_loadu_ps(scales1 + source_n);
  const __m512 scale2 = _mm512_loadu_ps(scales2 + source_n);
  __m512 c = _mm512_loadu_ps(bias + source_n);
  for (int k = 0; k < K; ++k) {
    const int64_t offset = (int64_t)k * N + source_n;
    const __m512 w = decode16(codes0_t + offset, codes1_t + offset, codes2_t + offset,
                              codebook, scale0, scale1, scale2, stages);
    c = _mm512_fmadd_ps(_mm512_set1_ps(x[k]), w, c);
  }
  _mm512_storeu_ps(out + output_n, c);
}

int triposplat_gemm_rnf8_avx512_range(
    const float* x,
    const uint8_t* codes0_t,
    const uint8_t* codes1_t,
    const uint8_t* codes2_t,
    const float* scales0,
    const float* scales1,
    const float* scales2,
    const float* codebook,
    const float* bias,
    float* out,
    int M,
    int K,
    int N,
    int output_start,
    int output_count,
    int stride_x,
    int stride_out,
    int threads,
    int stages) {
  if (x == 0 || codes0_t == 0 || codes1_t == 0 || codes2_t == 0 || scales0 == 0 ||
      scales1 == 0 || scales2 == 0 || codebook == 0 || bias == 0 || out == 0) return -1;
  if (M <= 0 || K <= 0 || N <= 0 || output_start < 0 || output_count <= 0 || stages < 2 || stages > 3) return -2;
  if (output_start + output_count > N || stride_x < K || stride_out < output_count) return -3;
  if ((output_start & 15) != 0 || (output_count & 15) != 0) return -4;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  const int m8 = (M / 8) * 8;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m8; i += 8) {
    for (int local_n = 0; local_n < output_count; local_n += 16) {
      kernel_8x16(x + (int64_t)i * stride_x, codes0_t, codes1_t, codes2_t,
                  scales0, scales1, scales2, codebook, bias, out + (int64_t)i * stride_out,
                  K, N, stride_x, stride_out, output_start + local_n, local_n, stages);
    }
  }
#pragma omp parallel for schedule(static)
  for (int i = m8; i < M; ++i) {
    for (int local_n = 0; local_n < output_count; local_n += 16) {
      kernel_1x16(x + (int64_t)i * stride_x, codes0_t, codes1_t, codes2_t,
                  scales0, scales1, scales2, codebook, bias, out + (int64_t)i * stride_out,
                  K, N, output_start + local_n, local_n, stages);
    }
  }
  return 0;
}

int triposplat_gemm_rnf8_avx512(
    const float* x,
    const uint8_t* codes0_t,
    const uint8_t* codes1_t,
    const uint8_t* codes2_t,
    const float* scales0,
    const float* scales1,
    const float* scales2,
    const float* codebook,
    const float* bias,
    float* out,
    int M,
    int K,
    int N,
    int stride_x,
    int stride_out,
    int threads,
    int stages) {
  return triposplat_gemm_rnf8_avx512_range(
      x, codes0_t, codes1_t, codes2_t, scales0, scales1, scales2, codebook, bias, out,
      M, K, N, 0, N, stride_x, stride_out, threads, stages);
}
