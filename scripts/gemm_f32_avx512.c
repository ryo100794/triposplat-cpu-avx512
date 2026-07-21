#include <immintrin.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline void kernel_8x16(
    const float* x,
    const float* weight_t,
    const float* bias,
    float* out,
    int K,
    int N,
    int stride_x,
    int stride_out,
    int n) {
  __m512 c0 = _mm512_loadu_ps(bias + n);
  __m512 c1 = c0;
  __m512 c2 = c0;
  __m512 c3 = c0;
  __m512 c4 = c0;
  __m512 c5 = c0;
  __m512 c6 = c0;
  __m512 c7 = c0;

  for (int k = 0; k < K; ++k) {
    const __m512 w = _mm512_loadu_ps(weight_t + (int64_t)k * N + n);
    c0 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)0 * stride_x + k]), w, c0);
    c1 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)1 * stride_x + k]), w, c1);
    c2 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)2 * stride_x + k]), w, c2);
    c3 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)3 * stride_x + k]), w, c3);
    c4 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)4 * stride_x + k]), w, c4);
    c5 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)5 * stride_x + k]), w, c5);
    c6 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)6 * stride_x + k]), w, c6);
    c7 = _mm512_fmadd_ps(_mm512_set1_ps(x[(int64_t)7 * stride_x + k]), w, c7);
  }

  _mm512_storeu_ps(out + (int64_t)0 * stride_out + n, c0);
  _mm512_storeu_ps(out + (int64_t)1 * stride_out + n, c1);
  _mm512_storeu_ps(out + (int64_t)2 * stride_out + n, c2);
  _mm512_storeu_ps(out + (int64_t)3 * stride_out + n, c3);
  _mm512_storeu_ps(out + (int64_t)4 * stride_out + n, c4);
  _mm512_storeu_ps(out + (int64_t)5 * stride_out + n, c5);
  _mm512_storeu_ps(out + (int64_t)6 * stride_out + n, c6);
  _mm512_storeu_ps(out + (int64_t)7 * stride_out + n, c7);
}

static inline void kernel_1x16(
    const float* x,
    const float* weight_t,
    const float* bias,
    float* out,
    int K,
    int N,
    int n) {
  __m512 c = _mm512_loadu_ps(bias + n);
  for (int k = 0; k < K; ++k) {
    const __m512 w = _mm512_loadu_ps(weight_t + (int64_t)k * N + n);
    c = _mm512_fmadd_ps(_mm512_set1_ps(x[k]), w, c);
  }
  _mm512_storeu_ps(out + n, c);
}

int triposplat_gemm_f32_avx512(
    const float* x,
    const float* weight_t,
    const float* bias,
    float* out,
    int M,
    int K,
    int N,
    int stride_x,
    int stride_out,
    int threads) {
  if (x == 0 || weight_t == 0 || bias == 0 || out == 0) return -1;
  if (M <= 0 || K <= 0 || N <= 0 || stride_x < K || stride_out < N) return -2;
  if ((N & 15) != 0) return -3;

#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const int m8 = (M / 8) * 8;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < m8; i += 8) {
    const float* x_block = x + (int64_t)i * stride_x;
    float* out_block = out + (int64_t)i * stride_out;
    for (int n = 0; n < N; n += 16) {
      kernel_8x16(x_block, weight_t, bias, out_block, K, N, stride_x, stride_out, n);
    }
  }

#pragma omp parallel for schedule(static)
  for (int i = m8; i < M; ++i) {
    const float* x_row = x + (int64_t)i * stride_x;
    float* out_row = out + (int64_t)i * stride_out;
    for (int n = 0; n < N; n += 16) {
      kernel_1x16(x_row, weight_t, bias, out_row, K, N, n);
    }
  }
  return 0;
}
