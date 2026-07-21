#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int triposplat_gemm_f32_avx512_tail(
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
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel for schedule(static)
  for (int i = 0; i < M; ++i) {
    const float* x_row = x + (int64_t)i * stride_x;
    float* out_row = out + (int64_t)i * stride_out;
    for (int n = 0; n < N; ++n) {
      float acc = bias[n];
      for (int k = 0; k < K; ++k) {
        acc = fmaf(x_row[k], weight_t[(int64_t)k * N + n], acc);
      }
      out_row[n] = acc;
    }
  }
  return 0;
}
