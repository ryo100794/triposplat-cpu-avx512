#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int triposplat_gemm_rnf8_avx512_tail(
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
  if (x == 0 || codes0_t == 0 || codes1_t == 0 || codes2_t == 0 || scales0 == 0 ||
      scales1 == 0 || scales2 == 0 || codebook == 0 || bias == 0 || out == 0) return -1;
  if (M <= 0 || K <= 0 || N <= 0 || stride_x < K || stride_out < N || stages < 2 || stages > 3) return -2;
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
        const int64_t offset = (int64_t)k * N + n;
        float weight = codebook[codes0_t[offset]] * scales0[n];
        weight += codebook[codes1_t[offset]] * scales1[n];
        if (stages >= 3) weight += codebook[codes2_t[offset]] * scales2[n];
        acc = fmaf(x_row[k], weight, acc);
      }
      out_row[n] = acc;
    }
  }
  return 0;
}
