#include "native_nf24_mlp_avx512.c"

int triposplat_mlp_nf24_gelu_chunked_f32_avx512(
    const float* x,
    const uint8_t* w1_code01_t,
    const uint8_t* w1_code2_t,
    const float* w1_scales,
    const float* w1_bias,
    const uint8_t* w2_code01_t,
    const uint8_t* w2_code2_t,
    const float* w2_scales,
    const float* w2_bias,
    float* hidden,
    float* out,
    int M,
    int K,
    int H,
    int N,
    int stride_x,
    int stride_out,
    int hidden_capacity_rows,
    int chunk_rows,
    int threads) {
  if (x == 0 || w1_code01_t == 0 || w1_code2_t == 0 || w1_scales == 0 ||
      w1_bias == 0 || w2_code01_t == 0 || w2_code2_t == 0 || w2_scales == 0 ||
      w2_bias == 0 || hidden == 0 || out == 0) return -1;
  if (M <= 0 || K != INPUT_FEATURES || H != HIDDEN_FEATURES || N != OUTPUT_FEATURES ||
      stride_x < K || stride_out < N || chunk_rows <= 0 ||
      hidden_capacity_rows < chunk_rows) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

#pragma omp parallel
  {
    for (int start = 0; start < M; start += chunk_rows) {
      const int count = start + chunk_rows <= M ? chunk_rows : M - start;
      const int m_full = (count / ROW_TILE) * ROW_TILE;

#pragma omp for schedule(static)
      for (int local_i = 0; local_i < m_full; local_i += ROW_TILE) {
        const float* x_tile = x + (int64_t)(start + local_i) * stride_x;
        float* hidden_tile = hidden + (int64_t)local_i * H;
        for (int hn = 0; hn < H; hn += SIMD_WIDTH) {
          gemm_8x16(x_tile, w1_code01_t, w1_code2_t, w1_scales, w1_bias,
                    hidden_tile, K, H, stride_x, H, hn);
        }
      }

#pragma omp for schedule(static)
      for (int local_i = m_full; local_i < count; ++local_i) {
        const float* x_row = x + (int64_t)(start + local_i) * stride_x;
        float* hidden_row = hidden + (int64_t)local_i * H;
        for (int hn = 0; hn < H; hn += SIMD_WIDTH) {
          gemm_1x16(x_row, w1_code01_t, w1_code2_t, w1_scales, w1_bias,
                    hidden_row, K, H, hn);
        }
      }

#pragma omp for schedule(static)
      for (int64_t offset = 0; offset < (int64_t)count * H; offset += SIMD_WIDTH) {
        gelu_tanh_inplace(hidden + offset, SIMD_WIDTH);
      }

#pragma omp for schedule(static)
      for (int local_i = 0; local_i < m_full; local_i += ROW_TILE) {
        const float* hidden_tile = hidden + (int64_t)local_i * H;
        float* out_tile = out + (int64_t)(start + local_i) * stride_out;
        for (int n = 0; n < N; n += SIMD_WIDTH) {
          gemm_8x16(hidden_tile, w2_code01_t, w2_code2_t, w2_scales, w2_bias,
                    out_tile, H, N, H, stride_out, n);
        }
      }

#pragma omp for schedule(static)
      for (int local_i = m_full; local_i < count; ++local_i) {
        const float* hidden_row = hidden + (int64_t)local_i * H;
        float* out_row = out + (int64_t)(start + local_i) * stride_out;
        for (int n = 0; n < N; n += SIMD_WIDTH) {
          gemm_1x16(hidden_row, w2_code01_t, w2_code2_t, w2_scales, w2_bias,
                    out_row, H, N, n);
        }
      }
    }
  }
  return 0;
}
