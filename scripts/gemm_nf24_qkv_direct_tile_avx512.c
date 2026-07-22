#define triposplat_gemm_nf24_qkv_rope_rmsnorm_pack_f32_avx512 \
  triposplat_gemm_nf24_qkv_rope_rmsnorm_pack_f32_avx512_head_order
#include "gemm_nf24_qkv_direct_avx512.c"
#undef triposplat_gemm_nf24_qkv_rope_rmsnorm_pack_f32_avx512

enum {
  TRIPOSPLAT_QKV_HEADS = 16,
  TRIPOSPLAT_QKV_CHANNELS = 1024,
  TRIPOSPLAT_QKV_OUTPUTS = 3072,
};

static inline void qkv_store_cached_rows(
    const float* qkv_tile,
    int tile_stride,
    const float* frequencies,
    const float* q_gamma,
    const float* k_gamma,
    float* q_bhld,
    float* packed_k,
    float* packed_v,
    int first_row,
    int rows,
    int L,
    int H,
    int L_padded,
    int frequency_batches) {
  for (int local_row = 0; local_row < rows; ++local_row) {
    const int flat_row = first_row + local_row;
    const int b = flat_row / L;
    const int token = flat_row - b * L;
    const int frequency_b = frequency_batches == 1 ? 0 : b;
    const float* source = qkv_tile + (int64_t)local_row * tile_stride;
    for (int head = 0; head < H; ++head) {
      const int64_t bh = (int64_t)b * H + head;
      const float* frequency = frequencies +
          (((int64_t)frequency_b * L + token) * H + head) * QKV_HEAD_DIM;
      float* q_destination = q_bhld +
          (bh * L + token) * QKV_HEAD_DIM;
      qkv_transform64(
          source + head * QKV_HEAD_DIM,
          frequency,
          q_gamma + (int64_t)head * QKV_HEAD_DIM,
          q_destination);

      float k_normalized[QKV_HEAD_DIM] __attribute__((aligned(64)));
      qkv_transform64(
          source + H * QKV_HEAD_DIM + head * QKV_HEAD_DIM,
          frequency,
          k_gamma + (int64_t)head * QKV_HEAD_DIM,
          k_normalized);
      const float* v_source = source + 2 * H * QKV_HEAD_DIM + head * QKV_HEAD_DIM;
      float* packed_k_bh = packed_k + bh * QKV_HEAD_DIM * (int64_t)L_padded;
      float* packed_v_bh = packed_v + bh * QKV_HEAD_DIM * (int64_t)L_padded;
      for (int d = 0; d < QKV_HEAD_DIM; ++d) {
        packed_k_bh[(int64_t)d * L_padded + token] = k_normalized[d];
        packed_v_bh[(int64_t)d * L_padded + token] = v_source[d];
      }
    }
  }
}

int triposplat_gemm_nf24_qkv_rope_rmsnorm_pack_f32_avx512(
    const float* x,
    const uint8_t* codes0_t,
    const uint8_t* codes1_t,
    const uint8_t* codes2_t,
    const float* scales0,
    const float* scales1,
    const float* scales2,
    const float* codebook,
    const float* bias,
    const float* frequencies,
    const float* q_gamma,
    const float* k_gamma,
    float* q_bhld,
    float* packed_k,
    float* packed_v,
    int B,
    int L,
    int K,
    int N,
    int H,
    int D,
    int L_padded,
    int frequency_batches,
    int threads,
    int stages) {
  if (x == 0 || codes0_t == 0 || codes1_t == 0 || codes2_t == 0 ||
      scales0 == 0 || scales1 == 0 || scales2 == 0 || codebook == 0 ||
      bias == 0 || frequencies == 0 || q_gamma == 0 || k_gamma == 0 ||
      q_bhld == 0 || packed_k == 0 || packed_v == 0) return -1;
  if (B <= 0 || L <= 0 || K != TRIPOSPLAT_QKV_CHANNELS ||
      H != TRIPOSPLAT_QKV_HEADS || D != QKV_HEAD_DIM ||
      N != TRIPOSPLAT_QKV_OUTPUTS || L_padded < L || (L_padded & 15) != 0 ||
      (frequency_batches != 1 && frequency_batches != B) || stages != 3) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const int M = B * L;
  const int m_full = (M / ROW_TILE) * ROW_TILE;
  const int64_t packed_rows = (int64_t)B * H * D;
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t packed_row = 0; packed_row < packed_rows; ++packed_row) {
      float* k_tail = packed_k + packed_row * L_padded + L;
      float* v_tail = packed_v + packed_row * L_padded + L;
      for (int token = L; token < L_padded; ++token) {
        k_tail[token - L] = 0.0f;
        v_tail[token - L] = 0.0f;
      }
    }

#pragma omp for schedule(static)
    for (int first_row = 0; first_row < m_full; first_row += ROW_TILE) {
      float qkv_tile[ROW_TILE][TRIPOSPLAT_QKV_OUTPUTS] __attribute__((aligned(64)));
      for (int output = 0; output < N; output += QKV_SIMD_WIDTH) {
        kernel_mx16(
            x + (int64_t)first_row * K,
            codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
            codebook, bias, &qkv_tile[0][0], K, N, K, N,
            output, output, stages);
      }
      qkv_store_cached_rows(
          &qkv_tile[0][0], N, frequencies, q_gamma, k_gamma,
          q_bhld, packed_k, packed_v, first_row, ROW_TILE,
          L, H, L_padded, frequency_batches);
    }

#pragma omp for schedule(static)
    for (int first_row = m_full; first_row < M; ++first_row) {
      float qkv_row[TRIPOSPLAT_QKV_OUTPUTS] __attribute__((aligned(64)));
      for (int output = 0; output < N; output += QKV_SIMD_WIDTH) {
        kernel_1x16(
            x + (int64_t)first_row * K,
            codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
            codebook, bias, qkv_row, K, N, output, output, stages);
      }
      qkv_store_cached_rows(
          qkv_row, N, frequencies, q_gamma, k_gamma,
          q_bhld, packed_k, packed_v, first_row, 1,
          L, H, L_padded, frequency_batches);
    }
  }
  return 0;
}

int triposplat_gemm_nf24_qkv_direct_tile_bytes(void) {
  return ROW_TILE * TRIPOSPLAT_QKV_OUTPUTS * (int)sizeof(float);
}
