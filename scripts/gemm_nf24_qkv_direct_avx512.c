#include <math.h>
#include <stdint.h>

#include "gemm_nf24_i16_avx512.c"

enum {
  QKV_HEAD_DIM = 64,
  QKV_SIMD_WIDTH = 16,
};

static inline __m512 qkv_rope16(__m512 x, __m512 frequency) {
  const __m512 x_re = _mm512_moveldup_ps(x);
  const __m512 x_im = _mm512_movehdup_ps(x);
  const __m512 frequency_swap = _mm512_permute_ps(frequency, 0xb1);
  return _mm512_fmaddsub_ps(x_re, frequency, _mm512_mul_ps(x_im, frequency_swap));
}

static inline void qkv_transform64(
    const float* source,
    const float* frequency,
    const float* gamma,
    float* destination) {
  float rotated[QKV_HEAD_DIM] __attribute__((aligned(64)));
  __m512 sumsq = _mm512_setzero_ps();
  for (int d = 0; d < QKV_HEAD_DIM; d += QKV_SIMD_WIDTH) {
    const __m512 value = qkv_rope16(
        _mm512_loadu_ps(source + d), _mm512_loadu_ps(frequency + d));
    _mm512_store_ps(rotated + d, value);
    sumsq = _mm512_fmadd_ps(value, value, sumsq);
  }
  const float norm = sqrtf(_mm512_reduce_add_ps(sumsq));
  const __m512 factor = _mm512_set1_ps(8.0f / fmaxf(norm, 1.0e-12f));
  for (int d = 0; d < QKV_HEAD_DIM; d += QKV_SIMD_WIDTH) {
    const __m512 value = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_load_ps(rotated + d), factor),
        _mm512_loadu_ps(gamma + d));
    _mm512_storeu_ps(destination + d, value);
  }
}

static inline void qkv_store_rows(
    const float q_tile[ROW_TILE][QKV_HEAD_DIM],
    const float k_tile[ROW_TILE][QKV_HEAD_DIM],
    const float v_tile[ROW_TILE][QKV_HEAD_DIM],
    const float* frequencies,
    const float* q_gamma,
    const float* k_gamma,
    float* q_bhld,
    float* packed_k,
    float* packed_v,
    int first_row,
    int rows,
    int B,
    int L,
    int H,
    int L_padded,
    int frequency_batches,
    int head) {
  (void)B;
  for (int local_row = 0; local_row < rows; ++local_row) {
    const int flat_row = first_row + local_row;
    const int b = flat_row / L;
    const int token = flat_row - b * L;
    const int frequency_b = frequency_batches == 1 ? 0 : b;
    const int64_t bh = (int64_t)b * H + head;
    const float* frequency = frequencies +
        (((int64_t)frequency_b * L + token) * H + head) * QKV_HEAD_DIM;
    float* q_destination = q_bhld +
        (bh * L + token) * QKV_HEAD_DIM;
    qkv_transform64(
        q_tile[local_row], frequency,
        q_gamma + (int64_t)head * QKV_HEAD_DIM, q_destination);

    float k_normalized[QKV_HEAD_DIM] __attribute__((aligned(64)));
    qkv_transform64(
        k_tile[local_row], frequency,
        k_gamma + (int64_t)head * QKV_HEAD_DIM, k_normalized);
    float* packed_k_bh = packed_k + bh * QKV_HEAD_DIM * (int64_t)L_padded;
    float* packed_v_bh = packed_v + bh * QKV_HEAD_DIM * (int64_t)L_padded;
    for (int d = 0; d < QKV_HEAD_DIM; ++d) {
      packed_k_bh[(int64_t)d * L_padded + token] = k_normalized[d];
      packed_v_bh[(int64_t)d * L_padded + token] = v_tile[local_row][d];
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
  if (B <= 0 || L <= 0 || K <= 0 || H <= 0 || D != QKV_HEAD_DIM ||
      N != 3 * H * D || L_padded < L || (L_padded & 15) != 0 ||
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
      for (int head = 0; head < H; ++head) {
        float q_tile[ROW_TILE][QKV_HEAD_DIM] __attribute__((aligned(64)));
        float k_tile[ROW_TILE][QKV_HEAD_DIM] __attribute__((aligned(64)));
        float v_tile[ROW_TILE][QKV_HEAD_DIM] __attribute__((aligned(64)));
        for (int d = 0; d < D; d += QKV_SIMD_WIDTH) {
          kernel_mx16(
              x + (int64_t)first_row * K,
              codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
              codebook, bias, &q_tile[0][0], K, N, K, D,
              head * D + d, d, stages);
          kernel_mx16(
              x + (int64_t)first_row * K,
              codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
              codebook, bias, &k_tile[0][0], K, N, K, D,
              H * D + head * D + d, d, stages);
          kernel_mx16(
              x + (int64_t)first_row * K,
              codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
              codebook, bias, &v_tile[0][0], K, N, K, D,
              2 * H * D + head * D + d, d, stages);
        }
        qkv_store_rows(
            q_tile, k_tile, v_tile, frequencies, q_gamma, k_gamma,
            q_bhld, packed_k, packed_v, first_row, ROW_TILE,
            B, L, H, L_padded, frequency_batches, head);
      }
    }

#pragma omp for schedule(static)
    for (int first_row = m_full; first_row < M; ++first_row) {
      for (int head = 0; head < H; ++head) {
        float q_tile[ROW_TILE][QKV_HEAD_DIM] __attribute__((aligned(64)));
        float k_tile[ROW_TILE][QKV_HEAD_DIM] __attribute__((aligned(64)));
        float v_tile[ROW_TILE][QKV_HEAD_DIM] __attribute__((aligned(64)));
        for (int d = 0; d < D; d += QKV_SIMD_WIDTH) {
          kernel_1x16(
              x + (int64_t)first_row * K,
              codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
              codebook, bias, &q_tile[0][0], K, N, head * D + d, d, stages);
          kernel_1x16(
              x + (int64_t)first_row * K,
              codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
              codebook, bias, &k_tile[0][0], K, N,
              H * D + head * D + d, d, stages);
          kernel_1x16(
              x + (int64_t)first_row * K,
              codes0_t, codes1_t, codes2_t, scales0, scales1, scales2,
              codebook, bias, &v_tile[0][0], K, N,
              2 * H * D + head * D + d, d, stages);
        }
        qkv_store_rows(
            q_tile, k_tile, v_tile, frequencies, q_gamma, k_gamma,
            q_bhld, packed_k, packed_v, first_row, 1,
            B, L, H, L_padded, frequency_batches, head);
      }
    }
  }
  return 0;
}

int triposplat_gemm_nf24_qkv_direct_row_tile(void) {
  return ROW_TILE;
}

int triposplat_gemm_nf24_qkv_direct_residual_mode(void) {
  return 4;
}
