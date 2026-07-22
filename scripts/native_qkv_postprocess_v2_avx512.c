#include "native_qkv_postprocess_avx512.c"

static inline void transform_q64(
    const float* source,
    const float* frequency,
    const float* gamma,
    float* destination) {
  float rotated[HEAD_DIM] __attribute__((aligned(64)));
  __m512 sumsq = _mm512_setzero_ps();
  for (int d = 0; d < HEAD_DIM; d += SIMD_WIDTH) {
    const __m512 value = rope16(
        _mm512_loadu_ps(source + d), _mm512_loadu_ps(frequency + d));
    _mm512_store_ps(rotated + d, value);
    sumsq = _mm512_fmadd_ps(value, value, sumsq);
  }
  const float norm = sqrtf(_mm512_reduce_add_ps(sumsq));
  const __m512 factor = _mm512_set1_ps(8.0f / fmaxf(norm, 1.0e-12f));
  for (int d = 0; d < HEAD_DIM; d += SIMD_WIDTH) {
    const __m512 value = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_load_ps(rotated + d), factor),
        _mm512_loadu_ps(gamma + d));
    _mm512_storeu_ps(destination + d, value);
  }
}

static inline void copy_pack_tile(
    const float tile[TOKEN_TILE][HEAD_DIM],
    float* packed,
    int first_token,
    int rows,
    int length,
    int length_padded) {
  const int clear_tail = first_token + rows == length;
  for (int d = 0; d < HEAD_DIM; ++d) {
    float* destination = packed + (int64_t)d * length_padded + first_token;
    for (int row = 0; row < rows; ++row) destination[row] = tile[row][d];
    if (clear_tail) {
      for (int row = rows; first_token + row < length_padded; ++row) {
        destination[row] = 0.0f;
      }
    }
  }
}

int triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2(
    const float* qkv,
    const float* frequencies,
    const float* q_gamma,
    const float* k_gamma,
    float* q_bhld,
    float* packed_k,
    float* packed_v,
    int B,
    int L,
    int H,
    int D,
    int L_padded,
    int frequency_batches,
    int threads) {
  if (qkv == 0 || frequencies == 0 || q_gamma == 0 || k_gamma == 0 ||
      q_bhld == 0 || packed_k == 0 || packed_v == 0) return -1;
  if (B <= 0 || L <= 0 || H <= 0 || D != HEAD_DIM || L_padded < L ||
      (L_padded & 15) != 0 || (frequency_batches != 1 && frequency_batches != B)) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const int token_tiles = (L + TOKEN_TILE - 1) / TOKEN_TILE;
  const int64_t bh_count = (int64_t)B * H;
#pragma omp parallel for schedule(static)
  for (int64_t work = 0; work < bh_count * token_tiles; ++work) {
    const int tile = (int)(work % token_tiles);
    const int64_t bh = work / token_tiles;
    const int b = (int)(bh / H);
    const int h = (int)(bh % H);
    const int first_token = tile * TOKEN_TILE;
    const int rows = first_token + TOKEN_TILE <= L ? TOKEN_TILE : L - first_token;
    const int frequency_b = frequency_batches == 1 ? 0 : b;
    float k_tile[TOKEN_TILE][HEAD_DIM] __attribute__((aligned(64)));
    float v_tile[TOKEN_TILE][HEAD_DIM] __attribute__((aligned(64)));

    for (int row = 0; row < rows; ++row) {
      const int token = first_token + row;
      const int64_t qkv_row = (int64_t)b * L + token;
      const float* q_source = qkv + (qkv_row * 3 * H + h) * D;
      const float* k_source = q_source + (int64_t)H * D;
      const float* v_source = k_source + (int64_t)H * D;
      const float* frequency = frequencies + (((int64_t)frequency_b * L + token) * H + h) * D;
      float* q_destination = q_bhld + (bh * L + token) * D;
      transform_q64(q_source, frequency, q_gamma + (int64_t)h * D, q_destination);
      transform_q64(k_source, frequency, k_gamma + (int64_t)h * D, k_tile[row]);
      for (int d = 0; d < D; d += SIMD_WIDTH) {
        _mm512_store_ps(v_tile[row] + d, _mm512_loadu_ps(v_source + d));
      }
    }

    copy_pack_tile(
        k_tile, packed_k + bh * D * (int64_t)L_padded,
        first_token, rows, L, L_padded);
    copy_pack_tile(
        v_tile, packed_v + bh * D * (int64_t)L_padded,
        first_token, rows, L, L_padded);
  }
  return 0;
}

int triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2(
    const float* q,
    const float* kv,
    const float* frequencies,
    const int64_t* selected_indices,
    const float* q_gamma,
    const float* k_gamma,
    float* q_bhld,
    float* packed_k,
    float* packed_v,
    int B,
    int Lq,
    int Lk,
    int H,
    int D,
    int Lk_padded,
    int frequency_batches,
    int threads) {
  if (q == 0 || kv == 0 || frequencies == 0 || selected_indices == 0 ||
      q_gamma == 0 || k_gamma == 0 || q_bhld == 0 || packed_k == 0 || packed_v == 0) return -1;
  if (B <= 0 || Lq <= 0 || Lk <= 0 || H <= 0 || D != HEAD_DIM ||
      Lk_padded < Lk || (Lk_padded & 15) != 0 ||
      (frequency_batches != 1 && frequency_batches != B)) return -2;
  for (int query = 0; query < Lq; ++query) {
    if (selected_indices[query] < 0 || selected_indices[query] >= Lk) return -3;
  }
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const int64_t bh_count = (int64_t)B * H;
  const int q_tiles = (Lq + TOKEN_TILE - 1) / TOKEN_TILE;
  const int kv_tiles = (Lk + TOKEN_TILE - 1) / TOKEN_TILE;
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (int64_t work = 0; work < bh_count * q_tiles; ++work) {
      const int tile = (int)(work % q_tiles);
      const int64_t bh = work / q_tiles;
      const int b = (int)(bh / H);
      const int h = (int)(bh % H);
      const int first_query = tile * TOKEN_TILE;
      const int rows = first_query + TOKEN_TILE <= Lq ? TOKEN_TILE : Lq - first_query;
      const int frequency_b = frequency_batches == 1 ? 0 : b;
      for (int row = 0; row < rows; ++row) {
        const int query = first_query + row;
        const int64_t key_index = selected_indices[query];
        const float* q_source = q + (((int64_t)b * Lq + query) * H + h) * D;
        const float* frequency = frequencies + (((int64_t)frequency_b * Lk + key_index) * H + h) * D;
        float* q_destination = q_bhld + (bh * Lq + query) * D;
        transform_q64(q_source, frequency, q_gamma + (int64_t)h * D, q_destination);
      }
    }

#pragma omp for schedule(static)
    for (int64_t work = 0; work < bh_count * kv_tiles; ++work) {
      const int tile = (int)(work % kv_tiles);
      const int64_t bh = work / kv_tiles;
      const int b = (int)(bh / H);
      const int h = (int)(bh % H);
      const int first_token = tile * TOKEN_TILE;
      const int rows = first_token + TOKEN_TILE <= Lk ? TOKEN_TILE : Lk - first_token;
      const int frequency_b = frequency_batches == 1 ? 0 : b;
      float k_tile[TOKEN_TILE][HEAD_DIM] __attribute__((aligned(64)));
      float v_tile[TOKEN_TILE][HEAD_DIM] __attribute__((aligned(64)));
      for (int row = 0; row < rows; ++row) {
        const int token = first_token + row;
        const float* k_source = kv + (((int64_t)b * Lk + token) * 2 * H + h) * D;
        const float* v_source = k_source + (int64_t)H * D;
        const float* frequency = frequencies + (((int64_t)frequency_b * Lk + token) * H + h) * D;
        transform_q64(k_source, frequency, k_gamma + (int64_t)h * D, k_tile[row]);
        for (int d = 0; d < D; d += SIMD_WIDTH) {
          _mm512_store_ps(v_tile[row] + d, _mm512_loadu_ps(v_source + d));
        }
      }
      copy_pack_tile(
          k_tile, packed_k + bh * D * (int64_t)Lk_padded,
          first_token, rows, Lk, Lk_padded);
      copy_pack_tile(
          v_tile, packed_v + bh * D * (int64_t)Lk_padded,
          first_token, rows, Lk, Lk_padded);
    }
  }
  return 0;
}

int triposplat_qkv_postprocess_v2_tail_only_clear(void) {
  return 1;
}
