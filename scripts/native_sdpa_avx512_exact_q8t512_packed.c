#include "native_sdpa_avx512_exact_q8t512.c"

int triposplat_sdpa_f32_avx512_exact_q8t512_packed_blhd(
    const float* restrict q,
    const float* restrict packed_k,
    const float* restrict packed_v,
    const float* restrict key_bias_or_null,
    float* restrict out_blhd,
    int B,
    int H,
    int Lq,
    int Lk,
    int D,
    int Lkp,
    int has_key_bias,
    int key_bias_len,
    int threads) {
  if (q == NULL || packed_k == NULL || packed_v == NULL || out_blhd == NULL) return -1;
  if (B <= 0 || H <= 0 || Lq <= 0 || Lk <= 0 || D != HEAD_DIM ||
      Lkp < Lk || (Lkp & 15) != 0) return -2;
  if (has_key_bias && (key_bias_or_null == NULL || key_bias_len != Lk)) return -3;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const float scale = 0.125f;
  const int64_t bh_count = (int64_t)B * H;
  const int64_t q_blocks = (Lq + QUERY_BLOCK - 1) / QUERY_BLOCK;
  const int64_t total_q_blocks = bh_count * q_blocks;

#pragma omp parallel for schedule(static)
  for (int64_t block_index = 0; block_index < total_q_blocks; ++block_index) {
    const int64_t qb = block_index % q_blocks;
    const int64_t bh = block_index / q_blocks;
    const int b = (int)(bh / H);
    const int h = (int)(bh % H);
    const int i0 = (int)(qb * QUERY_BLOCK);
    const int rows = i0 + QUERY_BLOCK <= Lq ? QUERY_BLOCK : Lq - i0;
    const float* q_base = q + bh * (int64_t)Lq * HEAD_DIM;
    const float* pk_base = packed_k + bh * HEAD_DIM * (int64_t)Lkp;
    const float* pv_base = packed_v + bh * HEAD_DIM * (int64_t)Lkp;

    float qrow[QUERY_BLOCK][HEAD_DIM] __attribute__((aligned(64)));
    for (int r = 0; r < rows; ++r) {
      const float* source = q_base + (int64_t)(i0 + r) * HEAD_DIM;
      for (int d = 0; d < HEAD_DIM; ++d) qrow[r][d] = source[d];
    }

    float global_max[QUERY_BLOCK];
    float global_sum[QUERY_BLOCK];
    __m512 global_acc[QUERY_BLOCK][HEAD_DIM / SIMD_WIDTH];
    for (int r = 0; r < QUERY_BLOCK; ++r) {
      global_max[r] = -INFINITY;
      global_sum[r] = 0.0f;
      for (int dc = 0; dc < HEAD_DIM / SIMD_WIDTH; ++dc) {
        global_acc[r][dc] = _mm512_setzero_ps();
      }
    }

    for (int jb = 0; jb < Lk; jb += KEY_TILE) {
      const int count = jb + KEY_TILE <= Lk ? KEY_TILE : Lk - jb;
      const int chunks = (count + SIMD_WIDTH - 1) / SIMD_WIDTH;
      float scores[QUERY_BLOCK][KEY_TILE] __attribute__((aligned(64)));
      float local_max[QUERY_BLOCK];
      float local_sum[QUERY_BLOCK];
      float local_out[QUERY_BLOCK][HEAD_DIM] __attribute__((aligned(64)));
      for (int r = 0; r < QUERY_BLOCK; ++r) {
        local_max[r] = -INFINITY;
        local_sum[r] = 0.0f;
      }

      for (int tc = 0; tc < chunks; ++tc) {
        const int key = jb + tc * SIMD_WIDTH;
        __m512 score_vec[QUERY_BLOCK];
        for (int r = 0; r < QUERY_BLOCK; ++r) score_vec[r] = _mm512_setzero_ps();
        for (int d = 0; d < HEAD_DIM; ++d) {
          const __m512 key_value = _mm512_loadu_ps(pk_base + (int64_t)d * Lkp + key);
#pragma GCC unroll 8
          for (int r = 0; r < rows; ++r) {
            score_vec[r] = _mm512_fmadd_ps(_mm512_set1_ps(qrow[r][d]), key_value, score_vec[r]);
          }
        }
        const int valid = Lk - key < SIMD_WIDTH ? Lk - key : SIMD_WIDTH;
        const __mmask16 valid_mask = (__mmask16)((1u << valid) - 1u);
        __m512 bias = _mm512_setzero_ps();
        if (has_key_bias) bias = _mm512_maskz_loadu_ps(valid_mask, key_bias_or_null + key);
#pragma GCC unroll 8
        for (int r = 0; r < rows; ++r) {
          __m512 score = _mm512_mul_ps(score_vec[r], _mm512_set1_ps(scale));
          if (has_key_bias) score = _mm512_add_ps(score, bias);
          if (valid < SIMD_WIDTH) {
            score = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), valid_mask, score);
          }
          _mm512_store_ps(scores[r] + tc * SIMD_WIDTH, score);
          const float maximum = _mm512_reduce_max_ps(score);
          if (maximum > local_max[r]) local_max[r] = maximum;
        }
      }

      for (int r = 0; r < rows; ++r) {
        const __m512 maximum = _mm512_set1_ps(local_max[r]);
        for (int tc = 0; tc < chunks; ++tc) {
          const int key = jb + tc * SIMD_WIDTH;
          const int valid = Lk - key < SIMD_WIDTH ? Lk - key : SIMD_WIDTH;
          const __mmask16 valid_mask = (__mmask16)((1u << valid) - 1u);
          __m512 weight = exp512_ps(
              _mm512_sub_ps(_mm512_load_ps(scores[r] + tc * SIMD_WIDTH), maximum));
          if (valid < SIMD_WIDTH) weight = _mm512_maskz_mov_ps(valid_mask, weight);
          _mm512_store_ps(scores[r] + tc * SIMD_WIDTH, weight);
          local_sum[r] += _mm512_reduce_add_ps(weight);
        }
      }

      for (int d = 0; d < HEAD_DIM; ++d) {
        __m512 value_acc[QUERY_BLOCK];
        for (int r = 0; r < QUERY_BLOCK; ++r) value_acc[r] = _mm512_setzero_ps();
        for (int tc = 0; tc < chunks; ++tc) {
          const int key = jb + tc * SIMD_WIDTH;
          const __m512 value = _mm512_loadu_ps(pv_base + (int64_t)d * Lkp + key);
#pragma GCC unroll 8
          for (int r = 0; r < rows; ++r) {
            value_acc[r] = _mm512_fmadd_ps(
                _mm512_load_ps(scores[r] + tc * SIMD_WIDTH), value, value_acc[r]);
          }
        }
        for (int r = 0; r < rows; ++r) {
          local_out[r][d] = _mm512_reduce_add_ps(value_acc[r]);
        }
      }

      for (int r = 0; r < rows; ++r) {
        const float new_max = local_max[r] > global_max[r] ? local_max[r] : global_max[r];
        const float old_scale = isinf(global_max[r]) ? 0.0f : expf(global_max[r] - new_max);
        const float local_scale = expf(local_max[r] - new_max);
        const __m512 old_vector = _mm512_set1_ps(old_scale);
        const __m512 local_vector = _mm512_set1_ps(local_scale);
        for (int dc = 0; dc < HEAD_DIM / SIMD_WIDTH; ++dc) {
          const __m512 local_value = _mm512_load_ps(local_out[r] + dc * SIMD_WIDTH);
          global_acc[r][dc] = _mm512_fmadd_ps(
              local_value, local_vector, _mm512_mul_ps(global_acc[r][dc], old_vector));
        }
        global_sum[r] = global_sum[r] * old_scale + local_sum[r] * local_scale;
        global_max[r] = new_max;
      }
    }

    for (int r = 0; r < rows; ++r) {
      float* destination = out_blhd + (((int64_t)b * Lq + i0 + r) * H + h) * HEAD_DIM;
      const __m512 inverse = _mm512_set1_ps(1.0f / global_sum[r]);
      for (int dc = 0; dc < HEAD_DIM / SIMD_WIDTH; ++dc) {
        _mm512_storeu_ps(
            destination + dc * SIMD_WIDTH,
            _mm512_mul_ps(global_acc[r][dc], inverse));
      }
    }
  }
  return 0;
}

int triposplat_sdpa_packed_output_is_blhd(void) {
  return 1;
}
