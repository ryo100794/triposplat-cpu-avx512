#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
  HEAD_DIM = 64,
  QUERY_BLOCK = 8,
  KEY_TILE = 512,
  SIMD_WIDTH = 16
};

static inline __m512 exp512_ps(__m512 x) {
  const __m512 exp_hi = _mm512_set1_ps(88.3762626647949f);
  const __m512 exp_lo = _mm512_set1_ps(-88.3762626647949f);
  const __m512 log2ef = _mm512_set1_ps(1.44269504088896341f);
  const __m512 half = _mm512_set1_ps(0.5f);
  const __m512 one = _mm512_set1_ps(1.0f);

  x = _mm512_min_ps(x, exp_hi);
  x = _mm512_max_ps(x, exp_lo);
  __m512 fx = _mm512_fmadd_ps(x, log2ef, half);
  fx = _mm512_floor_ps(fx);

  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
  x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
  const __m512 z = _mm512_mul_ps(x, x);

  __m512 y = _mm512_set1_ps(1.9875691500e-4f);
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
  y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
  y = _mm512_fmadd_ps(y, z, x);
  y = _mm512_add_ps(y, one);

  __m512i emm = _mm512_cvttps_epi32(fx);
  emm = _mm512_add_epi32(emm, _mm512_set1_epi32(0x7f));
  emm = _mm512_slli_epi32(emm, 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(emm));
}

static inline int round_up_16(int value) {
  return (value + 15) & ~15;
}

int triposplat_sdpa_f32_avx512_exact_q8t512(
    const float* restrict q,
    const float* restrict k,
    const float* restrict v,
    const float* restrict key_bias_or_null,
    float* restrict out,
    int B,
    int H,
    int Lq,
    int Lk,
    int D,
    int has_key_bias,
    int key_bias_len,
    int threads) {
  if (q == NULL || k == NULL || v == NULL || out == NULL) return -1;
  if (B <= 0 || H <= 0 || Lq <= 0 || Lk <= 0 || D != HEAD_DIM) return -2;
  if (has_key_bias && (key_bias_or_null == NULL || key_bias_len != Lk)) return -3;

  const int Lkp = round_up_16(Lk);
  const int64_t bh_count = (int64_t)B * H;
  const int64_t packed_count = bh_count * HEAD_DIM * (int64_t)Lkp;
  float* packed = NULL;
  if (posix_memalign((void**)&packed, 64, (size_t)(2 * packed_count) * sizeof(float)) != 0) return -5;
  float* packed_k = packed;
  float* packed_v = packed + packed_count;

#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  const float scale = 0.125f;
  const int64_t q_blocks = (Lq + QUERY_BLOCK - 1) / QUERY_BLOCK;
  const int64_t total_q_blocks = bh_count * q_blocks;

#pragma omp parallel
  {
#pragma omp for collapse(2) schedule(static)
    for (int64_t bh = 0; bh < bh_count; ++bh) {
      for (int d = 0; d < HEAD_DIM; ++d) {
        float* pk = packed_k + (bh * HEAD_DIM + d) * (int64_t)Lkp;
        float* pv = packed_v + (bh * HEAD_DIM + d) * (int64_t)Lkp;
        const float* src_k = k + bh * (int64_t)Lk * HEAD_DIM;
        const float* src_v = v + bh * (int64_t)Lk * HEAD_DIM;
        int j = 0;
        for (; j < Lk; ++j) {
          pk[j] = src_k[(int64_t)j * HEAD_DIM + d];
          pv[j] = src_v[(int64_t)j * HEAD_DIM + d];
        }
        for (; j < Lkp; ++j) {
          pk[j] = 0.0f;
          pv[j] = 0.0f;
        }
      }
    }

#pragma omp for schedule(static)
    for (int64_t block_index = 0; block_index < total_q_blocks; ++block_index) {
      const int64_t qb = block_index % q_blocks;
      const int64_t bh = block_index / q_blocks;
      const int i0 = (int)(qb * QUERY_BLOCK);
      const int rows = (i0 + QUERY_BLOCK <= Lq) ? QUERY_BLOCK : (Lq - i0);
      const float* q_base = q + bh * (int64_t)Lq * HEAD_DIM;
      float* out_base = out + bh * (int64_t)Lq * HEAD_DIM;
      const float* pk_base = packed_k + bh * HEAD_DIM * (int64_t)Lkp;
      const float* pv_base = packed_v + bh * HEAD_DIM * (int64_t)Lkp;

      float qrow[QUERY_BLOCK][HEAD_DIM] __attribute__((aligned(64)));
      for (int r = 0; r < rows; ++r) {
        const float* src = q_base + (int64_t)(i0 + r) * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; ++d) qrow[r][d] = src[d];
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
        const int count = (jb + KEY_TILE <= Lk) ? KEY_TILE : (Lk - jb);
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
            const __m512 kv = _mm512_loadu_ps(pk_base + (int64_t)d * Lkp + key);
#pragma GCC unroll 8
            for (int r = 0; r < rows; ++r) {
              score_vec[r] = _mm512_fmadd_ps(_mm512_set1_ps(qrow[r][d]), kv, score_vec[r]);
            }
          }
          const int valid = Lk - key < SIMD_WIDTH ? Lk - key : SIMD_WIDTH;
          const __mmask16 valid_mask = (__mmask16)((1u << valid) - 1u);
          __m512 bias = _mm512_setzero_ps();
          if (has_key_bias) bias = _mm512_maskz_loadu_ps(valid_mask, key_bias_or_null + key);
#pragma GCC unroll 8
          for (int r = 0; r < rows; ++r) {
            __m512 s = _mm512_mul_ps(score_vec[r], _mm512_set1_ps(scale));
            if (has_key_bias) s = _mm512_add_ps(s, bias);
            if (valid < SIMD_WIDTH) s = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), valid_mask, s);
            _mm512_store_ps(scores[r] + tc * SIMD_WIDTH, s);
            const float m = _mm512_reduce_max_ps(s);
            if (m > local_max[r]) local_max[r] = m;
          }
        }

        for (int r = 0; r < rows; ++r) {
          const __m512 max_vec = _mm512_set1_ps(local_max[r]);
          for (int tc = 0; tc < chunks; ++tc) {
            const int key = jb + tc * SIMD_WIDTH;
            const int valid = Lk - key < SIMD_WIDTH ? Lk - key : SIMD_WIDTH;
            const __mmask16 valid_mask = (__mmask16)((1u << valid) - 1u);
            __m512 w = exp512_ps(_mm512_sub_ps(_mm512_load_ps(scores[r] + tc * SIMD_WIDTH), max_vec));
            if (valid < SIMD_WIDTH) w = _mm512_maskz_mov_ps(valid_mask, w);
            _mm512_store_ps(scores[r] + tc * SIMD_WIDTH, w);
            local_sum[r] += _mm512_reduce_add_ps(w);
          }
        }

        for (int d = 0; d < HEAD_DIM; ++d) {
          __m512 value_acc[QUERY_BLOCK];
          for (int r = 0; r < QUERY_BLOCK; ++r) value_acc[r] = _mm512_setzero_ps();
          for (int tc = 0; tc < chunks; ++tc) {
            const int key = jb + tc * SIMD_WIDTH;
            const __m512 vv = _mm512_loadu_ps(pv_base + (int64_t)d * Lkp + key);
#pragma GCC unroll 8
            for (int r = 0; r < rows; ++r) {
              const __m512 weight = _mm512_load_ps(scores[r] + tc * SIMD_WIDTH);
              value_acc[r] = _mm512_fmadd_ps(weight, vv, value_acc[r]);
            }
          }
          for (int r = 0; r < rows; ++r) local_out[r][d] = _mm512_reduce_add_ps(value_acc[r]);
        }

        for (int r = 0; r < rows; ++r) {
          const float new_max = local_max[r] > global_max[r] ? local_max[r] : global_max[r];
          const float old_scale = isinf(global_max[r]) ? 0.0f : expf(global_max[r] - new_max);
          const float local_scale = expf(local_max[r] - new_max);
          const __m512 oldv = _mm512_set1_ps(old_scale);
          const __m512 localv = _mm512_set1_ps(local_scale);
          for (int dc = 0; dc < HEAD_DIM / SIMD_WIDTH; ++dc) {
            const __m512 loc = _mm512_load_ps(local_out[r] + dc * SIMD_WIDTH);
            global_acc[r][dc] = _mm512_fmadd_ps(loc, localv, _mm512_mul_ps(global_acc[r][dc], oldv));
          }
          global_sum[r] = global_sum[r] * old_scale + local_sum[r] * local_scale;
          global_max[r] = new_max;
        }
      }

      for (int r = 0; r < rows; ++r) {
        float* dst = out_base + (int64_t)(i0 + r) * HEAD_DIM;
        const __m512 inv = _mm512_set1_ps(1.0f / global_sum[r]);
        for (int dc = 0; dc < HEAD_DIM / SIMD_WIDTH; ++dc) {
          _mm512_storeu_ps(dst + dc * SIMD_WIDTH, _mm512_mul_ps(global_acc[r][dc], inv));
        }
      }
    }
  }

  free(packed);
  return 0;
}

int triposplat_sdpa_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* key_bias_or_null,
    float* out,
    int B,
    int H,
    int Lq,
    int Lk,
    int D,
    int has_key_bias,
    int key_bias_len,
    int threads) {
  return triposplat_sdpa_f32_avx512_exact_q8t512(
      q, k, v, key_bias_or_null, out, B, H, Lq, Lk, D,
      has_key_bias, key_bias_len, threads);
}
