#include <immintrin.h>
#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static inline __mmask16 tail_mask(int64_t count) {
  return count >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << count) - 1u);
}

static inline void thread_bounds(int64_t count, int thread_id, int thread_count,
                                 int64_t *begin, int64_t *end) {
  *begin = (count * thread_id) / thread_count;
  *end = (count * (thread_id + 1)) / thread_count;
}

int triposplat_cfg_combine_inplace_f32_avx512(
    float *positive, const float *negative, const float *guidance,
    int batch, int64_t elements_per_batch, int threads) {
  if (!positive || !negative || !guidance) return -1;
  if (batch <= 0 || elements_per_batch <= 0) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
#else
  {
    const int thread_id = 0;
    const int thread_count = 1;
#endif
    for (int b = 0; b < batch; ++b) {
      int64_t begin;
      int64_t end;
      thread_bounds(elements_per_batch, thread_id, thread_count, &begin, &end);
      float *dst = positive + (int64_t)b * elements_per_batch;
      const float *neg = negative + (int64_t)b * elements_per_batch;
      const __m512 g = _mm512_set1_ps(guidance[b]);
      const __m512 gm1 = _mm512_sub_ps(g, _mm512_set1_ps(1.0f));
      for (int64_t i = begin; i < end; i += 16) {
        const __mmask16 mask = tail_mask(end - i);
        const __m512 pos_value = _mm512_maskz_loadu_ps(mask, dst + i);
        const __m512 neg_value = _mm512_maskz_loadu_ps(mask, neg + i);
        const __m512 pos_scaled = _mm512_mul_ps(g, pos_value);
        const __m512 neg_scaled = _mm512_mul_ps(gm1, neg_value);
        _mm512_mask_storeu_ps(dst + i, mask, _mm512_sub_ps(pos_scaled, neg_scaled));
      }
    }
  }
  return 0;
}

int triposplat_euler_update_inplace_f32_avx512(
    float *sample, const float *prediction, const float *dt,
    int batch, int64_t elements_per_batch, int threads) {
  if (!sample || !prediction || !dt) return -1;
  if (batch <= 0 || elements_per_batch <= 0) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
#else
  {
    const int thread_id = 0;
    const int thread_count = 1;
#endif
    for (int b = 0; b < batch; ++b) {
      int64_t begin;
      int64_t end;
      thread_bounds(elements_per_batch, thread_id, thread_count, &begin, &end);
      float *dst = sample + (int64_t)b * elements_per_batch;
      const float *pred = prediction + (int64_t)b * elements_per_batch;
      const __m512 step = _mm512_set1_ps(dt[b]);
      for (int64_t i = begin; i < end; i += 16) {
        const __mmask16 mask = tail_mask(end - i);
        const __m512 x = _mm512_maskz_loadu_ps(mask, dst + i);
        const __m512 v = _mm512_maskz_loadu_ps(mask, pred + i);
        const __m512 delta = _mm512_mul_ps(step, v);
        _mm512_mask_storeu_ps(dst + i, mask, _mm512_sub_ps(x, delta));
      }
    }
  }
  return 0;
}

int triposplat_ab2_update_inplace_f32_avx512(
    float *sample, const float *prediction, const float *previous_prediction,
    const float *dt, const float *previous_dt,
    int batch, int64_t elements_per_batch, int threads) {
  if (!sample || !prediction || !previous_prediction || !dt || !previous_dt) return -1;
  if (batch <= 0 || elements_per_batch <= 0) return -2;
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#pragma omp parallel
  {
    const int thread_id = omp_get_thread_num();
    const int thread_count = omp_get_num_threads();
#else
  {
    const int thread_id = 0;
    const int thread_count = 1;
#endif
    for (int b = 0; b < batch; ++b) {
      int64_t begin;
      int64_t end;
      thread_bounds(elements_per_batch, thread_id, thread_count, &begin, &end);
      float *dst = sample + (int64_t)b * elements_per_batch;
      const float *pred = prediction + (int64_t)b * elements_per_batch;
      const float *prev = previous_prediction + (int64_t)b * elements_per_batch;
      const float denominator = fmaxf(previous_dt[b], 1.0e-12f);
      const float half_ratio_scalar = 0.5f * (dt[b] / denominator);
      const __m512 half_ratio = _mm512_set1_ps(half_ratio_scalar);
      const __m512 current_scale = _mm512_add_ps(_mm512_set1_ps(1.0f), half_ratio);
      const __m512 step = _mm512_set1_ps(dt[b]);
      for (int64_t i = begin; i < end; i += 16) {
        const __mmask16 mask = tail_mask(end - i);
        const __m512 x = _mm512_maskz_loadu_ps(mask, dst + i);
        const __m512 current = _mm512_maskz_loadu_ps(mask, pred + i);
        const __m512 previous = _mm512_maskz_loadu_ps(mask, prev + i);
        const __m512 current_term = _mm512_mul_ps(current_scale, current);
        const __m512 previous_term = _mm512_mul_ps(half_ratio, previous);
        const __m512 effective = _mm512_sub_ps(current_term, previous_term);
        const __m512 delta = _mm512_mul_ps(step, effective);
        _mm512_mask_storeu_ps(dst + i, mask, _mm512_sub_ps(x, delta));
      }
    }
  }
  return 0;
}
