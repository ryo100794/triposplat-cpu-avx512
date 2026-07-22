#include <stdint.h>

#define triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2 \
  triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2_exact
#define triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2 \
  triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2_exact
#include "native_qkv_postprocess_v2_avx512.c"
#undef triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2
#undef triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2

static inline float bf16_roundtrip(float value) {
  union {
    float f;
    uint32_t u;
  } bits = {.f = value};
  const uint32_t exponent = bits.u & 0x7f800000u;
  if (exponent != 0x7f800000u) {
    bits.u += 0x00007fffu + ((bits.u >> 16) & 1u);
  }
  bits.u &= 0xffff0000u;
  return bits.f;
}

static void round_packed(float* values, int64_t count, int threads) {
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
#pragma omp parallel for schedule(static)
  for (int64_t index = 0; index < count; ++index) {
    values[index] = bf16_roundtrip(values[index]);
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
  const int status = triposplat_qkv_rope_rmsnorm_pack_f32_avx512_v2_exact(
      qkv, frequencies, q_gamma, k_gamma, q_bhld, packed_k, packed_v,
      B, L, H, D, L_padded, frequency_batches, threads);
  if (status != 0) return status;
  const int64_t packed_count = (int64_t)B * H * D * L_padded;
#ifdef TRIPOSPLAT_BF16_ROUND_K
  round_packed(packed_k, packed_count, threads);
#endif
#ifdef TRIPOSPLAT_BF16_ROUND_V
  round_packed(packed_v, packed_count, threads);
#endif
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
  const int status = triposplat_q_kv_selected_rope_rmsnorm_pack_f32_avx512_v2_exact(
      q, kv, frequencies, selected_indices, q_gamma, k_gamma,
      q_bhld, packed_k, packed_v,
      B, Lq, Lk, H, D, Lk_padded, frequency_batches, threads);
  if (status != 0) return status;
  const int64_t packed_count = (int64_t)B * H * D * Lk_padded;
#ifdef TRIPOSPLAT_BF16_ROUND_K
  round_packed(packed_k, packed_count, threads);
#endif
#ifdef TRIPOSPLAT_BF16_ROUND_V
  round_packed(packed_v, packed_count, threads);
#endif
  return 0;
}

int triposplat_qkv_postprocess_bf16_probe_mode(void) {
  int mode = 0;
#ifdef TRIPOSPLAT_BF16_ROUND_K
  mode |= 1;
#endif
#ifdef TRIPOSPLAT_BF16_ROUND_V
  mode |= 2;
#endif
  return mode;
}
