#include <immintrin.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline __mmask16 tail_mask(int n) { return n >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << n) - 1u); }

int triposplat_modulate_inplace_f32_avx512(float* h, const float* scale, const float* shift,
    int batch, int rows, int cols, int threads) {
  if (!h || !scale || !shift) return -1; if (batch<=0 || rows<=0 || cols<=0) return -2;
#ifdef _OPENMP
  if (threads>0) omp_set_num_threads(threads);
#endif
  const int64_t total=(int64_t)batch*rows;
#pragma omp parallel for schedule(static)
  for (int64_t r=0;r<total;++r) {
    const int b=(int)(r/rows); float* dst=h+r*cols; const float* s=scale+(int64_t)b*cols; const float* t=shift+(int64_t)b*cols;
    for(int c=0;c<cols;c+=16){const __mmask16 m=tail_mask(cols-c); const __m512 one_s=_mm512_add_ps(_mm512_set1_ps(1.0f),_mm512_maskz_loadu_ps(m,s+c)); __m512 y=_mm512_mul_ps(_mm512_maskz_loadu_ps(m,dst+c),one_s); y=_mm512_add_ps(y,_mm512_maskz_loadu_ps(m,t+c)); _mm512_mask_storeu_ps(dst+c,m,y);}
  }
  return 0;
}

int triposplat_gated_residual_inplace_f32_avx512(float* base, float* delta, const float* gate,
    int batch, int rows, int cols, int threads) {
  if (!base || !delta || !gate) return -1; if (batch<=0 || rows<=0 || cols<=0) return -2;
#ifdef _OPENMP
  if (threads>0) omp_set_num_threads(threads);
#endif
  const int64_t total=(int64_t)batch*rows;
#pragma omp parallel for schedule(static)
  for(int64_t r=0;r<total;++r){const int b=(int)(r/rows); float* x=base+r*cols; float* h=delta+r*cols; const float* g=gate+(int64_t)b*cols;
    for(int c=0;c<cols;c+=16){const __mmask16 m=tail_mask(cols-c); __m512 hv=_mm512_mul_ps(_mm512_maskz_loadu_ps(m,h+c),_mm512_maskz_loadu_ps(m,g+c)); _mm512_mask_storeu_ps(h+c,m,hv); __m512 xv=_mm512_add_ps(_mm512_maskz_loadu_ps(m,x+c),hv); _mm512_mask_storeu_ps(x+c,m,xv);}}
  return 0;
}

int triposplat_add_inplace_f32_avx512(float* base, const float* delta, int64_t count, int threads) {
  if(!base || !delta) return -1; if(count<=0) return -2;
#ifdef _OPENMP
  if(threads>0) omp_set_num_threads(threads);
#endif
  const int64_t chunks=(count+15)/16;
#pragma omp parallel for schedule(static)
  for(int64_t k=0;k<chunks;++k){const int64_t o=k*16; const int rem=(int)(count-o); const __mmask16 m=tail_mask(rem); _mm512_mask_storeu_ps(base+o,m,_mm512_add_ps(_mm512_maskz_loadu_ps(m,base+o),_mm512_maskz_loadu_ps(m,delta+o)));}
  return 0;
}
