#include <immintrin.h>
#include <string.h>
#include <stdlib.h>

static inline int Max(int a, int b) {
    return a < b ? b: a;
}

static inline int Min(int a, int b) {
    return a > b ? b: a;
}

static inline float _mm256_reduce_add_ps(__m256 x) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static inline void fma_storeu(float * mem_addr, __m256 a, __m256 b) {
    _mm256_storeu_ps(mem_addr, _mm256_fmadd_ps(a, b, _mm256_loadu_ps(mem_addr)));
}

static inline void add_storeu(float * mem_addr, __m256 a) {
    _mm256_storeu_ps(mem_addr, a + _mm256_loadu_ps(mem_addr));
}


#ifdef STANDALONE
void convolution(const float *__restrict__ image,
                 const float *__restrict__ weight,
                 const float *__restrict__ bias,
                 float *__restrict__ out);
int main() {
    // sub for autotuning timing
    float* image = aligned_alloc(32, IMAGE_LEN * sizeof(float));
    float* weight = aligned_alloc(32, WEIGHT_LEN * sizeof(float));
    float* bias = aligned_alloc(32, BIAS_LEN * sizeof(float));
    float* out = aligned_alloc(32, OUT_LEN * sizeof(float));
    for(int t=0; t<TIMES; ++t) {
        asm volatile("" ::: "memory");
        convolution(image, weight, bias, out);
    }
    return 0;
}
#endif