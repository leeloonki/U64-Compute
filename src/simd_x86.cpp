#include "simd_x86.h"

#ifdef SIMD_ARCH_X86_64
#include <algorithm>

namespace uint64_compute {
namespace simd_x86 {

// SSE2 实现
void SSE2Kernels::add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~1ULL; // 处理2的倍数
    
    for (size_t i = 0; i < simd_count; i += 2) {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m128i vr = _mm_add_epi64(va, vb);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SSE2Kernels::multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~1ULL;
    
    for (size_t i = 0; i < simd_count; i += 2) {
        // SSE2没有64位乘法，需要手动实现
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        
        // 分解为32位进行乘法
        __m128i va_low = _mm_and_si128(va, _mm_set1_epi64x(0xFFFFFFFF));
        __m128i va_high = _mm_srli_epi64(va, 32);
        __m128i vb_low = _mm_and_si128(vb, _mm_set1_epi64x(0xFFFFFFFF));
        __m128i vb_high = _mm_srli_epi64(vb, 32);
        
        // 计算部分乘积 (只保留低64位)
        __m128i mul_ll = _mm_mul_epu32(va_low, vb_low);
        __m128i mul_lh = _mm_mul_epu32(va_low, vb_high);
        __m128i mul_hl = _mm_mul_epu32(va_high, vb_low);
        
        // 组合结果
        __m128i result_high = _mm_add_epi64(_mm_slli_epi64(mul_lh, 32), _mm_slli_epi64(mul_hl, 32));
        __m128i vr = _mm_add_epi64(mul_ll, result_high);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void SSE2Kernels::add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~1ULL;
    __m128i vs = _mm_set1_epi64x(scalar);
    
    for (size_t i = 0; i < simd_count; i += 2) {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vr = _mm_add_epi64(va, vs);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void SSE2Kernels::multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~1ULL;
    __m128i vs = _mm_set1_epi64x(scalar);
    
    for (size_t i = 0; i < simd_count; i += 2) {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        
        // 使用与multiply_u64相同的方法
        __m128i va_low = _mm_and_si128(va, _mm_set1_epi64x(0xFFFFFFFF));
        __m128i va_high = _mm_srli_epi64(va, 32);
        __m128i vs_low = _mm_and_si128(vs, _mm_set1_epi64x(0xFFFFFFFF));
        __m128i vs_high = _mm_srli_epi64(vs, 32);
        
        __m128i mul_ll = _mm_mul_epu32(va_low, vs_low);
        __m128i mul_lh = _mm_mul_epu32(va_low, vs_high);
        __m128i mul_hl = _mm_mul_epu32(va_high, vs_low);
        
        __m128i result_high = _mm_add_epi64(_mm_slli_epi64(mul_lh, 32), _mm_slli_epi64(mul_hl, 32));
        __m128i vr = _mm_add_epi64(mul_ll, result_high);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

#ifdef SIMD_HAS_AVX2
// AVX2 实现
void AVX2Kernels::add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~3ULL; // 处理4的倍数
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i vr = _mm256_add_epi64(va, vb);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void AVX2Kernels::multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~3ULL;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        
        // AVX2的64位乘法
        __m256i va_low = _mm256_and_si256(va, _mm256_set1_epi64x(0xFFFFFFFF));
        __m256i va_high = _mm256_srli_epi64(va, 32);
        __m256i vb_low = _mm256_and_si256(vb, _mm256_set1_epi64x(0xFFFFFFFF));
        __m256i vb_high = _mm256_srli_epi64(vb, 32);
        
        __m256i mul_ll = _mm256_mul_epu32(va_low, vb_low);
        __m256i mul_lh = _mm256_mul_epu32(va_low, vb_high);
        __m256i mul_hl = _mm256_mul_epu32(va_high, vb_low);
        
        __m256i result_high = _mm256_add_epi64(_mm256_slli_epi64(mul_lh, 32), _mm256_slli_epi64(mul_hl, 32));
        __m256i vr = _mm256_add_epi64(mul_ll, result_high);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void AVX2Kernels::add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~3ULL;
    __m256i vs = _mm256_set1_epi64x(scalar);
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vr = _mm256_add_epi64(va, vs);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void AVX2Kernels::multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~3ULL;
    __m256i vs = _mm256_set1_epi64x(scalar);
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        
        __m256i va_low = _mm256_and_si256(va, _mm256_set1_epi64x(0xFFFFFFFF));
        __m256i va_high = _mm256_srli_epi64(va, 32);
        __m256i vs_low = _mm256_and_si256(vs, _mm256_set1_epi64x(0xFFFFFFFF));
        __m256i vs_high = _mm256_srli_epi64(vs, 32);
        
        __m256i mul_ll = _mm256_mul_epu32(va_low, vs_low);
        __m256i mul_lh = _mm256_mul_epu32(va_low, vs_high);
        __m256i mul_hl = _mm256_mul_epu32(va_high, vs_low);
        
        __m256i result_high = _mm256_add_epi64(_mm256_slli_epi64(mul_lh, 32), _mm256_slli_epi64(mul_hl, 32));
        __m256i vr = _mm256_add_epi64(mul_ll, result_high);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

void AVX2Kernels::matmul_block_u64(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                  size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    // 分块矩阵乘法，利用AVX2进行优化
    constexpr size_t BLOCK_SIZE = 64; // 缓存友好的块大小
    
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);
                
                // 内层循环使用AVX2优化
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        __m256i a_val = _mm256_set1_epi64x(A[ii * lda + kk]);
                        
                        size_t jj = j;
                        // 向量化处理4个元素
                        for (; jj + 3 < j_end; jj += 4) {
                            __m256i b_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[kk * ldb + jj]));
                            __m256i c_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&C[ii * ldc + jj]));
                            
                            // 64位乘法
                            __m256i a_low = _mm256_and_si256(a_val, _mm256_set1_epi64x(0xFFFFFFFF));
                            __m256i a_high = _mm256_srli_epi64(a_val, 32);
                            __m256i b_low = _mm256_and_si256(b_vals, _mm256_set1_epi64x(0xFFFFFFFF));
                            __m256i b_high = _mm256_srli_epi64(b_vals, 32);
                            
                            __m256i mul_ll = _mm256_mul_epu32(a_low, b_low);
                            __m256i mul_lh = _mm256_mul_epu32(a_low, b_high);
                            __m256i mul_hl = _mm256_mul_epu32(a_high, b_low);
                            
                            __m256i result_high = _mm256_add_epi64(_mm256_slli_epi64(mul_lh, 32), _mm256_slli_epi64(mul_hl, 32));
                            __m256i product = _mm256_add_epi64(mul_ll, result_high);
                            
                            c_vals = _mm256_add_epi64(c_vals, product);
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[ii * ldc + jj]), c_vals);
                        }
                        
                        // 处理剩余元素
                        for (; jj < j_end; ++jj) {
                            C[ii * ldc + jj] += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                    }
                }
            }
        }
    }
}
#endif

#ifdef SIMD_HAS_AVX512F
// AVX512 实现
void AVX512Kernels::add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~7ULL; // 处理8的倍数
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vr = _mm512_add_epi64(va, vb);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void AVX512Kernels::multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~7ULL;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        
        // AVX512有原生64位乘法支持（需要DQ扩展）
        #ifdef __AVX512DQ__
        __m512i vr = _mm512_mullo_epi64(va, vb);
        #else
        // 没有DQ扩展，手动实现
        __m512i va_low = _mm512_and_si512(va, _mm512_set1_epi64(0xFFFFFFFF));
        __m512i va_high = _mm512_srli_epi64(va, 32);
        __m512i vb_low = _mm512_and_si512(vb, _mm512_set1_epi64(0xFFFFFFFF));
        __m512i vb_high = _mm512_srli_epi64(vb, 32);
        
        __m512i mul_ll = _mm512_mul_epu32(va_low, vb_low);
        __m512i mul_lh = _mm512_mul_epu32(va_low, vb_high);
        __m512i mul_hl = _mm512_mul_epu32(va_high, vb_low);
        
        __m512i result_high = _mm512_add_epi64(_mm512_slli_epi64(mul_lh, 32), _mm512_slli_epi64(mul_hl, 32));
        __m512i vr = _mm512_add_epi64(mul_ll, result_high);
        #endif
        
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void AVX512Kernels::add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~7ULL;
    __m512i vs = _mm512_set1_epi64(scalar);
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vr = _mm512_add_epi64(va, vs);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void AVX512Kernels::multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~7ULL;
    __m512i vs = _mm512_set1_epi64(scalar);
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        
        #ifdef __AVX512DQ__
        __m512i vr = _mm512_mullo_epi64(va, vs);
        #else
        __m512i va_low = _mm512_and_si512(va, _mm512_set1_epi64(0xFFFFFFFF));
        __m512i va_high = _mm512_srli_epi64(va, 32);
        __m512i vs_low = _mm512_and_si512(vs, _mm512_set1_epi64(0xFFFFFFFF));
        __m512i vs_high = _mm512_srli_epi64(vs, 32);
        
        __m512i mul_ll = _mm512_mul_epu32(va_low, vs_low);
        __m512i mul_lh = _mm512_mul_epu32(va_low, vs_high);
        __m512i mul_hl = _mm512_mul_epu32(va_high, vs_low);
        
        __m512i result_high = _mm512_add_epi64(_mm512_slli_epi64(mul_lh, 32), _mm512_slli_epi64(mul_hl, 32));
        __m512i vr = _mm512_add_epi64(mul_ll, result_high);
        #endif
        
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

void AVX512Kernels::fma_u64(const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t* result, size_t count) {
    const size_t simd_count = count & ~7ULL;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        __m512i vc = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(c + i));
        
        // 计算 a * b + c
        #ifdef __AVX512DQ__
        __m512i product = _mm512_mullo_epi64(va, vb);
        #else
        // 手动64位乘法
        __m512i va_low = _mm512_and_si512(va, _mm512_set1_epi64(0xFFFFFFFF));
        __m512i va_high = _mm512_srli_epi64(va, 32);
        __m512i vb_low = _mm512_and_si512(vb, _mm512_set1_epi64(0xFFFFFFFF));
        __m512i vb_high = _mm512_srli_epi64(vb, 32);
        
        __m512i mul_ll = _mm512_mul_epu32(va_low, vb_low);
        __m512i mul_lh = _mm512_mul_epu32(va_low, vb_high);
        __m512i mul_hl = _mm512_mul_epu32(va_high, vb_low);
        
        __m512i result_high = _mm512_add_epi64(_mm512_slli_epi64(mul_lh, 32), _mm512_slli_epi64(mul_hl, 32));
        __m512i product = _mm512_add_epi64(mul_ll, result_high);
        #endif
        
        __m512i vr = _mm512_add_epi64(product, vc);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result + i), vr);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

void AVX512Kernels::matmul_block_u64(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                     size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    constexpr size_t BLOCK_SIZE = 64;
    
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);
                
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        __m512i a_val = _mm512_set1_epi64(A[ii * lda + kk]);
                        
                        size_t jj = j;
                        // 向量化处理8个元素
                        for (; jj + 7 < j_end; jj += 8) {
                            __m512i b_vals = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&B[kk * ldb + jj]));
                            __m512i c_vals = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&C[ii * ldc + jj]));
                            
                            #ifdef __AVX512DQ__
                            __m512i product = _mm512_mullo_epi64(a_val, b_vals);
                            #else
                            // 手动64位乘法
                            __m512i a_low = _mm512_and_si512(a_val, _mm512_set1_epi64(0xFFFFFFFF));
                            __m512i a_high = _mm512_srli_epi64(a_val, 32);
                            __m512i b_low = _mm512_and_si512(b_vals, _mm512_set1_epi64(0xFFFFFFFF));
                            __m512i b_high = _mm512_srli_epi64(b_vals, 32);
                            
                            __m512i mul_ll = _mm512_mul_epu32(a_low, b_low);
                            __m512i mul_lh = _mm512_mul_epu32(a_low, b_high);
                            __m512i mul_hl = _mm512_mul_epu32(a_high, b_low);
                            
                            __m512i result_high = _mm512_add_epi64(_mm512_slli_epi64(mul_lh, 32), _mm512_slli_epi64(mul_hl, 32));
                            __m512i product = _mm512_add_epi64(mul_ll, result_high);
                            #endif
                            
                            c_vals = _mm512_add_epi64(c_vals, product);
                            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&C[ii * ldc + jj]), c_vals);
                        }
                        
                        // 处理剩余元素
                        for (; jj < j_end; ++jj) {
                            C[ii * ldc + jj] += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                    }
                }
            }
        }
    }
}
#endif

// x86优化工具
bool X86Utils::is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

void X86Utils::prefetch_read(const void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
}

void X86Utils::prefetch_write(void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
}

size_t X86Utils::get_optimal_block_size(size_t data_size) {
    // 基于L1缓存大小的启发式算法
    constexpr size_t L1_CACHE_SIZE = 32768; // 32KB
    constexpr size_t ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(uint64_t);
    
    size_t block_size = std::min(data_size, L1_CACHE_SIZE / sizeof(uint64_t));
    
    // 对齐到缓存行边界
    return (block_size / ELEMENTS_PER_CACHE_LINE) * ELEMENTS_PER_CACHE_LINE;
}

} // namespace simd_x86
} // namespace uint64_compute

#endif // SIMD_ARCH_X86_64