#pragma once

#include "simd_detect.h"
#include <cstdint>

#ifdef SIMD_ARCH_X86_64
#include <immintrin.h>

namespace uint64_compute {
namespace simd_x86 {

// SSE2 实现 (128位，2个uint64_t)
class SSE2Kernels {
public:
    static void add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
};

// AVX2 实现 (256位，4个uint64_t)
#ifdef SIMD_HAS_AVX2
class AVX2Kernels {
public:
    static void add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    
    // 矩阵运算优化
    static void matmul_block_u64(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
};
#endif

// AVX-512 实现 (512位，8个uint64_t)
#ifdef SIMD_HAS_AVX512F
class AVX512Kernels {
public:
    static void add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    
    // 高级操作
    static void fma_u64(const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t* result, size_t count);
    static void matmul_block_u64(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
};
#endif

// x86通用优化工具
class X86Utils {
public:
    // 内存对齐检查
    static bool is_aligned(const void* ptr, size_t alignment = 32);
    
    // 预取指令
    static void prefetch_read(const void* addr);
    static void prefetch_write(void* addr);
    
    // 缓存行大小优化
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static size_t get_optimal_block_size(size_t data_size);
};

} // namespace simd_x86
} // namespace uint64_compute

#endif // SIMD_ARCH_X86_64