#pragma once

#include "simd_detect.h"
#include <cstdint>

#if defined(SIMD_ARCH_ARM64)

#ifdef SIMD_HAS_NEON
#include <arm_neon.h>
#endif

namespace uint64_compute {
namespace simd_arm {

// NEON 实现 (128位，2个uint64_t)
#ifdef SIMD_HAS_NEON
class NEONKernels {
public:
    static void add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    
    // ARM64特有的高级操作
    #ifdef SIMD_ARCH_ARM64
    static void fma_u64(const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t* result, size_t count);
    static void matmul_block_u64(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
    
    // ARM64的向量化reduction操作
    static uint64_t reduce_sum_u64(const uint64_t* data, size_t count);
    static uint64_t reduce_max_u64(const uint64_t* data, size_t count);
    static uint64_t reduce_min_u64(const uint64_t* data, size_t count);
    #endif
    
    // 向量化dot product
    static uint64_t dot_product_u64(const uint64_t* a, const uint64_t* b, size_t count);
    
    // 内存操作优化
    static void memcpy_aligned(void* dst, const void* src, size_t bytes);
    static void memset_u64(uint64_t* dst, uint64_t value, size_t count);
};
#endif

// SVE实现 (可变宽度SIMD)
#ifdef SIMD_HAS_SVE
class SVEKernels {
public:
    static void add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    
    // SVE特有功能
    static size_t get_vector_length(); // 获取SVE向量长度
    static void predicated_add(const uint64_t* a, const uint64_t* b, uint64_t* result, 
                              const bool* mask, size_t count);
};
#endif

// ARM优化工具
class ARMUtils {
public:
    // 内存对齐检查
    static bool is_aligned(const void* ptr, size_t alignment = 16);
    
    // 预取指令
    static void prefetch_read(const void* addr);
    static void prefetch_write(void* addr);
    
    // 缓存操作
    static void cache_flush(const void* addr, size_t size);
    static void cache_invalidate(const void* addr, size_t size);
    
    // ARM特有的优化
    static constexpr size_t CACHE_LINE_SIZE = 64; // ARM64典型值
    static size_t get_optimal_block_size(size_t data_size);
    
    // CPU特性检测
    static bool supports_crypto(); // 加密扩展
    static bool supports_fp16();   // 半精度浮点
    static bool supports_dotprod(); // 点积指令
    
    #ifdef SIMD_ARCH_ARM64
    // ARM64特有功能
    static uint64_t get_cycle_counter(); // 性能计数器
    static void memory_barrier();        // 内存屏障
    #endif
};

// 跨ARM架构的通用接口
class ARMDispatcher {
public:
    static void initialize();
    static bool is_neon_available();
    static bool is_sve_available();
    
    // 自动选择最佳实现
    static void add_u64_auto(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply_u64_auto(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
};

} // namespace simd_arm
} // namespace uint64_compute

#endif // SIMD_ARCH_ARM64