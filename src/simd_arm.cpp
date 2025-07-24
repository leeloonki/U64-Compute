#include "simd_arm.h"

// Only compile ARM code on ARM platforms
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <algorithm>
#include <cstring>

// ARM64 NEON 优化实现 - 针对64位整数运算优化

namespace uint64_compute {
namespace simd_arm {

#ifdef SIMD_HAS_NEON

// 64位整数乘法辅助函数 (NEON没有直接的64位乘法指令)
inline uint64x2_t vmulq_u64_emulated(uint64x2_t a, uint64x2_t b) {
    // 将64位数分解为高32位和低32位
    uint32x2_t a_lo = vmovn_u64(a);                    // 低32位
    uint32x2_t a_hi = vshrn_n_u64(a, 32);              // 高32位
    uint32x2_t b_lo = vmovn_u64(b);                    // 低32位
    uint32x2_t b_hi = vshrn_n_u64(b, 32);              // 高32位
    
    // 计算 (a_hi * b_lo + a_lo * b_hi) << 32 + a_lo * b_lo
    uint64x2_t lo_lo = vmull_u32(a_lo, b_lo);          // a_lo * b_lo
    uint64x2_t hi_lo = vmull_u32(a_hi, b_lo);          // a_hi * b_lo
    uint64x2_t lo_hi = vmull_u32(a_lo, b_hi);          // a_lo * b_hi
    
    // (a_hi * b_lo + a_lo * b_hi) << 32
    uint64x2_t mid = vaddq_u64(hi_lo, lo_hi);
    uint64x2_t mid_shifted = vshlq_n_u64(mid, 32);
    
    // 最终结果
    return vaddq_u64(lo_lo, mid_shifted);
}

// NEON优化的加法实现
void NEONKernels::add_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    size_t simd_count = (count / 2) * 2;  // 处理2的倍数
    
    // NEON向量化处理
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vr = vaddq_u64(va, vb);
        vst1q_u64(result + i, vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

// NEON优化的乘法实现
void NEONKernels::multiply_u64(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    size_t simd_count = (count / 2) * 2;  // 处理2的倍数
    
    // NEON向量化处理
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vr = vmulq_u64_emulated(va, vb);
        vst1q_u64(result + i, vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

// NEON优化的标量加法
void NEONKernels::add_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    size_t simd_count = (count / 2) * 2;  // 处理2的倍数
    uint64x2_t vs = vdupq_n_u64(scalar);  // 广播标量到向量
    
    // NEON向量化处理
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vr = vaddq_u64(va, vs);
        vst1q_u64(result + i, vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

// NEON优化的标量乘法
void NEONKernels::multiply_scalar_u64(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    size_t simd_count = (count / 2) * 2;  // 处理2的倍数
    uint64x2_t vs = vdupq_n_u64(scalar);  // 广播标量到向量
    
    // NEON向量化处理
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vr = vmulq_u64_emulated(va, vs);
        vst1q_u64(result + i, vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

#ifdef SIMD_ARCH_ARM64
// 融合乘加运算 (a * b + c)
void NEONKernels::fma_u64(const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t* result, size_t count) {
    size_t simd_count = (count / 2) * 2;
    
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vc = vld1q_u64(c + i);
        uint64x2_t prod = vmulq_u64_emulated(va, vb);
        uint64x2_t vr = vaddq_u64(prod, vc);
        vst1q_u64(result + i, vr);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// 优化的矩阵乘法块
void NEONKernels::matmul_block_u64(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                  size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    // 使用分块策略提高缓存效率
    constexpr size_t BLOCK_SIZE = 32;
    
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);
                
                // 计算子块
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; ++jj) {
                        uint64x2_t sum_vec = vdupq_n_u64(0);
                        size_t kk;
                        
                        // NEON向量化内积
                        for (kk = k; kk + 1 < k_end; kk += 2) {
                            uint64x2_t a_vec = vld1q_u64(&A[ii * lda + kk]);
                            uint64x2_t b_vec = {B[kk * ldb + jj], B[(kk + 1) * ldb + jj]};
                            uint64x2_t prod = vmulq_u64_emulated(a_vec, b_vec);
                            sum_vec = vaddq_u64(sum_vec, prod);
                        }
                        
                        // 归约求和
                        uint64_t sum = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);
                        
                        // 处理剩余元素
                        for (; kk < k_end; ++kk) {
                            sum += A[ii * lda + kk] * B[kk * ldb + jj];
                        }
                        
                        C[ii * ldc + jj] += sum;
                    }
                }
            }
        }
    }
}

// 归约求和
uint64_t NEONKernels::reduce_sum_u64(const uint64_t* data, size_t count) {
    uint64x2_t sum_vec = vdupq_n_u64(0);
    size_t simd_count = (count / 2) * 2;
    
    // NEON向量化累加
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t vdata = vld1q_u64(data + i);
        sum_vec = vaddq_u64(sum_vec, vdata);
    }
    
    // 向量内部求和
    uint64_t result = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result += data[i];
    }
    
    return result;
}

// 最大值 (注意：NEON没有64位最大值指令，需要模拟)
uint64_t NEONKernels::reduce_max_u64(const uint64_t* data, size_t count) {
    if (count == 0) return 0;
    
    uint64_t result = data[0];
    
    // 简单的标量实现，因为NEON缺少64位比较指令
    for (size_t i = 1; i < count; ++i) {
        if (data[i] > result) {
            result = data[i];
        }
    }
    
    return result;
}

// 最小值
uint64_t NEONKernels::reduce_min_u64(const uint64_t* data, size_t count) {
    if (count == 0) return 0;
    
    uint64_t result = data[0];
    
    // 简单的标量实现，因为NEON缺少64位比较指令
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < result) {
            result = data[i];
        }
    }
    
    return result;
}
#endif

// 点积运算
uint64_t NEONKernels::dot_product_u64(const uint64_t* a, const uint64_t* b, size_t count) {
    uint64x2_t sum_vec = vdupq_n_u64(0);
    size_t simd_count = (count / 2) * 2;
    
    // NEON向量化点积
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t prod = vmulq_u64_emulated(va, vb);
        sum_vec = vaddq_u64(sum_vec, prod);
    }
    
    // 向量内部求和
    uint64_t result = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

// 内存操作
void NEONKernels::memcpy_aligned(void* dst, const void* src, size_t bytes) {
    // 对齐的内存复制可以使用NEON优化
    if (bytes >= 16 && (reinterpret_cast<uintptr_t>(dst) % 16 == 0) && 
        (reinterpret_cast<uintptr_t>(src) % 16 == 0)) {
        
        uint8_t* d = static_cast<uint8_t*>(dst);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        size_t simd_bytes = (bytes / 16) * 16;
        
        for (size_t i = 0; i < simd_bytes; i += 16) {
            uint8x16_t data = vld1q_u8(s + i);
            vst1q_u8(d + i, data);
        }
        
        // 复制剩余字节
        std::memcpy(d + simd_bytes, s + simd_bytes, bytes - simd_bytes);
    } else {
        std::memcpy(dst, src, bytes);
    }
}

void NEONKernels::memset_u64(uint64_t* dst, uint64_t value, size_t count) {
    size_t simd_count = (count / 2) * 2;
    uint64x2_t v = vdupq_n_u64(value);
    
    // NEON向量化设置
    for (size_t i = 0; i < simd_count; i += 2) {
        vst1q_u64(dst + i, v);
    }
    
    // 处理剩余元素
    for (size_t i = simd_count; i < count; ++i) {
        dst[i] = value;
    }
}

#endif // SIMD_HAS_NEON

// ARM调度器相关实现
static bool arm_dispatcher_initialized = false;
static bool neon_available = false;
static bool sve_available = false;

void ARMDispatcher::initialize() {
    if (arm_dispatcher_initialized) return;
    
    #ifdef SIMD_HAS_NEON
    // 简单检测：如果编译时启用了NEON，就认为可用
    neon_available = true;
    #else
    neon_available = false;
    #endif
    
    #ifdef SIMD_HAS_SVE
    sve_available = true;
    #else
    sve_available = false;
    #endif
    
    arm_dispatcher_initialized = true;
}

bool ARMDispatcher::is_neon_available() {
    initialize();
    return neon_available;
}

bool ARMDispatcher::is_sve_available() {
    initialize();
    return sve_available;
}

void ARMDispatcher::add_u64_auto(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    #ifdef SIMD_HAS_NEON
    if (is_neon_available()) {
        NEONKernels::add_u64(a, b, result, count);
        return;
    }
    #endif
    
    // 标量回退
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void ARMDispatcher::multiply_u64_auto(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    #ifdef SIMD_HAS_NEON
    if (is_neon_available()) {
        NEONKernels::multiply_u64(a, b, result, count);
        return;
    }
    #endif
    
    // 标量回退
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

// ARM工具函数实现
bool ARMUtils::is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

void ARMUtils::prefetch_read(const void* addr) {
    #ifdef __ARM_ACLE
    __builtin_prefetch(addr, 0, 3);
    #else
    (void)addr;
    #endif
}

void ARMUtils::prefetch_write(void* addr) {
    #ifdef __ARM_ACLE
    __builtin_prefetch(addr, 1, 3);
    #else
    (void)addr;
    #endif
}

void ARMUtils::cache_flush(const void* addr, size_t size) {
    (void)addr; (void)size;
    // 平台特定的缓存刷新实现
}

void ARMUtils::cache_invalidate(const void* addr, size_t size) {
    (void)addr; (void)size;
    // 平台特定的缓存无效化实现
}

size_t ARMUtils::get_optimal_block_size(size_t data_size) {
    return std::min(data_size, static_cast<size_t>(1024));
}

bool ARMUtils::supports_crypto() {
    return false;  // 需要运行时检测
}

bool ARMUtils::supports_fp16() {
    return false;  // 需要运行时检测
}

bool ARMUtils::supports_dotprod() {
    return false;  // 需要运行时检测
}

#ifdef SIMD_ARCH_ARM64
uint64_t ARMUtils::get_cycle_counter() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

void ARMUtils::memory_barrier() {
    asm volatile("dmb sy" ::: "memory");
}
#endif

} // namespace simd_arm
} // namespace uint64_compute

#endif // ARM64 platform check