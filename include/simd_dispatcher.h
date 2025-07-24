#pragma once

#include "simd_detect.h"
#include "simd_x86.h"
#include "simd_arm.h"
#include <cstdint>
#include <functional>

namespace uint64_compute {

// 统一的SIMD函数指针类型
using AddFunc = std::function<void(const uint64_t*, const uint64_t*, uint64_t*, size_t)>;
using MulFunc = std::function<void(const uint64_t*, const uint64_t*, uint64_t*, size_t)>;
using AddScalarFunc = std::function<void(const uint64_t*, uint64_t, uint64_t*, size_t)>;
using MulScalarFunc = std::function<void(const uint64_t*, uint64_t, uint64_t*, size_t)>;
using MatMulFunc = std::function<void(const uint64_t*, const uint64_t*, uint64_t*, size_t, size_t, size_t, size_t, size_t, size_t)>;

// 性能测试结果
struct SIMDPerformance {
    double latency_ns;     // 延迟（纳秒）
    double throughput_gops; // 吞吐量（GOPS - 每秒千兆运算）
    std::string impl_name;  // 实现名称
};

// 跨平台SIMD调度器
class CrossPlatformSIMDDispatcher {
private:
    static bool initialized;
    static std::string selected_impl;
    
    // 函数指针存储
    static AddFunc best_add_func;
    static MulFunc best_mul_func;
    static AddScalarFunc best_add_scalar_func;
    static MulScalarFunc best_mul_scalar_func;
    static MatMulFunc best_matmul_func;
    
    // 标量实现（回退）
    static void scalar_add(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void scalar_multiply(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void scalar_add_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void scalar_multiply_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void scalar_matmul(const uint64_t* A, const uint64_t* B, uint64_t* C,
                             size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
    
    // 性能测试
    static SIMDPerformance benchmark_implementation(const std::string& name, 
                                                   AddFunc add_func, 
                                                   MulFunc mul_func,
                                                   size_t test_size = 10000);
    
    // 初始化函数
    static void initialize_x86();
    static void initialize_arm();
    static void select_best_implementation();
    
public:
    // 初始化调度器
    static void initialize();
    
    // 获取信息
    static std::string get_selected_implementation() { return selected_impl; }
    static std::string get_available_implementations();
    static SIMDPerformance get_performance_info();
    
    // 主要API - 自动选择最佳实现
    static void add(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void multiply(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void add_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void multiply_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    static void matmul(const uint64_t* A, const uint64_t* B, uint64_t* C,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
    
    // 手动选择实现（用于测试和调优）
    static bool set_implementation(const std::string& impl_name);
    static void reset_to_auto();
    
    // 实用工具
    static bool is_memory_aligned(const void* ptr, size_t alignment = 32);
    static void prefetch_for_read(const void* addr);
    static void prefetch_for_write(void* addr);
    static size_t get_optimal_block_size(size_t data_size);
};

// 便捷的RAII性能分析器
class SIMDProfiler {
private:
    std::string operation_name;
    size_t element_count;
    uint64_t start_time;
    
public:
    SIMDProfiler(const std::string& op_name, size_t count);
    ~SIMDProfiler();
    
    static void print_summary();
    static void reset_stats();
};

// 自适应批处理器 - 自动选择最佳批大小
class AdaptiveBatcher {
private:
    static size_t optimal_batch_size;
    static bool calibrated;
    
    static void calibrate();
    
public:
    static void process_add(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    static void process_multiply(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    
    static size_t get_optimal_batch_size();
    static void set_batch_size(size_t size);
};

// 全局便捷函数
namespace ops {
    // 基本运算 - 使用最佳SIMD实现
    void add_optimized(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    void multiply_optimized(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count);
    void add_scalar_optimized(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    void multiply_scalar_optimized(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count);
    
    // 矩阵运算
    void matmul_optimized(const uint64_t* A, const uint64_t* B, uint64_t* C,
                         size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
    
    // 高级运算
    uint64_t dot_product_optimized(const uint64_t* a, const uint64_t* b, size_t count);
    uint64_t reduce_sum_optimized(const uint64_t* data, size_t count);
    uint64_t reduce_max_optimized(const uint64_t* data, size_t count);
    uint64_t reduce_min_optimized(const uint64_t* data, size_t count);
    
    // 融合运算 (Fused Multiply-Add)
    void fma_optimized(const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t* result, size_t count);
    
    // 内存操作
    void memcpy_optimized(void* dst, const void* src, size_t bytes);
    void memset_u64_optimized(uint64_t* dst, uint64_t value, size_t count);
}

} // namespace uint64_compute