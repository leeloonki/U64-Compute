#include "simd_dispatcher.h"
#include "simd_detect.h"
#include "simd_x86.h"
#include "simd_arm.h"
#include <chrono>
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <cstring>

namespace uint64_compute {

// 静态成员初始化
bool CrossPlatformSIMDDispatcher::initialized = false;
std::string CrossPlatformSIMDDispatcher::selected_impl = "scalar";
AddFunc CrossPlatformSIMDDispatcher::best_add_func = nullptr;
MulFunc CrossPlatformSIMDDispatcher::best_mul_func = nullptr;
AddScalarFunc CrossPlatformSIMDDispatcher::best_add_scalar_func = nullptr;
MulScalarFunc CrossPlatformSIMDDispatcher::best_mul_scalar_func = nullptr;
MatMulFunc CrossPlatformSIMDDispatcher::best_matmul_func = nullptr;

// 标量实现
void CrossPlatformSIMDDispatcher::scalar_add(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void CrossPlatformSIMDDispatcher::scalar_multiply(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void CrossPlatformSIMDDispatcher::scalar_add_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void CrossPlatformSIMDDispatcher::scalar_multiply_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

void CrossPlatformSIMDDispatcher::scalar_matmul(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                               size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            uint64_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// 性能测试
SIMDPerformance CrossPlatformSIMDDispatcher::benchmark_implementation(const std::string& name, 
                                                                     AddFunc add_func, 
                                                                     MulFunc /* mul_func */,
                                                                     size_t test_size) {
    SIMDPerformance perf;
    perf.impl_name = name;
    
    // 准备测试数据
    std::vector<uint64_t> a(test_size);
    std::vector<uint64_t> b(test_size);
    std::vector<uint64_t> result(test_size);
    
    // 随机填充数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1, 1000);
    
    for (size_t i = 0; i < test_size; ++i) {
        a[i] = dis(gen);
        b[i] = dis(gen);
    }
    
    // 预热
    add_func(a.data(), b.data(), result.data(), test_size);
    
    // 测试加法性能
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        add_func(a.data(), b.data(), result.data(), test_size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    perf.latency_ns = static_cast<double>(duration.count()) / iterations;
    perf.throughput_gops = (test_size * 1e9) / perf.latency_ns / 1e9; // GOPS
    
    return perf;
}

void CrossPlatformSIMDDispatcher::initialize_x86() {
    #if defined(SIMD_ARCH_X86_64) || defined(SIMD_ARCH_X86)
        // 可用的x86实现按优先级排序
        std::vector<std::pair<std::string, std::function<void()>>> implementations;
        
        #ifdef SIMD_HAS_AVX512F
        std::cout << "Checking AVX512F support..." << std::endl;
        bool avx512f_available = CrossPlatformSIMD::has_avx512f();
        std::cout << "AVX512F detection result: " << (avx512f_available ? "YES" : "NO") << std::endl;
        if (avx512f_available) {
            std::cout << "AVX512F detected and added!" << std::endl;
            implementations.emplace_back("AVX512", []() {
                best_add_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                    simd_x86::AVX512Kernels::add_u64(a, b, result, count);
                };
                best_mul_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                    simd_x86::AVX512Kernels::multiply_u64(a, b, result, count);
                };
                best_add_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                    simd_x86::AVX512Kernels::add_scalar_u64(a, scalar, result, count);
                };
                best_mul_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                    simd_x86::AVX512Kernels::multiply_scalar_u64(a, scalar, result, count);
                };
            });
        }
        #endif
        
        #ifdef SIMD_HAS_AVX2
        std::cout << "Checking AVX2 support..." << std::endl;
        bool avx2_available = CrossPlatformSIMD::has_avx2();
        std::cout << "AVX2 detection result: " << (avx2_available ? "YES" : "NO") << std::endl;
        if (avx2_available) {
            implementations.emplace_back("AVX2", []() {
                best_add_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                    simd_x86::AVX2Kernels::add_u64(a, b, result, count);
                };
                best_mul_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                    simd_x86::AVX2Kernels::multiply_u64(a, b, result, count);
                };
                best_add_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                    simd_x86::AVX2Kernels::add_scalar_u64(a, scalar, result, count);
                };
                best_mul_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                    simd_x86::AVX2Kernels::multiply_scalar_u64(a, scalar, result, count);
                };
                best_matmul_func = [](const uint64_t* A, const uint64_t* B, uint64_t* C,
                                     size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
                    simd_x86::AVX2Kernels::matmul_block_u64(A, B, C, M, N, K, lda, ldb, ldc);
                };
            });
        }
        #endif
        
        if (CrossPlatformSIMD::has_sse2()) {
            implementations.emplace_back("SSE2", []() {
                best_add_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                    simd_x86::SSE2Kernels::add_u64(a, b, result, count);
                };
                best_mul_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                    simd_x86::SSE2Kernels::multiply_u64(a, b, result, count);
                };
                best_add_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                    simd_x86::SSE2Kernels::add_scalar_u64(a, scalar, result, count);
                };
                best_mul_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                    simd_x86::SSE2Kernels::multiply_scalar_u64(a, scalar, result, count);
                };
            });
        }
        
        // 选择最佳实现
        if (!implementations.empty()) {
            implementations[0].second(); // 使用优先级最高的
            selected_impl = implementations[0].first;
            std::cout << "Selected x86 SIMD: " << selected_impl << std::endl;
        }
    #endif
}

void CrossPlatformSIMDDispatcher::initialize_arm() {
    #if defined(SIMD_ARCH_ARM64)
        simd_arm::ARMDispatcher::initialize();
        
        #ifdef SIMD_HAS_NEON
        if (simd_arm::ARMDispatcher::is_neon_available()) {
            best_add_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                simd_arm::NEONKernels::add_u64(a, b, result, count);
            };
            best_mul_func = [](const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
                simd_arm::NEONKernels::multiply_u64(a, b, result, count);
            };
            best_add_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                simd_arm::NEONKernels::add_scalar_u64(a, scalar, result, count);
            };
            best_mul_scalar_func = [](const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
                simd_arm::NEONKernels::multiply_scalar_u64(a, scalar, result, count);
            };
            
            
            best_matmul_func = [](const uint64_t* A, const uint64_t* B, uint64_t* C,
                                 size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
                simd_arm::NEONKernels::matmul_block_u64(A, B, C, M, N, K, lda, ldb, ldc);
            };
            

            selected_impl = "NEON";
            std::cout << "Selected ARM SIMD: NEON" << std::endl;
            return;
        }
        #endif
    #endif
}

void CrossPlatformSIMDDispatcher::select_best_implementation() {
    // 设置默认的标量实现
    best_add_func = scalar_add;
    best_mul_func = scalar_multiply;
    best_add_scalar_func = scalar_add_scalar;
    best_mul_scalar_func = scalar_multiply_scalar;
    best_matmul_func = scalar_matmul;
    
    // 根据架构选择最佳实现
    CrossPlatformSIMD::detect();
    ArchType arch = CrossPlatformSIMD::get_architecture();
    
    switch (arch) {
        case ArchType::X86_64:
            initialize_x86();
            break;
        case ArchType::ARM64:
            initialize_arm();
            break;
        default:
            std::cout << "Using scalar implementation (unknown architecture)" << std::endl;
            break;
    }
}

void CrossPlatformSIMDDispatcher::initialize() {
    if (initialized) return;
    
    std::cout << "Initializing CrossPlatform SIMD Dispatcher..." << std::endl;
    std::cout << "Architecture: ";
    
    ArchType arch = CrossPlatformSIMD::get_architecture();
    switch (arch) {
        case ArchType::X86_64: std::cout << "x86-64"; break;
        case ArchType::ARM64:  std::cout << "ARM64"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
    
    std::cout << "Available SIMD features: " << CrossPlatformSIMD::get_feature_string() << std::endl;
    std::cout << "Best SIMD width: " << CrossPlatformSIMD::get_best_simd_width() << " bits" << std::endl;
    
    select_best_implementation();
    
    initialized = true;
    std::cout << "SIMD Dispatcher initialized successfully!" << std::endl << std::endl;
}

std::string CrossPlatformSIMDDispatcher::get_available_implementations() {
    std::string result = "Available implementations: Scalar";
    
    #ifdef SIMD_ARCH_X86
        if (CrossPlatformSIMD::has_sse2()) result += ", SSE2";
        if (CrossPlatformSIMD::has_avx()) result += ", AVX";
        if (CrossPlatformSIMD::has_avx2()) result += ", AVX2";
        if (CrossPlatformSIMD::has_avx512f()) result += ", AVX512F";
    #endif
    
    #if defined(SIMD_ARCH_ARM64)
        if (CrossPlatformSIMD::has_neon()) result += ", NEON";
    #endif
    
    return result;
}

SIMDPerformance CrossPlatformSIMDDispatcher::get_performance_info() {
    initialize();
    return benchmark_implementation(selected_impl, best_add_func, best_mul_func);
}

// OpenMP + SIMD 结合的配置
static constexpr size_t SIMD_BLOCK_SIZE = 256;       // 每个线程的SIMD块大小

// 获取全局阈值的函数声明
namespace ops {
    class PerformanceConfig {
    public:
        static size_t get_parallel_threshold();
    };
}

// 主要API实现 - 自动选择串行或并行
void CrossPlatformSIMDDispatcher::add(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    initialize();
    
    #ifdef _OPENMP
    size_t threshold = ops::PerformanceConfig::get_parallel_threshold();
    if (count >= threshold) {
        // OpenMP并行 + SIMD向量化
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i += SIMD_BLOCK_SIZE) {
            size_t block_size = std::min(SIMD_BLOCK_SIZE, count - i);
            best_add_func(a + i, b + i, result + i, block_size);
        }
    } else {
        // 小数据量直接使用SIMD
        best_add_func(a, b, result, count);
    }
    #else
    best_add_func(a, b, result, count);
    #endif
}

void CrossPlatformSIMDDispatcher::multiply(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    initialize();
    
    #ifdef _OPENMP
    size_t threshold = ops::PerformanceConfig::get_parallel_threshold();
    if (count >= threshold) {
        // OpenMP并行 + SIMD向量化
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i += SIMD_BLOCK_SIZE) {
            size_t block_size = std::min(SIMD_BLOCK_SIZE, count - i);
            best_mul_func(a + i, b + i, result + i, block_size);
        }
    } else {
        // 小数据量直接使用SIMD
        best_mul_func(a, b, result, count);
    }
    #else
    best_mul_func(a, b, result, count);
    #endif
}

void CrossPlatformSIMDDispatcher::add_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    initialize();
    
    #ifdef _OPENMP
    size_t threshold = ops::PerformanceConfig::get_parallel_threshold();
    if (count >= threshold) {
        // OpenMP并行 + SIMD向量化
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i += SIMD_BLOCK_SIZE) {
            size_t block_size = std::min(SIMD_BLOCK_SIZE, count - i);
            best_add_scalar_func(a + i, scalar, result + i, block_size);
        }
    } else {
        // 小数据量直接使用SIMD
        best_add_scalar_func(a, scalar, result, count);
    }
    #else
    best_add_scalar_func(a, scalar, result, count);
    #endif
}

void CrossPlatformSIMDDispatcher::multiply_scalar(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    initialize();
    
    #ifdef _OPENMP
    size_t threshold = ops::PerformanceConfig::get_parallel_threshold();
    if (count >= threshold) {
        // OpenMP并行 + SIMD向量化
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; i += SIMD_BLOCK_SIZE) {
            size_t block_size = std::min(SIMD_BLOCK_SIZE, count - i);
            best_mul_scalar_func(a + i, scalar, result + i, block_size);
        }
    } else {
        // 小数据量直接使用SIMD
        best_mul_scalar_func(a, scalar, result, count);
    }
    #else
    best_mul_scalar_func(a, scalar, result, count);
    #endif
}

void CrossPlatformSIMDDispatcher::matmul(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                        size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    initialize();
    
    // 添加参数验证
    if (!A || !B || !C) {
        throw std::runtime_error("Null pointer in matmul");
    }
    if (M == 0 || N == 0 || K == 0) {
        return; // 空矩阵
    }
    if (lda < K || ldb < N || ldc < N) {
        throw std::runtime_error("Invalid leading dimensions in matmul");
    }
    
    #ifdef _OPENMP
    size_t threshold = ops::PerformanceConfig::get_parallel_threshold();
    if (false) { // 临时禁用OpenMP并行化
        // 矩阵乘法的OpenMP并行化 - 按行并行，添加边界检查
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < M; ++i) {
            // 每个线程处理一行，内部使用SIMD优化
            for (size_t j = 0; j < N; ++j) {
                uint64_t sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    // 计算数组索引
                    size_t a_idx = i * lda + k;
                    size_t b_idx = k * ldb + j;
                    
                    // 正确的边界检查：检查是否超出了合理的索引范围
                    // A矩阵是M×K，所以最大索引是(M-1)*lda + (K-1)
                    // B矩阵是K×N，所以最大索引是(K-1)*ldb + (N-1)
                    if (i >= M || k >= K || j >= N || 
                        a_idx >= M * lda || b_idx >= K * ldb) {
                        continue; // 跳过越界访问
                    }
                    
                    sum += A[a_idx] * B[b_idx];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else {
        // 小矩阵直接使用SIMD优化的分块矩阵乘法
        best_matmul_func(A, B, C, M, N, K, lda, ldb, ldc);
    }
    #else
    best_matmul_func(A, B, C, M, N, K, lda, ldb, ldc);
    #endif
}

bool CrossPlatformSIMDDispatcher::is_memory_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

void CrossPlatformSIMDDispatcher::prefetch_for_read(const void* addr) {
    #ifdef SIMD_ARCH_X86
        simd_x86::X86Utils::prefetch_read(addr);
    #elif defined(SIMD_ARCH_ARM64)
        simd_arm::ARMUtils::prefetch_read(addr);
    #else
        (void)addr; // Suppress unused parameter warning
    #endif
}

void CrossPlatformSIMDDispatcher::prefetch_for_write(void* addr) {
    #ifdef SIMD_ARCH_X86
        simd_x86::X86Utils::prefetch_write(addr);
    #elif defined(SIMD_ARCH_ARM64)
        simd_arm::ARMUtils::prefetch_write(addr);
    #else
        (void)addr; // Suppress unused parameter warning
    #endif
}

size_t CrossPlatformSIMDDispatcher::get_optimal_block_size(size_t data_size) {
    #ifdef SIMD_ARCH_X86
        return simd_x86::X86Utils::get_optimal_block_size(data_size);
    #elif defined(SIMD_ARCH_ARM64)
        return simd_arm::ARMUtils::get_optimal_block_size(data_size);
    #else
        return std::min(data_size, static_cast<size_t>(1024));
    #endif
}

// SIMDProfiler实现
static std::map<std::string, std::vector<double>> profiler_stats;

SIMDProfiler::SIMDProfiler(const std::string& op_name, size_t count) 
    : operation_name(op_name), element_count(count) {
    start_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

SIMDProfiler::~SIMDProfiler() {
    uint64_t end_time = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    double duration_ns = static_cast<double>(end_time - start_time);
    double throughput = (element_count * 1e9) / duration_ns; // elements per second
    
    profiler_stats[operation_name].push_back(throughput);
}

void SIMDProfiler::print_summary() {
    std::cout << "\n=== SIMD Performance Summary ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    for (const auto& pair : profiler_stats) {
        const auto& op_name = pair.first;
        const auto& measurements = pair.second;
        if (measurements.empty()) continue;
        
        double avg = 0.0;
        double min_val = measurements[0];
        double max_val = measurements[0];
        
        for (double val : measurements) {
            avg += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        avg /= measurements.size();
        
        std::cout << op_name << ":" << std::endl;
        std::cout << "  Average: " << avg / 1e6 << " M elements/sec" << std::endl;
        std::cout << "  Min:     " << min_val / 1e6 << " M elements/sec" << std::endl;
        std::cout << "  Max:     " << max_val / 1e6 << " M elements/sec" << std::endl;
        std::cout << "  Samples: " << measurements.size() << std::endl;
    }
    std::cout << "=================================" << std::endl;
}

void SIMDProfiler::reset_stats() {
    profiler_stats.clear();
}

// AdaptiveBatcher实现
size_t AdaptiveBatcher::optimal_batch_size = 1024;
bool AdaptiveBatcher::calibrated = false;

void AdaptiveBatcher::calibrate() {
    if (calibrated) return;
    
    const size_t test_sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    const size_t total_elements = 100000;
    
    std::vector<uint64_t> a(total_elements);
    std::vector<uint64_t> b(total_elements);
    std::vector<uint64_t> result(total_elements);
    
    double best_throughput = 0;
    size_t best_size = 1024;
    
    for (size_t batch_size : test_sizes) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < total_elements; i += batch_size) {
            size_t current_batch = std::min(batch_size, total_elements - i);
            CrossPlatformSIMDDispatcher::add(a.data() + i, b.data() + i, result.data() + i, current_batch);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double throughput = (total_elements * 1e9) / duration.count();
        
        if (throughput > best_throughput) {
            best_throughput = throughput;
            best_size = batch_size;
        }
    }
    
    optimal_batch_size = best_size;
    calibrated = true;
}

void AdaptiveBatcher::process_add(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    calibrate();
    
    for (size_t i = 0; i < count; i += optimal_batch_size) {
        size_t current_batch = std::min(optimal_batch_size, count - i);
        CrossPlatformSIMDDispatcher::add(a + i, b + i, result + i, current_batch);
    }
}

void AdaptiveBatcher::process_multiply(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    calibrate();
    
    for (size_t i = 0; i < count; i += optimal_batch_size) {
        size_t current_batch = std::min(optimal_batch_size, count - i);
        CrossPlatformSIMDDispatcher::multiply(a + i, b + i, result + i, current_batch);
    }
}

size_t AdaptiveBatcher::get_optimal_batch_size() {
    calibrate();
    return optimal_batch_size;
}

void AdaptiveBatcher::set_batch_size(size_t size) {
    optimal_batch_size = size;
    calibrated = true;
}

// 全局便捷函数实现
namespace ops {
    void add_optimized(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
        CrossPlatformSIMDDispatcher::add(a, b, result, count);
    }
    
    void multiply_optimized(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
        CrossPlatformSIMDDispatcher::multiply(a, b, result, count);
    }
    
    void add_scalar_optimized(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
        CrossPlatformSIMDDispatcher::add_scalar(a, scalar, result, count);
    }
    
    void multiply_scalar_optimized(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
        CrossPlatformSIMDDispatcher::multiply_scalar(a, scalar, result, count);
    }
    
    void matmul_optimized(const uint64_t* A, const uint64_t* B, uint64_t* C,
                         size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
        CrossPlatformSIMDDispatcher::matmul(A, B, C, M, N, K, lda, ldb, ldc);
    }
    
    uint64_t dot_product_optimized(const uint64_t* a, const uint64_t* b, size_t count) {
        #ifdef SIMD_HAS_NEON
        if (CrossPlatformSIMD::has_neon()) {
            return simd_arm::NEONKernels::dot_product_u64(a, b, count);
        }
        #endif
        
        // 标量实现
        uint64_t result = 0;
        for (size_t i = 0; i < count; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    uint64_t reduce_sum_optimized(const uint64_t* data, size_t count) {
        #ifdef SIMD_ARCH_ARM64
        if (CrossPlatformSIMD::has_neon()) {
            return simd_arm::NEONKernels::reduce_sum_u64(data, count);
        }
        #endif
        
        // 标量实现
        uint64_t result = 0;
        for (size_t i = 0; i < count; ++i) {
            result += data[i];
        }
        return result;
    }
    
    uint64_t reduce_max_optimized(const uint64_t* data, size_t count) {
        #ifdef SIMD_ARCH_ARM64
        if (CrossPlatformSIMD::has_neon()) {
            return simd_arm::NEONKernels::reduce_max_u64(data, count);
        }
        #endif
        
        // 标量实现
        if (count == 0) return 0;
        uint64_t result = data[0];
        for (size_t i = 1; i < count; ++i) {
            result = std::max(result, data[i]);
        }
        return result;
    }
    
    uint64_t reduce_min_optimized(const uint64_t* data, size_t count) {
        #ifdef SIMD_ARCH_ARM64
        if (CrossPlatformSIMD::has_neon()) {
            return simd_arm::NEONKernels::reduce_min_u64(data, count);
        }
        #endif
        
        // 标量实现
        if (count == 0) return UINT64_MAX;
        uint64_t result = data[0];
        for (size_t i = 1; i < count; ++i) {
            result = std::min(result, data[i]);
        }
        return result;
    }
    
    void fma_optimized(const uint64_t* a, const uint64_t* b, const uint64_t* c, uint64_t* result, size_t count) {
        #ifdef SIMD_HAS_AVX512F
        if (CrossPlatformSIMD::has_avx512f()) {
            simd_x86::AVX512Kernels::fma_u64(a, b, c, result, count);
            return;
        }
        #endif
        
        #ifdef SIMD_ARCH_ARM64
        if (CrossPlatformSIMD::has_neon()) {
            simd_arm::NEONKernels::fma_u64(a, b, c, result, count);
            return;
        }
        #endif
        
        // 标量实现
        for (size_t i = 0; i < count; ++i) {
            result[i] = a[i] * b[i] + c[i];
        }
    }
    
    void memcpy_optimized(void* dst, const void* src, size_t bytes) {
        #ifdef SIMD_HAS_NEON
        if (CrossPlatformSIMD::has_neon() && CrossPlatformSIMDDispatcher::is_memory_aligned(dst, 16) && 
            CrossPlatformSIMDDispatcher::is_memory_aligned(src, 16)) {
            simd_arm::NEONKernels::memcpy_aligned(dst, src, bytes);
            return;
        }
        #endif
        
        std::memcpy(dst, src, bytes);
    }
    
    void memset_u64_optimized(uint64_t* dst, uint64_t value, size_t count) {
        #ifdef SIMD_HAS_NEON
        if (CrossPlatformSIMD::has_neon()) {
            simd_arm::NEONKernels::memset_u64(dst, value, count);
            return;
        }
        #endif
        
        // 标量实现
        for (size_t i = 0; i < count; ++i) {
            dst[i] = value;
        }
    }
}

} // namespace uint64_compute