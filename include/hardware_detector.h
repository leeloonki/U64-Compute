#pragma once

// Advanced Hardware Detection and Optimization Selection
// This header provides runtime hardware detection and automatic optimization selection

#include "uint64_compute_config.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace uint64_compute {
namespace hardware {

// ================================
// Hardware Capability Flags
// ================================

enum class CPUFeature : uint32_t {
    // x86 features
    SSE2     = 1 << 0,
    SSE41    = 1 << 1,
    AVX      = 1 << 2,
    AVX2     = 1 << 3,
    AVX512F  = 1 << 4,
    AVX512BW = 1 << 5,
    AVX512VL = 1 << 6,
    
    // ARM features
    NEON     = 1 << 8,
    SVE      = 1 << 9,
    SVE2     = 1 << 10,
    
    // Additional features
    FMA      = 1 << 16,
    BMI1     = 1 << 17,
    BMI2     = 1 << 18,
    
    // Cache features
    CLFLUSH  = 1 << 24,
    PREFETCH = 1 << 25
};

enum class MemoryFeature : uint32_t {
    ALIGNED_MALLOC = 1 << 0,
    HUGE_PAGES     = 1 << 1,
    NUMA_AWARE     = 1 << 2
};

enum class ParallelFeature : uint32_t {
    OPENMP      = 1 << 0,
    STD_THREAD  = 1 << 1,
    CUDA        = 1 << 2,
    OPENCL      = 1 << 3
};

// ================================
// Hardware Information Structure
// ================================

struct CPUInfo {
    std::string vendor;
    std::string brand;
    std::string architecture;
    int cores_physical;
    int cores_logical;
    int cache_l1_size;      // KB
    int cache_l2_size;      // KB
    int cache_l3_size;      // KB
    int cache_line_size;    // bytes
    uint32_t features;      // CPUFeature flags
};

struct MemoryInfo {
    size_t total_ram;       // bytes
    size_t available_ram;   // bytes
    int numa_nodes;
    uint32_t features;      // MemoryFeature flags
};

struct GPUInfo {
    bool available;
    std::string name;
    std::string driver_version;
    size_t memory_total;    // bytes
    size_t memory_available; // bytes
    int compute_units;
    int max_threads_per_block;
};

struct SystemInfo {
    CPUInfo cpu;
    MemoryInfo memory;
    GPUInfo gpu;
    uint32_t parallel_features; // ParallelFeature flags
    std::string os_name;
    std::string os_version;
};

// ================================
// Performance Characteristics
// ================================

struct PerformanceProfile {
    std::string implementation_name;
    double throughput_gops;     // Giga operations per second
    double latency_ns;          // Nanoseconds per operation
    double efficiency_score;    // Composite performance score
    size_t memory_bandwidth;    // MB/s
    bool supports_inplace;      // Supports in-place operations
    size_t optimal_block_size;  // Optimal processing block size
};

// ================================
// Hardware Detector Class
// ================================

class HardwareDetector {
public:
    // Singleton access
    static HardwareDetector& instance();
    
    // Hardware detection
    const SystemInfo& detect_system() const;
    bool has_feature(CPUFeature feature) const;
    bool has_feature(MemoryFeature feature) const;
    bool has_feature(ParallelFeature feature) const;
    
    // Performance profiling
    std::vector<PerformanceProfile> benchmark_implementations(size_t test_size = 100000);
    const PerformanceProfile& get_best_implementation() const;
    
    // Architecture queries
    bool is_x86() const;
    bool is_arm() const;
    bool is_64bit() const;
    int get_simd_width() const;     // Best SIMD width in bits
    int get_optimal_thread_count() const;
    size_t get_cache_line_size() const;
    
    // Memory optimization hints
    size_t get_optimal_alignment() const;
    size_t get_optimal_block_size(size_t data_size) const;
    bool should_use_parallel(size_t data_size) const;
    
    // Debug information
    std::string get_system_summary() const;
    void print_system_info() const;
    
private:
    HardwareDetector() = default;
    ~HardwareDetector() = default;
    HardwareDetector(const HardwareDetector&) = delete;
    HardwareDetector& operator=(const HardwareDetector&) = delete;
    
    // Internal detection methods
    void detect_cpu_info() const;
    void detect_memory_info() const;
    void detect_gpu_info() const;
    void detect_parallel_features() const;
    
#ifdef UINT64COMPUTE_ARCH_X86_64
    void detect_x86_features() const;
    void run_cpuid(uint32_t function, uint32_t subfunction, uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) const;
#endif

#ifdef UINT64COMPUTE_ARCH_ARM64
    void detect_arm_features() const;
#endif
    
    // Performance benchmarking
    PerformanceProfile benchmark_scalar(size_t test_size);
    PerformanceProfile benchmark_simd(const std::string& impl_name, size_t test_size);
    
    // State
    mutable bool detection_done = false;
    mutable SystemInfo system_info_;
    mutable PerformanceProfile best_implementation_;
    mutable bool performance_profiled_ = false;
};

// ================================
// Optimization Strategy Selector
// ================================

enum class OptimizationStrategy {
    SCALAR,           // No SIMD, single-threaded
    SIMD_ONLY,        // SIMD but single-threaded
    PARALLEL_ONLY,    // Multi-threaded but no SIMD
    SIMD_PARALLEL,    // Both SIMD and multi-threading
    GPU_OFFLOAD,      // Use GPU acceleration
    ADAPTIVE          // Runtime decision based on data size
};

class OptimizationSelector {
public:
    OptimizationSelector();
    
    // Strategy selection
    OptimizationStrategy select_strategy(size_t data_size, const std::string& operation) const;
    void force_strategy(OptimizationStrategy strategy);
    void reset_to_adaptive();
    
    // Operation-specific hints
    bool should_use_gpu(size_t data_size, const std::string& operation) const;
    bool should_use_parallel(size_t data_size, const std::string& operation) const;
    std::string get_best_simd_implementation() const;
    
    // Thresholds (can be tuned)
    void set_parallel_threshold(size_t threshold);
    void set_gpu_threshold(size_t threshold);
    void set_simd_threshold(size_t threshold);
    
    size_t get_parallel_threshold() const { return parallel_threshold_; }
    size_t get_gpu_threshold() const { return gpu_threshold_; }
    size_t get_simd_threshold() const { return simd_threshold_; }
    
private:
    OptimizationStrategy forced_strategy_ = OptimizationStrategy::ADAPTIVE;
    size_t parallel_threshold_ = 1024;
    size_t gpu_threshold_ = 100000;
    size_t simd_threshold_ = 8;
    
    const HardwareDetector& detector_;
};

// ================================
// Memory Allocator with SIMD Alignment
// ================================

class AlignedAllocator {
public:
    static void* allocate(size_t size, size_t alignment = 0);
    static void deallocate(void* ptr);
    static size_t get_default_alignment();
    
    // Convenience functions for uint64_t arrays
    static uint64_t* allocate_uint64_array(size_t count, size_t alignment = 0);
    static void deallocate_uint64_array(uint64_t* ptr);
    
    // Check if pointer is properly aligned
    static bool is_aligned(const void* ptr, size_t alignment);
    
private:
    static size_t round_up_to_alignment(size_t size, size_t alignment);
};

// ================================
// Utility Functions
// ================================

namespace utils {
    // Cache optimization
    inline void prefetch_read(const void* addr) {
        (void)addr; // Simplified version
    }
    
    inline void prefetch_write(void* addr) {
        (void)addr; // Simplified version
    }
    
    // Memory bandwidth testing
    double measure_memory_bandwidth();
    
    // CPU frequency detection
    double get_cpu_frequency_ghz();
    
    // Thread affinity (platform-specific)
    bool set_thread_affinity(int thread_id, int cpu_core);
    
    // Timing utilities
    double get_time_nanoseconds();
    
    // Cache warming
    void warm_cache(const void* data, size_t size);
    
} // namespace utils

// ================================
// Global Configuration
// ================================

// Thread-safe global configuration
class GlobalConfig {
public:
    static GlobalConfig& instance();
    
    // Get current optimization selector
    OptimizationSelector& get_optimizer();
    const OptimizationSelector& get_optimizer() const;
    
    // Performance monitoring
    void enable_performance_monitoring(bool enable = true);
    bool is_performance_monitoring_enabled() const;
    
    // Verbosity control
    void set_verbose(bool verbose = true);
    bool is_verbose() const;
    
    // Auto-tuning
    void enable_auto_tuning(bool enable = true);
    bool is_auto_tuning_enabled() const;
    void run_auto_tuning();
    
private:
    GlobalConfig() = default;
    std::unique_ptr<OptimizationSelector> optimizer_;
    bool performance_monitoring_ = false;
    bool verbose_ = false;
    bool auto_tuning_ = true;
};

// ================================
// Convenience Macros
// ================================

// Get the global instances
#define UINT64COMPUTE_HARDWARE() (uint64_compute::hardware::HardwareDetector::instance())
#define UINT64COMPUTE_OPTIMIZER() (uint64_compute::hardware::GlobalConfig::instance().get_optimizer())
#define UINT64COMPUTE_CONFIG() (uint64_compute::hardware::GlobalConfig::instance())

// Feature testing macros
#define UINT64COMPUTE_HAS_CPU_FEATURE(feature) \
    (UINT64COMPUTE_HARDWARE().has_feature(uint64_compute::hardware::CPUFeature::feature))

#define UINT64COMPUTE_HAS_PARALLEL_FEATURE(feature) \
    (UINT64COMPUTE_HARDWARE().has_feature(uint64_compute::hardware::ParallelFeature::feature))

// Memory allocation macros
#define UINT64COMPUTE_ALIGNED_ALLOC(size) \
    (uint64_compute::hardware::AlignedAllocator::allocate(size))

#define UINT64COMPUTE_ALIGNED_FREE(ptr) \
    (uint64_compute::hardware::AlignedAllocator::deallocate(ptr))

} // namespace hardware
} // namespace uint64_compute