#pragma once

#include <cstddef>
#include <string>
#include <utility>

// CUDA可用性检测和条件编译支持

// CUDA检测宏
#ifdef __CUDACC__
    #define CUDA_COMPILER_AVAILABLE
#endif

#ifdef __CUDA_ARCH__
    #define CUDA_DEVICE_CODE
#endif

// CUDA运行时检测
#if defined(CUDA_COMPILER_AVAILABLE) || defined(CUDA_ENABLED)
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
    #define HAVE_CUDA_RUNTIME
#endif

// 条件编译宏
#ifdef HAVE_CUDA_RUNTIME
    #ifndef CUDA_DISABLED
        // CUDA is available and enabled
    #endif
#else
    #define CUDA_DISABLED
#endif

// CUDA函数修饰符
#ifdef CUDA_ENABLED
    #define CUDA_CALLABLE __host__ __device__
    #define CUDA_KERNEL __global__
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
#else
    #define CUDA_CALLABLE
    #define CUDA_KERNEL
    #define CUDA_HOST
    #define CUDA_DEVICE
#endif

namespace uint64_compute {

// CUDA运行时检测器
class CUDADetector {
private:
    static bool detection_done;
    static bool cuda_available;
    static int device_count;
    static int compute_capability_major;
    static int compute_capability_minor;
    static size_t total_memory;
    
public:
    // 检测CUDA可用性
    static bool detect();
    
    // 查询函数
    static bool is_available();
    static int get_device_count();
    static bool has_sufficient_memory(size_t required_bytes);
    static std::string get_device_info(int device_id = 0);
    static std::pair<int, int> get_compute_capability(int device_id = 0);
    
    // 设备管理
    static bool set_device(int device_id);
    static int get_current_device();
    static void synchronize();
    
    // 内存查询
    static size_t get_free_memory();
    static size_t get_total_memory();
    
    // 性能查询
    static bool supports_unified_memory();
    static bool supports_cooperative_launch();
    static int get_max_threads_per_block();
    static int get_multiprocessor_count();
};

// CUDA错误处理
#ifdef CUDA_ENABLED
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t error = call; \
            if (error != cudaSuccess) { \
                throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
            } \
        } while(0)
        
    #define CUDA_CHECK_KERNEL() \
        do { \
            cudaError_t error = cudaGetLastError(); \
            if (error != cudaSuccess) { \
                throw std::runtime_error("CUDA kernel error: " + std::string(cudaGetErrorString(error))); \
            } \
        } while(0)
#else
    #define CUDA_CHECK(call) do { (void)(call); } while(0)
    #define CUDA_CHECK_KERNEL() do {} while(0)
#endif

} // namespace uint64_compute