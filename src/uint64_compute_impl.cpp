#include "uint64_compute.h"
#include "simd_dispatcher.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// CUDA headers - only include if available
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Note: CUDA kernels disabled - using CPU fallback for CUDA memory operations
#endif

namespace uint64_compute {

// 性能配置实现
void ops::PerformanceConfig::set_num_threads(int num_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #else
    // OpenMP不可用时，忽略设置
    (void)num_threads;
    #endif
}

int ops::PerformanceConfig::get_num_threads() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

bool ops::PerformanceConfig::is_openmp_available() {
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

std::string ops::PerformanceConfig::get_simd_info() {
    return CrossPlatformSIMD::get_feature_string();
}

bool ops::PerformanceConfig::is_simd_available() {
    CrossPlatformSIMD::detect();
    return CrossPlatformSIMD::get_best_simd_width() > 64;
}

// 全局并行化阈值
static size_t global_parallel_threshold = 1024;

void ops::PerformanceConfig::set_parallel_threshold(size_t threshold) {
    global_parallel_threshold = threshold;
}

size_t ops::PerformanceConfig::get_parallel_threshold() {
    return global_parallel_threshold;
}

// 兼容旧接口
unsigned int ops::ParallelConfig::get_hardware_concurrency() {
    return ops::PerformanceConfig::get_num_threads();
}

// NumPy风格工厂函数实现
namespace ops {
namespace np {

UInt64Array arange(uint64_t start, uint64_t stop) {
    if (stop <= start) {
        return UInt64Array({0});
    }
    
    size_t count = stop - start;
    UInt64Array result({count});
    
    for (size_t i = 0; i < count; ++i) {
        result[i] = start + i;
    }
    
    return result;
}

UInt64Array zeros(const std::vector<size_t>& shape) {
    UInt64Array result(shape);
    std::memset(result.data(), 0, result.size() * sizeof(uint64_t));
    return result;
}

UInt64Array ones(const std::vector<size_t>& shape) {
    UInt64Array result(shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = 1;
    }
    return result;
}

UInt64Array full(const std::vector<size_t>& shape, uint64_t value) {
    UInt64Array result(shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = value;
    }
    return result;
}

UInt64Array concatenate(const std::vector<UInt64Array>& arrays, int axis) {
    if (arrays.empty()) {
        return UInt64Array({0});
    }
    
    // 简化版本：仅支持一维数组连接
    if (axis != 0) {
        throw std::invalid_argument("Only axis=0 is supported for concatenation");
    }
    
    // 计算总大小
    size_t total_size = 0;
    for (const auto& arr : arrays) {
        if (arr.shape().ndim() != 1) {
            throw std::invalid_argument("All arrays must be 1D for concatenation");
        }
        total_size += arr.size();
    }
    
    // 创建结果数组
    UInt64Array result({total_size});
    
    // 复制数据
    size_t offset = 0;
    for (const auto& arr : arrays) {
        std::memcpy(result.data() + offset, arr.data(), arr.size() * sizeof(uint64_t));
        offset += arr.size();
    }
    
    return result;
}

} // namespace np
} // namespace ops

// 使用新的SIMD调度器重新实现运算函数
namespace ops {

// 逐元素加法 - 自动处理CPU/GPU设备
UInt64Array add(const UInt64Array& a, const UInt64Array& b) {
    if (!a.shape().is_broadcastable_with(b.shape())) {
        throw std::invalid_argument("Arrays are not broadcastable for addition");
    }
    
    Shape result_shape = a.shape().broadcast_with(b.shape());
    
    // 检查设备类型，确保结果数组在正确的设备上
    DeviceType target_device = DeviceType::CPU;
    if (a.device() == DeviceType::GPU || b.device() == DeviceType::GPU) {
        target_device = DeviceType::GPU;
    }
    
    UInt64Array result(result_shape, target_device);
    
    if (a.shape().dims == b.shape().dims && a.shape().dims == result_shape.dims) {
        // 简单情况：形状完全相同
        if (target_device == DeviceType::GPU) {
#ifdef CUDA_ENABLED
            // GPU上的运算：确保所有数组都在GPU上
            auto gpu_a = (a.device() == DeviceType::GPU) ? a : a.to(DeviceType::GPU);
            auto gpu_b = (b.device() == DeviceType::GPU) ? b : b.to(DeviceType::GPU);
            cuda::add(gpu_a, gpu_b, result);
#else
            throw std::runtime_error("GPU operations not available - compiled without CUDA support");
#endif
        } else {
            // CPU上的运算：使用SIMD优化
            auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
            auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
            CrossPlatformSIMDDispatcher::add(cpu_a.data(), cpu_b.data(), result.data(), result.size());
        }
    } else {
        // 复杂情况：需要广播（暂时回退到CPU）
        auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
        auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
        UInt64Array cpu_result(result_shape, DeviceType::CPU);
        
        #pragma omp parallel for
        for (size_t i = 0; i < cpu_result.size(); i++) {
            cpu_result[i] = cpu_a[i % cpu_a.size()] + cpu_b[i % cpu_b.size()];
        }
        
        if (target_device == DeviceType::GPU) {
            result = cpu_result.to(DeviceType::GPU);
        } else {
            result = cpu_result;
        }
    }
    
    return result;
}

// 标量加法
UInt64Array add(const UInt64Array& a, uint64_t scalar) {
    UInt64Array result(a.shape());
    CrossPlatformSIMDDispatcher::add_scalar(a.data(), scalar, result.data(), result.size());
    return result;
}

// 逐元素乘法 - 使用SIMD调度器
UInt64Array multiply(const UInt64Array& a, const UInt64Array& b) {
    if (!a.shape().is_broadcastable_with(b.shape())) {
        throw std::invalid_argument("Arrays are not broadcastable for multiplication");
    }
    
    Shape result_shape = a.shape().broadcast_with(b.shape());
    
    // 检查设备类型，确保结果数组在正确的设备上
    DeviceType target_device = DeviceType::CPU;
    if (a.device() == DeviceType::GPU || b.device() == DeviceType::GPU) {
        target_device = DeviceType::GPU;
    }
    
    UInt64Array result(result_shape, target_device);
    
    if (a.shape().dims == b.shape().dims && a.shape().dims == result_shape.dims) {
        // 简单情况：形状完全相同
        if (target_device == DeviceType::GPU) {
#ifdef CUDA_ENABLED
            // GPU上的运算：确保所有数组都在GPU上
            auto gpu_a = (a.device() == DeviceType::GPU) ? a : a.to(DeviceType::GPU);
            auto gpu_b = (b.device() == DeviceType::GPU) ? b : b.to(DeviceType::GPU);
            cuda::multiply(gpu_a, gpu_b, result);
#else
            throw std::runtime_error("GPU operations not available - compiled without CUDA support");
#endif
        } else {
            // CPU上的运算：使用SIMD优化
            auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
            auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
            CrossPlatformSIMDDispatcher::multiply(cpu_a.data(), cpu_b.data(), result.data(), result.size());
        }
    } else {
        // 复杂情况：需要广播（暂时回退到CPU）
        auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
        auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
        UInt64Array cpu_result(result_shape, DeviceType::CPU);
        
        #pragma omp parallel for
        for (size_t i = 0; i < cpu_result.size(); i++) {
            cpu_result[i] = cpu_a[i % cpu_a.size()] * cpu_b[i % cpu_b.size()];
        }
        
        if (target_device == DeviceType::GPU) {
            result = cpu_result.to(DeviceType::GPU);
        } else {
            result = cpu_result;
        }
    }
    
    return result;
}

// 标量乘法
UInt64Array multiply(const UInt64Array& a, uint64_t scalar) {
    UInt64Array result(a.shape());
    CrossPlatformSIMDDispatcher::multiply_scalar(a.data(), scalar, result.data(), result.size());
    return result;
}

// 矩阵乘法 - 使用SIMD调度器
UInt64Array matmul(const UInt64Array& a, const UInt64Array& b) {
    if (a.shape().ndim() != 2 || b.shape().ndim() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D arrays");
    }
    
    size_t a_rows = a.shape().dims[0];
    size_t a_cols = a.shape().dims[1];
    size_t b_rows = b.shape().dims[0];
    size_t b_cols = b.shape().dims[1];
    
    if (a_cols != b_rows) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
    }
    
    UInt64Array result({a_rows, b_cols});
    
    // 使用SIMD优化的矩阵乘法 
    // A: M×K (a_rows × a_cols), B: K×N (a_cols × b_cols), C: M×N (a_rows × b_cols)
    // Leading dimensions: lda=K, ldb=N, ldc=N
    CrossPlatformSIMDDispatcher::matmul(
        a.data(), b.data(), result.data(),
        a_rows, b_cols, a_cols,        // M, N, K
        a_cols, b_cols, b_cols         // lda=K, ldb=N, ldc=N
    );
    
    return result;
}

// 复杂整数变换 - 使用SIMD调度器
void complex_uint64_transform_optimized(uint64_t* output, const uint64_t* input, size_t count, int num_rounds) {
    // 使用SIMD调度器进行自适应处理
    AdaptiveBatcher::process_multiply(input, input, output, count); // 简化版本
    
    // 对于复杂变换，可能需要多个步骤
    for (int round = 0; round < num_rounds; ++round) {
        // 这里可以添加更复杂的变换逻辑
        for (size_t i = 0; i < count; ++i) {
            output[i] = (output[i] ^ (output[i] >> 13)) * 0x2545F4914F6CDD1DULL;
            output[i] = (output[i] ^ (output[i] << 17)) * 0x106689D450497F33ULL;
            output[i] = (output[i] ^ (output[i] >> 16));
        }
    }
}

} // namespace ops

// 优化后的打印函数
void print_array(const UInt64Array& arr, const std::string& name, bool show_all) {
    std::cout << name << " (shape: ";
    auto shape = arr.shape();
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        std::cout << shape.dims[i];
        if (i < shape.dims.size() - 1) std::cout << "x";
    }
    std::cout << ", size: " << arr.size() << ", device: ";
    std::cout << (arr.device() == DeviceType::CPU ? "CPU" : "GPU");
    std::cout << "):" << std::endl;
    
    if (arr.device() == DeviceType::GPU) {
        std::cout << "  [GPU data - use .to(CPU) to view contents]" << std::endl;
        return;
    }
    
    size_t show_count = show_all ? arr.size() : std::min(arr.size(), static_cast<size_t>(20));
    
    if (shape.ndim() == 1) {
        // 1D数组
        std::cout << "  [";
        for (size_t i = 0; i < show_count; ++i) {
            std::cout << arr[i];
            if (i < show_count - 1) std::cout << ", ";
        }
        if (!show_all && arr.size() > show_count) {
            std::cout << ", ... (+" << (arr.size() - show_count) << " more)";
        }
        std::cout << "]" << std::endl;
    } else if (shape.ndim() == 2) {
        // 2D数组
        size_t rows = shape.dims[0];
        size_t cols = shape.dims[1];
        size_t show_rows = show_all ? rows : std::min(rows, static_cast<size_t>(10));
        size_t show_cols = show_all ? cols : std::min(cols, static_cast<size_t>(10));
        
        std::cout << "  [";
        for (size_t i = 0; i < show_rows; ++i) {
            if (i > 0) std::cout << "   ";
            std::cout << "[";
            for (size_t j = 0; j < show_cols; ++j) {
                std::cout << arr.at({i, j});
                if (j < show_cols - 1) std::cout << ", ";
            }
            if (!show_all && cols > show_cols) {
                std::cout << ", ... (+" << (cols - show_cols) << " more)";
            }
            std::cout << "]";
            if (i < show_rows - 1) std::cout << "," << std::endl;
        }
        if (!show_all && rows > show_rows) {
            std::cout << "," << std::endl << "   ... (+" << (rows - show_rows) << " more rows)";
        }
        std::cout << "]" << std::endl;
    } else {
        // 高维数组，简化显示
        std::cout << "  [";
        for (size_t i = 0; i < show_count; ++i) {
            std::cout << arr[i];
            if (i < show_count - 1) std::cout << ", ";
            if ((i + 1) % 10 == 0 && i < show_count - 1) std::cout << std::endl << "   ";
        }
        if (!show_all && arr.size() > show_count) {
            std::cout << ", ... (+" << (arr.size() - show_count) << " more)";
        }
        std::cout << "]" << std::endl;
    }
}

// 系统信息显示函数
void print_system_info() {
    std::cout << "=== UInt64Compute System Information ===" << std::endl;
    
    // 初始化SIMD调度器
    CrossPlatformSIMDDispatcher::initialize();
    
    // 显示架构信息
    std::cout << "Architecture: ";
    ArchType arch = CrossPlatformSIMD::get_architecture();
    switch (arch) {
        case ArchType::X86_64: std::cout << "x86-64"; break;
        case ArchType::ARM64:  std::cout << "ARM64"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;
    
    // 显示SIMD特性
    std::cout << "SIMD Features: " << CrossPlatformSIMD::get_feature_string() << std::endl;
    std::cout << "Best SIMD Width: " << CrossPlatformSIMD::get_best_simd_width() << " bits" << std::endl;
    
    // 显示可用实现
    std::cout << CrossPlatformSIMDDispatcher::get_available_implementations() << std::endl;
    
    // 显示选中的实现
    std::cout << "Selected Implementation: " << CrossPlatformSIMDDispatcher::get_selected_implementation() << std::endl;
    
    // 显示性能信息
    auto perf = CrossPlatformSIMDDispatcher::get_performance_info();
    std::cout << "Performance (" << perf.impl_name << "): " 
              << std::fixed << std::setprecision(2) << perf.throughput_gops 
              << " GOPS, " << perf.latency_ns << " ns latency" << std::endl;
    
    // 显示并行信息
    std::cout << "OpenMP Available: " << (ops::PerformanceConfig::is_openmp_available() ? "Yes" : "No") << std::endl;
    if (ops::PerformanceConfig::is_openmp_available()) {
        std::cout << "OpenMP Threads: " << ops::PerformanceConfig::get_num_threads() << std::endl;
        std::cout << "Parallel Threshold: " << ops::PerformanceConfig::get_parallel_threshold() << std::endl;
    } else {
        std::cout << "Running in single-threaded mode" << std::endl;
    }
    
    // 显示CUDA信息
    if (cuda::is_available()) {
        std::cout << "CUDA: Available" << std::endl;
        cuda::print_device_info();
    } else {
        std::cout << "CUDA: Not available" << std::endl;
    }
    
    std::cout << "=========================================" << std::endl;
}

// CUDA操作实现 - 自动回退到CPU如果CUDA不可用
namespace cuda {

bool is_available() {
#ifdef CUDA_ENABLED
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

void print_device_info() {
#ifdef CUDA_ENABLED
    if (!is_available()) {
        std::cout << "CUDA: Not available" << std::endl;
        return;
    }
    
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA Devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  Device " << i << ": " << prop.name 
                  << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    }
#else
    std::cout << "CUDA: Not available (compiled without CUDA support)" << std::endl;
#endif
}

int get_device_count() {
#ifdef CUDA_ENABLED
    if (!is_available()) return 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

void set_device(int device_id) {
#ifdef CUDA_ENABLED
    if (!is_available()) {
        throw std::runtime_error("CUDA not available");
    }
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + std::string(cudaGetErrorString(err)));
    }
#else
    (void)device_id;
    throw std::runtime_error("CUDA operations not available - compiled without CUDA support");
#endif
}

int get_current_device() {
#ifdef CUDA_ENABLED
    if (!is_available()) return -1;
    int device = -1;
    cudaGetDevice(&device);
    return device;
#else
    return -1;
#endif
}

void add(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
#ifdef CUDA_ENABLED
    if (!is_available()) {
        throw std::runtime_error("CUDA operations not available");
    }
    
    // 检查数组维度
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Array sizes must match for CUDA operations");
    }
    
    // 确保所有数组都在GPU上
    if (a.device() != DeviceType::GPU || b.device() != DeviceType::GPU || result.device() != DeviceType::GPU) {
        throw std::invalid_argument("All arrays must be on GPU for CUDA operations");
    }
    
    // 使用CUDA内核进行GPU上的加法运算
    size_t count = a.size();
    
    // 配置CUDA内核启动参数
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    (void)gridSize; // Suppress unused variable warning
    
    // CUDA kernel call (simplified version using device memory copy)
    uint64_t* cpu_a = new uint64_t[count];
    uint64_t* cpu_b = new uint64_t[count];
    uint64_t* cpu_result = new uint64_t[count];
    
    // Copy GPU data to CPU for computation (temporary workaround)
    cudaMemcpy(cpu_a, a.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_b, b.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Perform addition on CPU
    #pragma omp parallel for if(count > 1000)
    for (size_t i = 0; i < count; ++i) {
        cpu_result[i] = cpu_a[i] + cpu_b[i];
    }
    
    // Copy result back to GPU
    cudaMemcpy(result.data(), cpu_result, count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Cleanup
    delete[] cpu_a;
    delete[] cpu_b;
    delete[] cpu_result;
    
    // Synchronize to match CUDA behavior
    cudaDeviceSynchronize();
#else
    (void)a; (void)b; (void)result;
    throw std::runtime_error("CUDA operations not available - compiled without CUDA support");
#endif
}

void multiply(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
#ifdef CUDA_ENABLED
    if (!is_available()) {
        throw std::runtime_error("CUDA operations not available");
    }
    
    // 检查数组维度
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Array sizes must match for CUDA operations");
    }
    
    // 确保所有数组都在GPU上
    if (a.device() != DeviceType::GPU || b.device() != DeviceType::GPU || result.device() != DeviceType::GPU) {
        throw std::invalid_argument("All arrays must be on GPU for CUDA operations");
    }
    
    // 使用CUDA内核进行GPU上的乘法运算
    size_t count = a.size();
    
    // 配置CUDA内核启动参数
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    (void)gridSize; // Suppress unused variable warning
    
    // CUDA kernel call (simplified version using device memory copy)
    uint64_t* cpu_a = new uint64_t[count];
    uint64_t* cpu_b = new uint64_t[count];
    uint64_t* cpu_result = new uint64_t[count];
    
    // Copy GPU data to CPU for computation (temporary workaround)
    cudaMemcpy(cpu_a, a.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_b, b.data(), count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Perform multiplication on CPU
    #pragma omp parallel for if(count > 1000)
    for (size_t i = 0; i < count; ++i) {
        cpu_result[i] = cpu_a[i] * cpu_b[i];
    }
    
    // Copy result back to GPU
    cudaMemcpy(result.data(), cpu_result, count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Cleanup
    delete[] cpu_a;
    delete[] cpu_b;
    delete[] cpu_result;
    
    // Synchronize to match CUDA behavior
    cudaDeviceSynchronize();
#else
    (void)a; (void)b; (void)result;
    throw std::runtime_error("CUDA operations not available - compiled without CUDA support");
#endif
}

void matmul(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
#ifdef CUDA_ENABLED
    if (!is_available()) {
        throw std::runtime_error("CUDA operations not available");
    }
    // 调用真正的CUDA kernel实现
    cuda_ops::matmul_adaptive(a, b, result);
#else
    (void)a; (void)b; (void)result;
    throw std::runtime_error("CUDA operations not available - compiled without CUDA support");
#endif
}

UInt64Array add_with_transfer(const UInt64Array& a, const UInt64Array& b) {
    if (is_available()) {
        // Auto-transfer to GPU, compute, transfer back
        auto gpu_a = a.to(DeviceType::GPU);
        auto gpu_b = b.to(DeviceType::GPU);
        UInt64Array gpu_result(a.shape(), DeviceType::GPU);
        add(gpu_a, gpu_b, gpu_result);
        return gpu_result.to(DeviceType::CPU);
    } else {
        // Fallback to CPU
        return ops::add(a, b);
    }
}

UInt64Array multiply_with_transfer(const UInt64Array& a, const UInt64Array& b) {
    if (is_available()) {
        auto gpu_a = a.to(DeviceType::GPU);
        auto gpu_b = b.to(DeviceType::GPU);
        UInt64Array gpu_result(a.shape(), DeviceType::GPU);
        multiply(gpu_a, gpu_b, gpu_result);
        return gpu_result.to(DeviceType::CPU);
    } else {
        return ops::multiply(a, b);
    }
}

UInt64Array matmul_with_transfer(const UInt64Array& a, const UInt64Array& b) {
    if (!is_available()) {
        throw std::runtime_error("CUDA not available for matmul_with_transfer");
    }
    
    // 直接执行CUDA矩阵乘法，如果失败就让程序知道具体错误
    std::cout << "Executing CUDA matrix multiplication..." << std::endl;
    auto gpu_a = a.to(DeviceType::GPU);
    auto gpu_b = b.to(DeviceType::GPU);
    UInt64Array gpu_result({a.shape().dims[0], b.shape().dims[1]}, DeviceType::GPU);
    matmul(gpu_a, gpu_b, gpu_result);
    return gpu_result.to(DeviceType::CPU);
}

void* allocate(size_t bytes) {
#ifdef CUDA_ENABLED
    if (!is_available()) {
        throw std::runtime_error("CUDA memory allocation not available");
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
#else
    (void)bytes;
    throw std::runtime_error("GPU memory allocation not available - compiled without CUDA support");
#endif
}

void deallocate(void* ptr) {
#ifdef CUDA_ENABLED
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
#else
    (void)ptr;
    throw std::runtime_error("GPU memory deallocation not available - compiled without CUDA support");
#endif
}

void copy_to_gpu(void* dst, const void* src, size_t bytes) {
#ifdef CUDA_ENABLED
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory copy to GPU failed: " + std::string(cudaGetErrorString(err)));
    }
#else
    (void)dst; (void)src; (void)bytes;
    throw std::runtime_error("GPU memory copy not available - compiled without CUDA support");
#endif
}

void copy_to_cpu(void* dst, const void* src, size_t bytes) {
#ifdef CUDA_ENABLED
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory copy to CPU failed: " + std::string(cudaGetErrorString(err)));
    }
#else
    (void)dst; (void)src; (void)bytes;
    throw std::runtime_error("GPU memory copy not available - compiled without CUDA support");
#endif
}

} // namespace cuda

} // namespace uint64_compute