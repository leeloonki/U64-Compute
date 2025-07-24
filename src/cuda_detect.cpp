#include "cuda_detect.h"
#include <iostream>
#include <sstream>

namespace uint64_compute {

// 静态成员初始化
bool CUDADetector::detection_done = false;
bool CUDADetector::cuda_available = false;
int CUDADetector::device_count = 0;
int CUDADetector::compute_capability_major = 0;
int CUDADetector::compute_capability_minor = 0;
size_t CUDADetector::total_memory = 0;

bool CUDADetector::detect() {
    if (detection_done) return cuda_available;
    
    detection_done = true;
    
#ifdef CUDA_ENABLED
    try {
        // 检查CUDA运行时
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            cuda_available = false;
            return false;
        }
        
        if (device_count == 0) {
            cuda_available = false;
            return false;
        }
        
        // 获取第一个设备的属性
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, 0);
        if (error != cudaSuccess) {
            cuda_available = false;
            return false;
        }
        
        compute_capability_major = prop.major;
        compute_capability_minor = prop.minor;
        total_memory = prop.totalGlobalMem;
        
        // 检查计算能力（至少需要3.0）
        if (compute_capability_major < 3) {
            cuda_available = false;
            return false;
        }
        
        // 尝试创建CUDA上下文
        error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            cuda_available = false;
            return false;
        }
        
        cuda_available = true;
        return true;
        
    } catch (...) {
        cuda_available = false;
        return false;
    }
#else
    cuda_available = false;
    return false;
#endif
}

bool CUDADetector::is_available() {
    detect();
    return cuda_available;
}

int CUDADetector::get_device_count() {
    detect();
    return device_count;
}

bool CUDADetector::has_sufficient_memory(size_t required_bytes) {
#ifdef CUDA_ENABLED
    if (!is_available()) return false;
    
    size_t free_mem, total_mem;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    return (error == cudaSuccess) && (free_mem >= required_bytes);
#else
    (void)required_bytes;
    return false;
#endif
}

std::string CUDADetector::get_device_info(int device_id) {
#ifdef CUDA_ENABLED
    if (!is_available() || device_id >= device_count) {
        return "CUDA not available or invalid device ID";
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    if (error != cudaSuccess) {
        return "Failed to get device properties";
    }
    
    std::ostringstream oss;
    oss << "Device " << device_id << ": " << prop.name << "\n";
    oss << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    oss << "  Total Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    oss << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    oss << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    oss << "  Max Grid Size: (" << prop.maxGridSize[0] << ", " 
        << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
    oss << "  Warp Size: " << prop.warpSize << "\n";
    oss << "  Unified Memory: " << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
    
    return oss.str();
#else
    (void)device_id;
    return "CUDA not available (not compiled with CUDA support)";
#endif
}

std::pair<int, int> CUDADetector::get_compute_capability(int device_id) {
#ifdef CUDA_ENABLED
    if (!is_available() || device_id >= device_count) {
        return {0, 0};
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    if (error != cudaSuccess) {
        return {0, 0};
    }
    
    return {prop.major, prop.minor};
#else
    (void)device_id;
    return {0, 0};
#endif
}

bool CUDADetector::set_device(int device_id) {
#ifdef CUDA_ENABLED
    if (!is_available() || device_id >= device_count) {
        return false;
    }
    
    cudaError_t error = cudaSetDevice(device_id);
    return error == cudaSuccess;
#else
    (void)device_id;
    return false;
#endif
}

int CUDADetector::get_current_device() {
#ifdef CUDA_ENABLED
    if (!is_available()) return -1;
    
    int device;
    cudaError_t error = cudaGetDevice(&device);
    return (error == cudaSuccess) ? device : -1;
#else
    return -1;
#endif
}

void CUDADetector::synchronize() {
#ifdef CUDA_ENABLED
    if (is_available()) {
        cudaDeviceSynchronize();
    }
#endif
}

size_t CUDADetector::get_free_memory() {
#ifdef CUDA_ENABLED
    if (!is_available()) return 0;
    
    size_t free_mem, total_mem;
    cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
    return (error == cudaSuccess) ? free_mem : 0;
#else
    return 0;
#endif
}

size_t CUDADetector::get_total_memory() {
    detect();
    return total_memory;
}

bool CUDADetector::supports_unified_memory() {
#ifdef CUDA_ENABLED
    if (!is_available()) return false;
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    return (error == cudaSuccess) && prop.unifiedAddressing;
#else
    return false;
#endif
}

bool CUDADetector::supports_cooperative_launch() {
#ifdef CUDA_ENABLED
    if (!is_available()) return false;
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    return (error == cudaSuccess) && prop.cooperativeLaunch;
#else
    return false;
#endif
}

int CUDADetector::get_max_threads_per_block() {
#ifdef CUDA_ENABLED
    if (!is_available()) return 0;
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    return (error == cudaSuccess) ? prop.maxThreadsPerBlock : 0;
#else
    return 0;
#endif
}

int CUDADetector::get_multiprocessor_count() {
#ifdef CUDA_ENABLED
    if (!is_available()) return 0;
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    return (error == cudaSuccess) ? prop.multiProcessorCount : 0;
#else
    return 0;
#endif
}

} // namespace uint64_compute