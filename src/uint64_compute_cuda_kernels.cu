#include "uint64_compute.h"
#include "cuda_detect.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

namespace uint64_compute {

#ifdef CUDA_ENABLED

// CUDA内存管理
void cuda_allocate(void** ptr, size_t bytes) {
    CUDA_CHECK(cudaMalloc(ptr, bytes));
}

void cuda_deallocate(void* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// copy_to_gpu and copy_to_cpu functions are defined in uint64_compute_impl.cpp
// to avoid duplicate definitions

// CUDA内核函数

// 向量加法内核
__global__ void cuda_add_kernel(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        result[i] = a[i] + b[i];
    }
}

// 向量加法内核启动器（供C++调用）
extern "C" void launch_cuda_add_kernel(const uint64_t* a, const uint64_t* b, uint64_t* result, 
                                      size_t count, int gridSize, int blockSize) {
    cuda_add_kernel<<<gridSize, blockSize>>>(a, b, result, count);
    
    // 检查内核启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// 向量乘法内核
__global__ void cuda_multiply_kernel(const uint64_t* a, const uint64_t* b, uint64_t* result, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        result[i] = a[i] * b[i];
    }
}

// 向量乘法内核启动器（供C++调用）
extern "C" void launch_cuda_multiply_kernel(const uint64_t* a, const uint64_t* b, uint64_t* result, 
                                          size_t count, int gridSize, int blockSize) {
    cuda_multiply_kernel<<<gridSize, blockSize>>>(a, b, result, count);
    
    // 检查内核启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// 标量加法内核
__global__ void cuda_add_scalar_kernel(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        result[i] = a[i] + scalar;
    }
}

// 标量乘法内核
__global__ void cuda_multiply_scalar_kernel(const uint64_t* a, uint64_t scalar, uint64_t* result, size_t count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        result[i] = a[i] * scalar;
    }
}

// 矩阵乘法内核（简化版本，带边界检查）
__global__ void cuda_matmul_kernel(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 严格边界检查
    if (row >= M || col >= N || row < 0 || col < 0) {
        return;
    }
    
    uint64_t sum = 0;
    for (int k = 0; k < K; ++k) {
        // 验证数组访问边界
        int a_idx = row * K + k;
        int b_idx = k * N + col;
        int c_idx = row * N + col;
        
        if (a_idx < 0 || a_idx >= M * K || 
            b_idx < 0 || b_idx >= K * N ||
            c_idx < 0 || c_idx >= M * N) {
            return; // 边界外访问，安全退出
        }
        
        sum += A[a_idx] * B[b_idx];
    }
    C[row * N + col] = sum;
}

// 优化的矩阵乘法内核（使用共享内存）
template<int TILE_SIZE>
__global__ void cuda_matmul_tiled_kernel(const uint64_t* A, const uint64_t* B, uint64_t* C,
                                        size_t M, size_t N, size_t K) {
    __shared__ uint64_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ uint64_t tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    uint64_t sum = 0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载tile到共享内存
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Reduction内核
__global__ void cuda_reduce_sum_kernel(const uint64_t* data, uint64_t* result, size_t count) {
    extern __shared__ uint64_t sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (idx < count) ? data[idx] : 0;
    __syncthreads();
    
    // 归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 写回结果
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// 计算最优网格和块大小
dim3 calculate_grid_block_1d(size_t count, int block_size = 256) {
    int grid_size = (count + block_size - 1) / block_size;
    // 限制网格大小避免过多的块
    grid_size = std::min(grid_size, 65535);
    return dim3(grid_size, 1, 1);
}

dim3 calculate_grid_block_2d(size_t rows, size_t cols, int block_size = 16) {
    dim3 block(block_size, block_size);
    dim3 grid((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
    return grid;
}

namespace cuda_ops {

bool is_cuda_available() {
    return CUDADetector::is_available();
}

void print_device_info() {
    if (CUDADetector::is_available()) {
        int device_count = CUDADetector::get_device_count();
        std::cout << "CUDA Devices Found: " << device_count << std::endl;
        for (int i = 0; i < device_count; ++i) {
            std::cout << CUDADetector::get_device_info(i) << std::endl;
        }
    } else {
        std::cout << "CUDA not available" << std::endl;
    }
}

void add(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
    if (!CUDADetector::is_available()) {
        throw std::runtime_error("CUDA not available");
    }
    
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Array sizes must match");
    }
    
    size_t count = a.size();
    size_t bytes = count * sizeof(uint64_t);
    
    // 检查GPU内存
    if (!CUDADetector::has_sufficient_memory(bytes * 3)) {
        throw std::runtime_error("Insufficient GPU memory");
    }
    
    // 分配GPU内存
    uint64_t *d_a, *d_b, *d_result;
    cuda_allocate(reinterpret_cast<void**>(&d_a), bytes);
    cuda_allocate(reinterpret_cast<void**>(&d_b), bytes);
    cuda_allocate(reinterpret_cast<void**>(&d_result), bytes);
    
    try {
        // 复制数据到GPU
        cuda::copy_to_gpu(d_a, a.data(), bytes);
        cuda::copy_to_gpu(d_b, b.data(), bytes);
        
        // 配置内核启动参数
        const int block_size = 256;
        dim3 grid = calculate_grid_block_1d(count, block_size);
        dim3 block(block_size);
        
        // 启动内核
        cuda_add_kernel<<<grid, block>>>(d_a, d_b, d_result, count);
        CUDA_CHECK_KERNEL();
        
        // 复制结果回CPU
        cuda::copy_to_cpu(result.data(), d_result, bytes);
        
        // 同步
        CUDADetector::synchronize();
        
    } catch (...) {
        cuda_deallocate(d_a);
        cuda_deallocate(d_b);
        cuda_deallocate(d_result);
        throw;
    }
    
    // 清理GPU内存
    cuda_deallocate(d_a);
    cuda_deallocate(d_b);
    cuda_deallocate(d_result);
}

void multiply(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
    if (!CUDADetector::is_available()) {
        throw std::runtime_error("CUDA not available");
    }
    
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Array sizes must match");
    }
    
    size_t count = a.size();
    size_t bytes = count * sizeof(uint64_t);
    
    if (!CUDADetector::has_sufficient_memory(bytes * 3)) {
        throw std::runtime_error("Insufficient GPU memory");
    }
    
    uint64_t *d_a, *d_b, *d_result;
    cuda_allocate(reinterpret_cast<void**>(&d_a), bytes);
    cuda_allocate(reinterpret_cast<void**>(&d_b), bytes);
    cuda_allocate(reinterpret_cast<void**>(&d_result), bytes);
    
    try {
        cuda::copy_to_gpu(d_a, a.data(), bytes);
        cuda::copy_to_gpu(d_b, b.data(), bytes);
        
        const int block_size = 256;
        dim3 grid = calculate_grid_block_1d(count, block_size);
        dim3 block(block_size);
        
        cuda_multiply_kernel<<<grid, block>>>(d_a, d_b, d_result, count);
        CUDA_CHECK_KERNEL();
        
        cuda::copy_to_cpu(result.data(), d_result, bytes);
        CUDADetector::synchronize();
        
    } catch (...) {
        cuda_deallocate(d_a);
        cuda_deallocate(d_b);
        cuda_deallocate(d_result);
        throw;
    }
    
    cuda_deallocate(d_a);
    cuda_deallocate(d_b);
    cuda_deallocate(d_result);
}

void matmul(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
    if (!CUDADetector::is_available()) {
        throw std::runtime_error("CUDA not available");
    }
    
    if (a.shape().ndim() != 2 || b.shape().ndim() != 2 || result.shape().ndim() != 2) {
        throw std::invalid_argument("All arrays must be 2D for matrix multiplication");
    }
    
    size_t M = a.shape().dims[0];
    size_t K = a.shape().dims[1];
    size_t K2 = b.shape().dims[0];
    size_t N = b.shape().dims[1];
    
    if (K != K2 || result.shape().dims[0] != M || result.shape().dims[1] != N) {
        throw std::invalid_argument("Matrix dimensions incompatible");
    }
    
    size_t bytes_a = M * K * sizeof(uint64_t);
    size_t bytes_b = K * N * sizeof(uint64_t);
    size_t bytes_c = M * N * sizeof(uint64_t);
    
    if (!CUDADetector::has_sufficient_memory(bytes_a + bytes_b + bytes_c)) {
        throw std::runtime_error("Insufficient GPU memory");
    }
    
    uint64_t *d_a, *d_b, *d_c;
    cuda_allocate(reinterpret_cast<void**>(&d_a), bytes_a);
    cuda_allocate(reinterpret_cast<void**>(&d_b), bytes_b);
    cuda_allocate(reinterpret_cast<void**>(&d_c), bytes_c);
    
    try {
        // 验证输入指针的有效性
        if (!a.data() || !b.data() || !result.data()) {
            throw std::runtime_error("Invalid input array pointers");
        }
        
        // 验证矩阵维度和数组大小
        if (a.size() != M * K || b.size() != K * N || result.size() != M * N) {
            throw std::runtime_error("Array size mismatch with matrix dimensions");
        }
        
        // 验证矩阵维度
        std::cout << "Matrix dimensions: A(" << M << "x" << K << "), B(" << K << "x" << N << "), C(" << M << "x" << N << ")" << std::endl;
        std::cout << "Array sizes: A=" << a.size() << ", B=" << b.size() << ", C=" << result.size() << std::endl;
        std::cout << "Memory sizes: A=" << bytes_a << ", B=" << bytes_b << ", C=" << bytes_c << std::endl;
        
        // 添加调试信息
        std::cout << "Copying matrix A to GPU: " << bytes_a << " bytes" << std::endl;
        std::cout << "Source (CPU): " << a.data() << ", Dest (GPU): " << d_a << std::endl;
        
        // Validate GPU pointers before copying
        if (!d_a || !d_b || !d_c) {
            throw std::runtime_error("Failed to allocate GPU memory");
        }
        
        try {
            CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes_a, cudaMemcpyHostToDevice));
            std::cout << "Matrix A copy completed successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Matrix A copy failed: " << e.what() << std::endl;
            throw;
        }
        
        std::cout << "Copying matrix B to GPU: " << bytes_b << " bytes" << std::endl;
        std::cout << "Source (CPU): " << b.data() << ", Dest (GPU): " << d_b << std::endl;
        
        try {
            CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes_b, cudaMemcpyHostToDevice));
            std::cout << "Matrix B copy completed successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Matrix B copy failed: " << e.what() << std::endl;
            throw;
        }
        
        // 验证数据完整性
        std::cout << "Synchronizing after data transfer..." << std::endl;
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Data transfer completed successfully" << std::endl;
        
        // 暂时只使用简单矩阵乘法kernel，避免tiled版本的潜在bug
        const int block_size = 16;
        dim3 block(block_size, block_size);
        dim3 grid((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
        
        // 确保参数类型正确，避免大数值导致的问题
        if (M > INT_MAX || N > INT_MAX || K > INT_MAX) {
            throw std::runtime_error("Matrix too large for CUDA kernel");
        }
        
        std::cout << "Launching CUDA kernel: grid(" << grid.x << "," << grid.y << "), block(" 
                  << block.x << "," << block.y << "), M=" << M << ", N=" << N << ", K=" << K << std::endl;
        
        // 先用最简单的测试：初始化结果矩阵为0
        std::cout << "First, initializing result matrix to zero..." << std::endl;
        CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Matrix initialized to zero" << std::endl;
        
        // 启动矩阵乘法kernel
        std::cout << "Now launching matmul kernel..." << std::endl;
        cuda_matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, (int)M, (int)N, (int)K);
        
        // 立即检查kernel启动错误
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(launch_err)));
        }
        
        // 等待kernel完成并检查执行错误
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::cout << "CUDA kernel completed successfully" << std::endl;
        std::cout << "Copying result back to CPU: " << bytes_c << " bytes" << std::endl;
        
        CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Result copy completed successfully" << std::endl;
        
    } catch (...) {
        cuda_deallocate(d_a);
        cuda_deallocate(d_b);
        cuda_deallocate(d_c);
        throw;
    }
    
    cuda_deallocate(d_a);
    cuda_deallocate(d_b);
    cuda_deallocate(d_c);
}

// 自适应版本 - 根据数据大小选择CPU或GPU
void add_adaptive(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
    const size_t threshold = 10000; // 阈值
    
    if (CUDADetector::is_available() && a.size() >= threshold) {
        try {
            add(a, b, result);
            return;
        } catch (...) {
            // 如果GPU失败，回退到CPU
        }
    }
    
    // CPU实现
    result = ops::add(a, b);
}

void multiply_adaptive(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
    const size_t threshold = 10000;
    
    if (CUDADetector::is_available() && a.size() >= threshold) {
        try {
            multiply(a, b, result);
            return;
        } catch (...) {
            // 回退到CPU
        }
    }
    
    result = ops::multiply(a, b);
}

void matmul_adaptive(const UInt64Array& a, const UInt64Array& b, UInt64Array& result) {
    const size_t threshold = 64; // 矩阵大小阈值
    
    size_t M = a.shape().dims[0];
    size_t N = b.shape().dims[1];
    
    if (CUDADetector::is_available() && M >= threshold && N >= threshold) {
        try {
            matmul(a, b, result);
            return;
        } catch (...) {
            // 回退到CPU
        }
    }
    
    result = ops::matmul(a, b);
}

} // namespace cuda_ops

#endif // CUDA_ENABLED

} // namespace uint64_compute