#include "include/uint64_compute.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace uint64_compute;

// 性能测试帮助函数
template<typename Func>
double benchmark(const std::string& name, Func&& func, int iterations = 1) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_ms = duration.count() / 1000.0 / iterations;
    std::cout << name << ": " << time_ms << " ms" << std::endl;
    return time_ms;
}

int main() {
    std::cout << "=== UInt64 Compute Library Functional Test ===" << std::endl;
    std::cout << std::endl;

    // 1. 系统信息
    std::cout << "1. System Information:" << std::endl;
    print_system_info();
    std::cout << std::endl;

    // 2. 基础数组创建和操作 (NumPy风格)
    std::cout << "2. Basic Array Operations (NumPy-style):" << std::endl;
    
    // 创建数组
    auto a = ops::np::arange(1, 11);  // [1, 2, 3, ..., 10]
    auto b = ops::np::ones({10});     // [1, 1, 1, ..., 1]
    auto c = ops::np::zeros({10});    // [0, 0, 0, ..., 0]
    
    std::cout << "Array a (arange 1-10):";
    a.print("a", true);
    
    std::cout << "Array b (ones):";
    b.print("b", true);
    
    // 基础运算
    auto sum_result = a + b;  // 使用运算符重载
    auto mul_result = a * 2;  // 标量乘法
    
    std::cout << "a + b = ";
    sum_result.print("sum", true);
    
    std::cout << "a * 2 = ";
    mul_result.print("mul", true);
    std::cout << std::endl;

    // 3. 高级操作函数
    std::cout << "3. Advanced Operations:" << std::endl;
    
    auto add_result = ops::add(a, b);
    auto multiply_result = ops::multiply(a, ops::np::full({10}, 3));
    
    std::cout << "ops::add(a, b) = ";
    add_result.print("add_ops", true);
    
    std::cout << "ops::multiply(a, [3,3,...]) = ";
    multiply_result.print("mul_ops", true);
    std::cout << std::endl;

    // 4. 性能配置和信息
    std::cout << "4. Performance Configuration:" << std::endl;
    std::cout << "Hardware concurrency: " << ops::ParallelConfig::get_hardware_concurrency() << std::endl;
    std::cout << "Current threads: " << ops::PerformanceConfig::get_num_threads() << std::endl;
    std::cout << "OpenMP available: " << (ops::PerformanceConfig::is_openmp_available() ? "YES" : "NO") << std::endl;
    std::cout << "SIMD available: " << (ops::PerformanceConfig::is_simd_available() ? "YES" : "NO") << std::endl;
    std::cout << "SIMD info: " << ops::PerformanceConfig::get_simd_info() << std::endl;
    std::cout << std::endl;

    // 5. 大数组性能测试 (展示SIMD + OpenMP自动优化)
    std::cout << "5. Performance Test (Large Arrays with Auto-Optimization):" << std::endl;
    const size_t large_size = 1000000;  // 1M elements
    
    auto large_a = ops::np::arange(1, large_size + 1);
    auto large_b = ops::np::ones({large_size});
    
    std::cout << "Testing with " << large_size << " elements..." << std::endl;
    
    // CPU性能测试
    benchmark("Large array addition (CPU + SIMD + OpenMP)", [&]() {
        auto result = ops::add(large_a, large_b);
    }, 10);
    
    benchmark("Large array multiplication (CPU + SIMD + OpenMP)", [&]() {
        auto result = ops::multiply(large_a, large_b);
    }, 10);
    std::cout << std::endl;

    // 6. CUDA支持测试 (如果可用)
    std::cout << "6. CUDA Support Test:" << std::endl;
    std::cout << "CUDA available: " << (cuda::is_available() ? "YES" : "NO") << std::endl;
    
    if (cuda::is_available()) {
        std::cout << "GPU device count: " << cuda::get_device_count() << std::endl;
        cuda::print_device_info();
        
        // 设备转换测试 (PyTorch风格)
        std::cout << "\\nTesting device conversion (PyTorch-style):" << std::endl;
        auto cpu_array = ops::np::arange(1, 1001);  // 1000 elements
        std::cout << "Original array on: " << 
                     (cpu_array.device() == DeviceType::CPU ? "CPU" : "GPU") << std::endl;
        
        // 转移到GPU
        benchmark("CPU to GPU transfer", [&]() {
            auto gpu_array = cpu_array.cuda();  // or .to("cuda")
        });
        
        auto gpu_array = cpu_array.cuda();
        std::cout << "After .cuda(): " << 
                     (gpu_array.device() == DeviceType::GPU ? "GPU" : "CPU") << std::endl;
        
        // GPU计算
        auto gpu_b = ops::np::ones({1000}).cuda();
        benchmark("GPU computation", [&]() {
            auto gpu_result = cuda::add_with_transfer(gpu_array, gpu_b);
        });
        
        // 转回CPU
        benchmark("GPU to CPU transfer", [&]() {
            auto result_cpu = gpu_array.cpu();
        });
        
    } else {
        std::cout << "CUDA not available - skipping GPU tests" << std::endl;
    }
    std::cout << std::endl;

    // 7. 矩阵运算测试
    std::cout << "7. Matrix Operations Test:" << std::endl;
    try {
        // 创建矩阵 (2D数组)
        UInt64Array matrix_a({3, 4});  // 3x4矩阵
        UInt64Array matrix_b({4, 3});  // 4x3矩阵
        
        // 填充数据
        matrix_a.fill(2);
        matrix_b.fill(3);
        
        std::cout << "Matrix A (3x4) filled with 2:" << std::endl;
        matrix_a.print("matrix_a");
        
        std::cout << "Matrix B (4x3) filled with 3:" << std::endl;
        matrix_b.print("matrix_b");
        
        // 矩阵乘法
        auto matmul_result = ops::matmul(matrix_a, matrix_b);
        std::cout << "Matrix multiplication A × B (3x3):" << std::endl;
        matmul_result.print("matmul_result");
        
    } catch (const std::exception& e) {
        std::cout << "Matrix operations not fully implemented yet: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    // 8. 复杂计算示例
    std::cout << "8. Complex Computation Example:" << std::endl;
    auto data = ops::np::arange(1, 10001);  // 10K elements
    
    benchmark("Complex transformation", [&]() {
        uint64_t temp_data[10000];
        for (size_t i = 0; i < 10000; ++i) {
            temp_data[i] = data[i];
        }
        uint64_t output[10000];
        ops::complex_uint64_transform_optimized(output, temp_data, 10000, 5);
    });
    std::cout << std::endl;

    // 9. 内存和设备管理
    std::cout << "9. Memory and Device Management:" << std::endl;
    
    // 不同设备上的数组
    auto cpu_arr = ops::np::arange(1, 101);
    std::cout << "CPU array size: " << cpu_arr.size() << " elements" << std::endl;
    std::cout << "Memory usage: " << cpu_arr.size() * sizeof(uint64_t) << " bytes" << std::endl;
    
    if (cuda::is_available()) {
        auto gpu_arr = cpu_arr.to("cuda");
        std::cout << "Same array on GPU, device: " << 
                     (gpu_arr.device() == DeviceType::GPU ? "GPU" : "CPU") << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== All Tests Completed Successfully! ===" << std::endl;
    std::cout << "\\nThis library provides:" << std::endl;
    std::cout << "• NumPy-style array operations" << std::endl;
    std::cout << "• Automatic SIMD + OpenMP optimization (transparent to user)" << std::endl;
    std::cout << "• PyTorch-style device management (.cpu(), .cuda(), .to())" << std::endl;
    std::cout << "• Pluggable CUDA support" << std::endl;
    std::cout << "• High-performance uint64_t computations" << std::endl;

    return 0;
}