#include "include/uint64_compute.h"
#include "include/hardware_detector.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>

using namespace uint64_compute;
using namespace std::chrono;

// 简单的基准测试辅助函数
template<typename Func>
double benchmark_operation(const std::string& name, Func&& func, int iterations = 5) {
    std::cout << "Testing " << name << "..." << std::flush;
    
    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max();
    
    for (int i = 0; i < iterations; ++i) {
        auto start = high_resolution_clock::now();
        try {
            func();
        } catch (const std::exception& e) {
            std::cout << " FAILED (" << e.what() << ")" << std::endl;
            throw;
        }
        auto end = high_resolution_clock::now();
        
        double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        total_time += time_ms;
        min_time = std::min(min_time, time_ms);
    }
    
    double avg_time = total_time / iterations;
    std::cout << " Done (" << avg_time << " ms)" << std::endl;
    return avg_time;
}

int main() {
    std::cout << "=== U64-Compute Performance Benchmark ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // 显示系统信息
    std::cout << "\n=== System Information ===" << std::endl;
    auto& detector = hardware::HardwareDetector::instance();
    auto system_info = detector.detect_system();
    
    std::cout << "CPU: " << system_info.cpu.brand << std::endl;
    std::cout << "Architecture: " << system_info.cpu.architecture << std::endl;
    std::cout << "Physical Cores: " << system_info.cpu.cores_physical << std::endl;
    std::cout << "SIMD: " << ops::PerformanceConfig::get_simd_info() << std::endl;
    std::cout << "OpenMP Threads: " << ops::PerformanceConfig::get_num_threads() << std::endl;
    
    // 测试不同大小的数组
    std::vector<size_t> test_sizes = {1000, 10000, 100000, 1000000};
    
    std::cout << "\n=== Performance Benchmarks ===" << std::endl;
    std::cout << "| Operation | Size | Time (ms) | Throughput (M ops/sec) |" << std::endl;
    std::cout << "|-----------|------|-----------|-------------------------|" << std::endl;
    
    for (size_t size : test_sizes) {
        // 创建测试数组
        auto a = ops::np::arange(1, size + 1);
        auto b = ops::np::ones({size});
        
        // 测试加法
        double add_time = benchmark_operation("Addition " + std::to_string(size), [&]() {
            volatile auto result = a + b;
        });
        double add_throughput = size / (add_time * 1000.0);  // M ops/sec
        
        // 测试乘法
        double mul_time = benchmark_operation("Multiplication " + std::to_string(size), [&]() {
            volatile auto result = a * 2;
        });
        double mul_throughput = size / (mul_time * 1000.0);  // M ops/sec
        
        // 输出结果
        std::cout << "| Addition | " << std::setw(6) << size << " | " 
                  << std::setw(8) << add_time << " | " 
                  << std::setw(21) << add_throughput << " |" << std::endl;
        std::cout << "| Multiplication | " << std::setw(6) << size << " | " 
                  << std::setw(8) << mul_time << " | " 
                  << std::setw(21) << mul_throughput << " |" << std::endl;
    }
    
    // 矩阵乘法测试
    std::cout << "\n=== Matrix Multiplication Benchmark (CPU) ===" << std::endl;
    std::vector<size_t> matrix_sizes = {256, 512, 1024, 2048};
    
    std::cout << "| Matrix Size | Time (ms) | GFLOPS |" << std::endl;
    std::cout << "|-------------|-----------|--------|" << std::endl;
    
    // 存储CPU测试结果，供后续GPU对比使用
    std::vector<std::pair<size_t, std::pair<double, double>>> cpu_results;
    
    for (size_t n : matrix_sizes) {
        auto mat_a = ops::np::full({n, n}, 2);
        auto mat_b = ops::np::full({n, n}, 3);
        
        double matmul_time = benchmark_operation("CPU MatMul " + std::to_string(n) + "x" + std::to_string(n), [&]() {
            volatile auto result = ops::matmul(mat_a, mat_b);
        });
        
        // 计算GFLOPS (2*n^3 operations for matrix multiplication)
        double operations = 2.0 * n * n * n;
        double gflops = operations / (matmul_time * 1000000.0);  // GFLOPS
        
        cpu_results.push_back({n, {matmul_time, gflops}});
        
        std::cout << "| " << std::setw(11) << (std::to_string(n) + "x" + std::to_string(n)) 
                  << " | " << std::setw(9) << matmul_time 
                  << " | " << std::setw(6) << gflops << " |" << std::endl;
    }
    
    // CUDA测试 (如果可用)
    if (cuda::is_available()) {
        std::cout << "\n=== CUDA Performance ===" << std::endl;
        int device_count = cuda::get_device_count();
        std::cout << "CUDA Devices: " << device_count << std::endl;
        
        // 测试每个GPU设备的加法性能
        for (int device = 0; device < device_count; ++device) {
            std::cout << "\n--- GPU " << device << " Addition Performance ---" << std::endl;
            cuda::set_device(device);
            
            auto cpu_array = ops::np::arange(1, 1000001);  // 1M elements
            auto cpu_ones = ops::np::ones({1000000});
            
            double cuda_time = benchmark_operation("CUDA Addition (1M elements) on GPU " + std::to_string(device), [&]() {
                volatile auto result = cuda::add_with_transfer(cpu_array, cpu_ones);
            });
            double cuda_throughput = 1000000 / (cuda_time * 1000.0);
            
            std::cout << "GPU " << device << " Addition: " << cuda_time << " ms (" 
                      << cuda_throughput << " M ops/sec)" << std::endl;
        }
        
        // CUDA矩阵乘法性能测试
        std::cout << "\n=== CUDA Matrix Multiplication Performance ===" << std::endl;
        cuda::set_device(0);  // 使用第一个GPU
        
        std::cout << "| Matrix Size | GPU Time (ms) | GPU GFLOPS | CPU Time (ms) | CPU GFLOPS | Speedup |" << std::endl;
        std::cout << "|-------------|---------------|------------|---------------|------------|---------|" << std::endl;
        
        // 测试所有矩阵大小，包括2048（如果GPU内存足够）
        for (const auto& cpu_result : cpu_results) {
            size_t n = cpu_result.first;
            double cpu_time = cpu_result.second.first;
            double cpu_gflops = cpu_result.second.second;
            
            try {
                auto mat_a = ops::np::full({n, n}, 2);
                auto mat_b = ops::np::full({n, n}, 3);
                
                // GPU矩阵乘法测试
                double gpu_time = benchmark_operation("CUDA MatMul " + std::to_string(n) + "x" + std::to_string(n), [&]() {
                    volatile auto result = cuda::matmul_with_transfer(mat_a, mat_b);
                });
                
                // 计算GPU GFLOPS
                double operations = 2.0 * n * n * n;
                double gpu_gflops = operations / (gpu_time * 1000000.0);
                double speedup = cpu_time / gpu_time;
                
                std::cout << "| " << std::setw(11) << (std::to_string(n) + "x" + std::to_string(n)) 
                          << " | " << std::setw(13) << gpu_time 
                          << " | " << std::setw(10) << gpu_gflops 
                          << " | " << std::setw(13) << cpu_time 
                          << " | " << std::setw(10) << cpu_gflops 
                          << " | " << std::setw(7) << speedup << " |" << std::endl;
                          
            } catch (const std::exception& e) {
                std::cout << "| " << std::setw(11) << (std::to_string(n) + "x" + std::to_string(n)) 
                          << " | " << std::setw(13) << "FAILED" 
                          << " | " << std::setw(10) << "N/A" 
                          << " | " << std::setw(13) << cpu_time 
                          << " | " << std::setw(10) << cpu_gflops 
                          << " | " << std::setw(7) << "N/A" << " |" << std::endl;
                std::cout << "    Error for " << n << "x" << n << ": " << e.what() << std::endl;
            }
        }
        
        // 测试更大的数组 (10M elements) 只在第一个GPU上
        std::cout << "\n--- Large Array Test (GPU 0) ---" << std::endl;
        cuda::set_device(0);
        auto large_cpu_array = ops::np::arange(1, 10000001);  // 10M elements
        auto large_cpu_ones = ops::np::ones({10000000});
        
        double large_cuda_time = benchmark_operation("CUDA Addition (10M elements)", [&]() {
            volatile auto result = cuda::add_with_transfer(large_cpu_array, large_cpu_ones);
        });
        double large_cuda_throughput = 10000000 / (large_cuda_time * 1000.0);
        
        std::cout << "Large Array (10M): " << large_cuda_time << " ms (" 
                  << large_cuda_throughput << " M ops/sec)" << std::endl;
        
    } else {
        std::cout << "\n=== CUDA Performance ===" << std::endl;
        std::cout << "CUDA: Not available" << std::endl;
    }
    
    std::cout << "\n=== Benchmark Completed ===" << std::endl;
    std::cout << "Copy these results to update README.md benchmarks section." << std::endl;
    
    return 0;
}