#pragma once

// UInt64Compute: High-Performance mod 2^64 Ring Arithmetic Library
// A NumPy-like library for uint64_t operations with automatic SIMD optimization

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cstdint>

// Configuration and platform detection
#include "simd_detect.h"

namespace uint64_compute {

// 前向声明
class UInt64Array;

// 设备类型枚举
enum class DeviceType {
    CPU,    // CPU设备 (自动使用SIMD + OpenMP优化)
    GPU     // GPU设备 (CUDA加速，仅在可用时)
};

// 设备字符串常量 (PyTorch风格)
namespace device {
    constexpr const char* CPU = "cpu";
    constexpr const char* GPU = "gpu";
    constexpr const char* CUDA = "cuda";  // GPU的别名
}

// 数组形状类
class Shape {
public:
    std::vector<size_t> dims;
    
    Shape() = default;
    Shape(const std::vector<size_t>& dimensions) : dims(dimensions) {}
    Shape(std::initializer_list<size_t> dimensions) : dims(dimensions) {}
    
    size_t ndim() const { return dims.size(); }
    size_t size() const;
    size_t stride(size_t dim) const;
    bool is_broadcastable_with(const Shape& other) const;
    Shape broadcast_with(const Shape& other) const;
    
    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return dims != other.dims; }
};

// 主要的UInt64Array类
class UInt64Array {
private:
    std::unique_ptr<uint64_t[]> data_ptr;
    Shape shape_;
    DeviceType device_type;
    
public:
    // 构造函数
    UInt64Array() : shape_({}), device_type(DeviceType::CPU) {}
    UInt64Array(const Shape& shape, DeviceType device = DeviceType::CPU);
    UInt64Array(const std::vector<size_t>& shape, DeviceType device = DeviceType::CPU);
    UInt64Array(std::initializer_list<size_t> shape, DeviceType device = DeviceType::CPU);
    
    // 复制和移动构造函数
    UInt64Array(const UInt64Array& other);
    UInt64Array(UInt64Array&& other) noexcept;
    UInt64Array& operator=(const UInt64Array& other);
    UInt64Array& operator=(UInt64Array&& other) noexcept;
    
    // 析构函数
    ~UInt64Array();
    
    // 基本属性
    const Shape& shape() const { return shape_; }
    size_t size() const { return shape_.size(); }
    size_t ndim() const { return shape_.ndim(); }
    DeviceType device() const { return device_type; }
    
    // 数据访问
    uint64_t* data() { return data_ptr.get(); }
    const uint64_t* data() const { return data_ptr.get(); }
    
    // 元素访问
    uint64_t& operator[](size_t index);
    const uint64_t& operator[](size_t index) const;
    uint64_t& at(const std::vector<size_t>& indices);
    const uint64_t& at(const std::vector<size_t>& indices) const;
    
    // 设备转换 (PyTorch风格)
    UInt64Array to(DeviceType target_device) const;
    UInt64Array to(const std::string& device_str) const;  // "cpu", "gpu", "cuda"
    UInt64Array cpu() const;  // 等价于 to("cpu")
    UInt64Array gpu() const;  // 等价于 to("gpu") 或 to("cuda")
    UInt64Array cuda() const; // 等价于 gpu()
    
    // 填充和初始化
    void fill(uint64_t value);
    void zeros();
    void ones();
    
    // 基本运算符重载
    UInt64Array operator+(const UInt64Array& other) const;
    UInt64Array operator+(uint64_t scalar) const;
    UInt64Array operator*(const UInt64Array& other) const;
    UInt64Array operator*(uint64_t scalar) const;
    
    // 打印支持
    void print(const std::string& name = "", bool show_all = false) const;
};

// 运算符重载（非成员函数）
UInt64Array operator+(uint64_t scalar, const UInt64Array& array);
UInt64Array operator*(uint64_t scalar, const UInt64Array& array);

// 并行配置类
namespace ops {
    class ParallelConfig {
    public:
        static unsigned int get_hardware_concurrency();
        static void set_num_threads(unsigned int num_threads);
        static unsigned int get_num_threads();
    };
}

// CPU运算函数 (自动透明使用SIMD + OpenMP优化)
namespace ops {
    // 基础运算 - 自动使用最佳CPU优化(SIMD + OpenMP)
    UInt64Array add(const UInt64Array& a, const UInt64Array& b);
    UInt64Array add(const UInt64Array& a, uint64_t scalar);
    UInt64Array multiply(const UInt64Array& a, const UInt64Array& b);
    UInt64Array multiply(const UInt64Array& a, uint64_t scalar);
    UInt64Array matmul(const UInt64Array& a, const UInt64Array& b);
    
    // 复杂运算
    void complex_uint64_transform_optimized(uint64_t* output, const uint64_t* input, size_t count, int num_rounds);
    
    // NumPy风格的工厂函数
    namespace np {
        UInt64Array arange(uint64_t start, uint64_t stop);
        UInt64Array zeros(const std::vector<size_t>& shape);
        UInt64Array ones(const std::vector<size_t>& shape);
        UInt64Array full(const std::vector<size_t>& shape, uint64_t value);
        UInt64Array concatenate(const std::vector<UInt64Array>& arrays, int axis = 0);
    }
    
    // 性能配置 (用户可选调整)
    class PerformanceConfig {
    public:
        // OpenMP控制
        static void set_num_threads(int num_threads);
        static int get_num_threads();
        static bool is_openmp_available();
        
        // SIMD信息
        static std::string get_simd_info();
        static bool is_simd_available();
        
        // 性能调优
        static void set_parallel_threshold(size_t threshold);
        static size_t get_parallel_threshold();
    };
}

// CUDA操作命名空间 (可选GPU加速 - 类似numpy.cuda)
namespace cuda {
    // CUDA可用性检查
    bool is_available();           // 检查CUDA是否可用
    void print_device_info();      // 打印GPU信息
    int get_device_count();        // 获取GPU数量
    void set_device(int device_id); // 设置当前GPU
    int get_current_device();      // 获取当前GPU
    
    // 基础CUDA操作 - 要求输入数据已在GPU上  
    void add(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    void multiply(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    void matmul(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    
    // 便捷操作 - 自动处理CPU/GPU数据传输
    UInt64Array add_with_transfer(const UInt64Array& a, const UInt64Array& b);
    UInt64Array multiply_with_transfer(const UInt64Array& a, const UInt64Array& b);
    UInt64Array matmul_with_transfer(const UInt64Array& a, const UInt64Array& b);
    
    // GPU内存管理函数
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void copy_to_gpu(void* dst, const void* src, size_t bytes);
    void copy_to_cpu(void* dst, const void* src, size_t bytes);
}

// CUDA 操作命名空间 (内部实现)
#ifdef CUDA_ENABLED
namespace cuda_ops {
    bool is_cuda_available();
    void add_adaptive(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    void multiply_adaptive(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    void matmul_adaptive(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
}
#endif

// 系统信息和辅助函数
void print_system_info();
void print_array(const UInt64Array& arr, const std::string& name = "", bool show_all = false);

} // namespace uint64_compute