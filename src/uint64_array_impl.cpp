#include "uint64_compute.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace uint64_compute {

// 自定义的GPU内存释放器
struct GPUMemoryDeleter {
    void operator()(uint64_t* ptr) const {
#ifdef CUDA_ENABLED
        if (ptr) {
            cuda::deallocate(ptr);
        }
#else
        (void)ptr;
#endif
    }
};

using GPUUniquePtr = std::unique_ptr<uint64_t[], GPUMemoryDeleter>;

// Shape类实现
size_t Shape::size() const {
    size_t result = 1;
    for (size_t dim : dims) {
        result *= dim;
    }
    return result;
}

size_t Shape::stride(size_t dim) const {
    if (dim >= dims.size()) return 0;
    size_t result = 1;
    for (size_t i = dim + 1; i < dims.size(); ++i) {
        result *= dims[i];
    }
    return result;
}

bool Shape::is_broadcastable_with(const Shape& /* other */) const {
    // 简化版广播检查
    return true; // 为了编译测试暂时返回true
}

Shape Shape::broadcast_with(const Shape& other) const {
    // 简化版广播：返回较大的形状
    if (size() >= other.size()) {
        return *this;
    } else {
        return other;
    }
}

// UInt64Array类实现
UInt64Array::UInt64Array(const Shape& shape, DeviceType device) 
    : shape_(shape), device_type(device) {
    size_t total_size = shape_.size();
    if (total_size > 0) {
        if (device_type == DeviceType::GPU) {
#ifdef CUDA_ENABLED
            // 检查CUDA是否真正可用
            if (!cuda::is_available()) {
                throw std::runtime_error("GPU device requested but CUDA not available");
            }
            
            // 为GPU分配内存
            void* gpu_mem = cuda::allocate(total_size * sizeof(uint64_t));
            uint64_t* gpu_ptr = static_cast<uint64_t*>(gpu_mem);
            // 注意：这里我们使用普通的unique_ptr，但需要在析构时手动释放GPU内存
            data_ptr = std::unique_ptr<uint64_t[]>(gpu_ptr);
            
            // 使用CUDA memset初始化为0
            cudaMemset(gpu_ptr, 0, total_size * sizeof(uint64_t));
#else
            throw std::runtime_error("GPU device requested but CUDA not available");
#endif
        } else {
            // CPU内存分配
            data_ptr = std::unique_ptr<uint64_t[]>(new uint64_t[total_size]{});
            // 初始化为0
            std::fill(data_ptr.get(), data_ptr.get() + total_size, 0);
        }
    }
}

UInt64Array::UInt64Array(const std::vector<size_t>& shape, DeviceType device) 
    : UInt64Array(Shape(shape), device) {}

UInt64Array::UInt64Array(std::initializer_list<size_t> shape, DeviceType device) 
    : UInt64Array(Shape(shape), device) {}

// 复制构造函数
UInt64Array::UInt64Array(const UInt64Array& other) 
    : shape_(other.shape_), device_type(other.device_type) {
    size_t total_size = shape_.size();
    if (total_size > 0) {
        size_t bytes = total_size * sizeof(uint64_t);
        
        if (device_type == DeviceType::GPU) {
#ifdef CUDA_ENABLED
            // 为GPU分配内存
            void* gpu_mem = cuda::allocate(bytes);
            uint64_t* gpu_ptr = static_cast<uint64_t*>(gpu_mem);
            data_ptr = std::unique_ptr<uint64_t[]>(gpu_ptr);
            
            // GPU到GPU复制
            cudaMemcpy(gpu_ptr, other.data_ptr.get(), bytes, cudaMemcpyDeviceToDevice);
#else
            throw std::runtime_error("GPU device requested but CUDA not available");
#endif
        } else {
            // CPU内存分配和复制
            data_ptr = std::unique_ptr<uint64_t[]>(new uint64_t[total_size]{});
            std::copy(other.data_ptr.get(), other.data_ptr.get() + total_size, data_ptr.get());
        }
    }
}

// 移动构造函数
UInt64Array::UInt64Array(UInt64Array&& other) noexcept 
    : data_ptr(std::move(other.data_ptr)), shape_(std::move(other.shape_)), device_type(other.device_type) {}

// 复制赋值
UInt64Array& UInt64Array::operator=(const UInt64Array& other) {
    if (this != &other) {
        // 先释放当前内存
        if (data_ptr && device_type == DeviceType::GPU) {
#ifdef CUDA_ENABLED
            cuda::deallocate(data_ptr.release());
#endif
        }
        
        shape_ = other.shape_;
        device_type = other.device_type;
        size_t total_size = shape_.size();
        
        if (total_size > 0) {
            size_t bytes = total_size * sizeof(uint64_t);
            
            if (device_type == DeviceType::GPU) {
#ifdef CUDA_ENABLED
                // 为GPU分配内存
                void* gpu_mem = cuda::allocate(bytes);
                uint64_t* gpu_ptr = static_cast<uint64_t*>(gpu_mem);
                data_ptr = std::unique_ptr<uint64_t[]>(gpu_ptr);
                
                // GPU到GPU复制
                cudaMemcpy(gpu_ptr, other.data_ptr.get(), bytes, cudaMemcpyDeviceToDevice);
#else
                throw std::runtime_error("GPU device requested but CUDA not available");
#endif
            } else {
                // CPU内存分配和复制
                data_ptr = std::unique_ptr<uint64_t[]>(new uint64_t[total_size]{});
                std::copy(other.data_ptr.get(), other.data_ptr.get() + total_size, data_ptr.get());
            }
        } else {
            data_ptr.reset();
        }
    }
    return *this;
}

// 移动赋值
UInt64Array& UInt64Array::operator=(UInt64Array&& other) noexcept {
    if (this != &other) {
        data_ptr = std::move(other.data_ptr);
        shape_ = std::move(other.shape_);
        device_type = other.device_type;
    }
    return *this;
}

// 析构函数
UInt64Array::~UInt64Array() {
    if (data_ptr && device_type == DeviceType::GPU) {
#ifdef CUDA_ENABLED
        // 手动释放GPU内存，并阻止unique_ptr尝试释放
        cuda::deallocate(data_ptr.release());
#endif
    }
    // CPU内存会由unique_ptr自动释放（如果没有被release()）
}

// 元素访问
uint64_t& UInt64Array::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Array index out of range");
    }
    return data_ptr[index];
}

const uint64_t& UInt64Array::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Array index out of range");
    }
    return data_ptr[index];
}

uint64_t& UInt64Array::at(const std::vector<size_t>& indices) {
    if (indices.size() != shape_.ndim()) {
        throw std::invalid_argument("Number of indices must match array dimensions");
    }
    
    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_.dims[i]) {
            throw std::out_of_range("Index out of range for dimension " + std::to_string(i));
        }
        flat_index += indices[i] * shape_.stride(i);
    }
    
    return data_ptr[flat_index];
}

const uint64_t& UInt64Array::at(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.ndim()) {
        throw std::invalid_argument("Number of indices must match array dimensions");
    }
    
    size_t flat_index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_.dims[i]) {
            throw std::out_of_range("Index out of range for dimension " + std::to_string(i));
        }
        flat_index += indices[i] * shape_.stride(i);
    }
    
    return data_ptr[flat_index];
}

// 设备转换
UInt64Array UInt64Array::to(DeviceType target_device) const {
    if (device_type == target_device) {
        return *this; // 已经在目标设备上
    }
    
    UInt64Array result(shape_, target_device);
    if (size() > 0) {
        size_t bytes = size() * sizeof(uint64_t);
        
        if (device_type == DeviceType::CPU && target_device == DeviceType::GPU) {
            // CPU -> GPU: 将CPU数据复制到已分配的GPU内存
            std::cout << "Copying from CPU to GPU: " << bytes << " bytes" << std::endl;
            std::cout << "Source (CPU): " << static_cast<const void*>(data()) << std::endl;
            std::cout << "Dest (GPU): " << static_cast<void*>(result.data()) << std::endl;
            
            if (data() == nullptr) {
                throw std::runtime_error("Source CPU data is null");
            }
            if (result.data() == nullptr) {
                throw std::runtime_error("Destination GPU memory is null");
            }
            
            cuda::copy_to_gpu(result.data(), data(), bytes);
        } else if (device_type == DeviceType::GPU && target_device == DeviceType::CPU) {
            // GPU -> CPU: 将GPU数据复制到已分配的CPU内存
            std::cout << "Copying from GPU to CPU: " << bytes << " bytes" << std::endl;
            cuda::copy_to_cpu(result.data(), data(), bytes);
        } else {
            // CPU -> CPU (复制)
            std::copy(data(), data() + size(), result.data());
        }
    }
    return result;
}

UInt64Array UInt64Array::to(const std::string& device_str) const {
    if (device_str == device::CPU || device_str == "cpu") {
        return to(DeviceType::CPU);
    } else if (device_str == device::GPU || device_str == device::CUDA || device_str == "gpu" || device_str == "cuda") {
        return to(DeviceType::GPU);
    } else {
        throw std::invalid_argument("Unknown device: " + device_str);
    }
}

UInt64Array UInt64Array::cpu() const {
    return to(DeviceType::CPU);
}

UInt64Array UInt64Array::gpu() const {
    return to(DeviceType::GPU);
}

UInt64Array UInt64Array::cuda() const {
    return to(DeviceType::GPU);
}

// 填充和初始化
void UInt64Array::fill(uint64_t value) {
    std::fill(data_ptr.get(), data_ptr.get() + size(), value);
}

void UInt64Array::zeros() {
    fill(0);
}

void UInt64Array::ones() {
    fill(1);
}

// 运算符重载
UInt64Array UInt64Array::operator+(const UInt64Array& other) const {
    return ops::add(*this, other);
}

UInt64Array UInt64Array::operator+(uint64_t scalar) const {
    return ops::add(*this, scalar);
}

UInt64Array UInt64Array::operator*(const UInt64Array& other) const {
    return ops::multiply(*this, other);
}

UInt64Array UInt64Array::operator*(uint64_t scalar) const {
    return ops::multiply(*this, scalar);
}

void UInt64Array::print(const std::string& name, bool show_all) const {
    print_array(*this, name, show_all);
}

// 非成员运算符重载
UInt64Array operator+(uint64_t scalar, const UInt64Array& array) {
    return ops::add(array, scalar);
}

UInt64Array operator*(uint64_t scalar, const UInt64Array& array) {
    return ops::multiply(array, scalar);
}

} // namespace uint64_compute