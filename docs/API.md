# U64-Compute API Documentation

Complete documentation for the U64-Compute high-performance uint64_t arithmetic library.

---


### 🎯 Quick Start Examples

#### Basic Usage

```cpp
#include "uint64_compute.h"
using namespace uint64_compute;

int main() {
    // Create arrays (NumPy-style)
    auto a = ops::np::arange(1, 11);  // [1, 2, 3, ..., 10]  
    auto b = ops::np::ones({10});     // [1, 1, 1, ..., 1]
    
    // Basic operations with automatic SIMD + OpenMP optimization
    auto sum = a + b;                 // Element-wise addition
    auto scaled = a * 2;              // Scalar multiplication
    
    // Advanced operations
    auto result = ops::add(a, b);     // Explicit vectorized addition
    
    return 0;
}
```

#### Device Management (PyTorch-style)

```cpp
#include "uint64_compute.h"
using namespace uint64_compute;

int main() {
    // CPU operations (default)
    auto cpu_array = ops::np::zeros({1000000});
    
    // GPU operations (if CUDA available)
    if (cuda::is_available()) {
        auto gpu_array = cpu_array.cuda();    // Transfer to GPU
        auto gpu_result = gpu_array * 2;      // GPU computation
        auto result = gpu_result.cpu();       // Transfer back to CPU
    }
    
    // Device-agnostic operations
    auto array = ops::np::ones({1000});
    auto gpu_array = array.to("cuda");       // String-based device selection
    auto cpu_array = array.to("cpu");
    
    return 0;
}
```

#### Performance Configuration

```cpp
#include "uint64_compute.h"
using namespace uint64_compute;

int main() {
    // Check system capabilities
    print_system_info();  // Print complete system summary
    std::cout << "SIMD support: " << ops::PerformanceConfig::get_simd_info() << std::endl;
    std::cout << "OpenMP threads: " << ops::PerformanceConfig::get_num_threads() << std::endl;
    std::cout << "CUDA available: " << (cuda::is_available() ? "Yes" : "No") << std::endl;
    
    // Configure performance settings
    ops::PerformanceConfig::set_num_threads(8);
    ops::PerformanceConfig::set_parallel_threshold(1024);
    
    // Large array example
    auto large_array = ops::np::arange(1, 1000001);  // 1M elements
    auto large_b = ops::np::ones({1000000});
    
    // CPU computation (automatic SIMD + OpenMP)
    auto cpu_result = large_array + large_b;
    
    // GPU computation (if available)
    if (cuda::is_available()) {
        auto gpu_result = cuda::add_with_transfer(large_array, large_b);
    }
    
    return 0;
}
```

#### Matrix Operations

```cpp
#include "uint64_compute.h"
using namespace uint64_compute;

int main() {
    // Create matrices (2D arrays)
    auto mat_a = ops::np::full({3, 4}, 2);    // 3x4 matrix filled with 2
    auto mat_b = ops::np::full({4, 3}, 3);    // 4x3 matrix filled with 3
    
    // Matrix multiplication
    auto result = ops::matmul(mat_a, mat_b);  // 3x3 result matrix
    
    // Print results
    mat_a.print("Matrix A");
    mat_b.print("Matrix B");
    result.print("A × B");
    
    return 0;
}
```

---

## Part II: API Reference

### 📦 Core Classes and Types

#### DeviceType Enum

```cpp
enum class DeviceType {
    CPU,    // CPU device (automatic SIMD + OpenMP optimization)
    GPU     // GPU device (CUDA acceleration, if available)
};
```

#### Shape Class

```cpp
class Shape {
public:
    std::vector<size_t> dims;
    
    Shape(const std::vector<size_t>& dimensions);
    Shape(std::initializer_list<size_t> dimensions);
    
    size_t ndim() const;                              // Number of dimensions
    size_t size() const;                              // Total number of elements
    size_t stride(size_t dim) const;                  // Stride for dimension
    bool is_broadcastable_with(const Shape& other) const;
    Shape broadcast_with(const Shape& other) const;
};
```

#### UInt64Array Class

```cpp
class UInt64Array {
public:
    // Constructors
    UInt64Array(const Shape& shape, DeviceType device = DeviceType::CPU);
    UInt64Array(const std::vector<size_t>& shape, DeviceType device = DeviceType::CPU);
    UInt64Array(std::initializer_list<size_t> shape, DeviceType device = DeviceType::CPU);
    
    // Basic Properties
    const Shape& shape() const;              // Get array dimensions
    size_t size() const;                     // Total number of elements  
    size_t ndim() const;                     // Number of dimensions
    DeviceType device() const;               // Current device (CPU/GPU)
    
    // Data Access
    uint64_t* data();                        // Raw data pointer
    const uint64_t* data() const;
    uint64_t& operator[](size_t index);      // Linear indexing
    const uint64_t& operator[](size_t index) const;
    uint64_t& at(const std::vector<size_t>& indices);  // Multi-dim indexing
    const uint64_t& at(const std::vector<size_t>& indices) const;
    
    // Device Management (PyTorch-style)
    UInt64Array to(DeviceType target_device) const;
    UInt64Array to(const std::string& device_str) const;  // "cpu", "gpu", "cuda"
    UInt64Array cpu() const;                 // Transfer to CPU
    UInt64Array gpu() const;                 // Transfer to GPU
    UInt64Array cuda() const;                // Transfer to GPU (alias)
    
    // Initialization
    void fill(uint64_t value);               // Fill with specific value
    void zeros();                            // Fill with zeros
    void ones();                             // Fill with ones
    
    // Element-wise Operations (automatic SIMD + OpenMP optimization)
    UInt64Array operator+(const UInt64Array& other) const;
    UInt64Array operator+(uint64_t scalar) const;
    UInt64Array operator*(const UInt64Array& other) const;
    UInt64Array operator*(uint64_t scalar) const;
    
    // Utility
    void print(const std::string& name = "", bool show_all = false) const;
};

// Non-member operators
UInt64Array operator+(uint64_t scalar, const UInt64Array& array);
UInt64Array operator*(uint64_t scalar, const UInt64Array& array);
```

### 🏭 Factory Functions (NumPy-style)

```cpp
namespace ops::np {
    // Array Creation
    UInt64Array arange(uint64_t start, uint64_t stop);
    UInt64Array zeros(const std::vector<size_t>& shape);
    UInt64Array ones(const std::vector<size_t>& shape);  
    UInt64Array full(const std::vector<size_t>& shape, uint64_t value);
    UInt64Array concatenate(const std::vector<UInt64Array>& arrays, int axis = 0);  // 1D arrays only
}
```

### 🔧 Mathematical Operations

```cpp
namespace ops {
    // Core Operations (automatically optimized)
    UInt64Array add(const UInt64Array& a, const UInt64Array& b);
    UInt64Array add(const UInt64Array& a, uint64_t scalar);
    UInt64Array multiply(const UInt64Array& a, const UInt64Array& b);
    UInt64Array multiply(const UInt64Array& a, uint64_t scalar);
    
    // Matrix Operations (basic implementation)
    UInt64Array matmul(const UInt64Array& a, const UInt64Array& b);
}
```

### ⚙️ Performance Configuration

```cpp
namespace ops {
    class PerformanceConfig {
    public:
        // OpenMP Control
        static void set_num_threads(int num_threads);
        static int get_num_threads();
        static bool is_openmp_available();
        
        // SIMD Information
        static std::string get_simd_info();
        static bool is_simd_available();
        
        // Performance Tuning
        static void set_parallel_threshold(size_t threshold);
        static size_t get_parallel_threshold();
    };
}
```

### 🎮 CUDA Support

```cpp
namespace cuda {
    // CUDA Availability
    bool is_available();                 // Check if CUDA is available
    int get_device_count();              // Number of CUDA devices
    void set_device(int device_id);      // Select active CUDA device
    int get_current_device();            // Get current active device
    
    // High-level Operations with Automatic CPU/GPU Transfer
    UInt64Array add_with_transfer(const UInt64Array& a, const UInt64Array& b);
    UInt64Array multiply_with_transfer(const UInt64Array& a, const UInt64Array& b);
    
    // Low-level GPU Operations (requires arrays already on GPU)
    void add(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    void multiply(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    void matmul(const UInt64Array& a, const UInt64Array& b, UInt64Array& result);
    
    // GPU Memory Management
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
    void copy_to_gpu(void* dst, const void* src, size_t bytes);
    void copy_to_cpu(void* dst, const void* src, size_t bytes);
}
```

### 🔍 System Information

```cpp
// Global utility functions
void print_system_info();               // Print complete system summary
void print_array(const UInt64Array& arr, const std::string& name = "", bool show_all = false);
```

### 📝 Device String Constants

```cpp
namespace device {
    constexpr const char* CPU = "cpu";
    constexpr const char* GPU = "gpu";
    constexpr const char* CUDA = "cuda";  // GPU alias
}
```

---

## 📋 Usage Notes

### Performance Tips
- Arrays automatically use SIMD + OpenMP optimization on CPU
- Use `.cuda()` or `.to("gpu")` for GPU acceleration when available
- Matrix operations benefit significantly from GPU acceleration
- Set appropriate parallel thresholds for your workload size

### Memory Management
- Arrays manage their own memory automatically
- Device transfers create new arrays (no in-place conversion)
- GPU memory is automatically freed when arrays go out of scope

### Compatibility
- Requires C++17 compiler
- OpenMP support is optional but recommended
- CUDA support is optional and auto-detected

### Error Handling
- Out-of-bounds access throws `std::out_of_range`
- Invalid operations throw `std::invalid_argument`  
- CUDA errors throw `std::runtime_error`

---

This completes the U64-Compute API documentation. For more examples, see `functional_test.cpp` in the project root.