#include "array_utils.h"
#include <cstring>
#include <iostream>

namespace u64comp {
ops op;
namespace np {
size_t Shape::size() const {
    size_t result = 1;
    for (size_t dim : dims) {
        result *= dim;
    }
    return result;
}

size_t Shape::stride(size_t dim) const {
    if (dim >= dims.size())
        return 0;
    size_t result = 1;
    for (size_t i = dim + 1; i < dims.size(); ++i) {
        result *= dims[i];
    }
    return result;
}

U64Array::U64Array(const Shape &shape, device::DeviceType device) : shape_(shape), device_type(device) {
    size_t total_size = shape_.size();
    if (total_size > 0) {
        if (device_type == device::DeviceType::GPU) {
            throw std::runtime_error("GPU device requested but CUDA not available");
        } else {
            data_ptr = std::unique_ptr<uint64_t[]>(new uint64_t[total_size]{});
            std::fill(data_ptr.get(), data_ptr.get() + total_size, 0);
        }
    }
}

U64Array::U64Array(const std::vector<size_t> &shape, device::DeviceType device) : U64Array(Shape(shape), device) {}

U64Array::U64Array(std::initializer_list<size_t> shape, device::DeviceType device) : U64Array(Shape(shape), device) {}

U64Array::U64Array(const U64Array &other) : shape_(other.shape_), device_type(other.device_type) {
    size_t total_size = shape_.size();
    if (total_size > 0) {
        size_t bytes = total_size * sizeof(uint64_t);

        if (device_type == device::DeviceType::GPU) {
            throw std::runtime_error("GPU device requested but CUDA not available");
        } else {
            data_ptr = std::unique_ptr<uint64_t[]>(new uint64_t[total_size]{});
            std::copy(other.data_ptr.get(), other.data_ptr.get() + total_size, data_ptr.get());
        }
    }
}

U64Array::U64Array(U64Array &&other) noexcept
    : data_ptr(std::move(other.data_ptr)), shape_(std::move(other.shape_)), device_type(other.device_type) {}

U64Array &U64Array::operator=(const U64Array &other) {
    if (this != &other) {
        if (data_ptr && device_type == device::DeviceType::GPU) {
        }

        shape_ = other.shape_;
        device_type = other.device_type;
        size_t total_size = shape_.size();

        if (total_size > 0) {
            size_t bytes = total_size * sizeof(uint64_t);

            if (device_type == device::DeviceType::GPU) {
                throw std::runtime_error("GPU device requested but CUDA not available");
            } else {
                data_ptr = std::unique_ptr<uint64_t[]>(new uint64_t[total_size]{});
                std::copy(other.data_ptr.get(), other.data_ptr.get() + total_size, data_ptr.get());
            }
        } else {
            data_ptr.reset();
        }
    }
    return *this;
}

U64Array &U64Array::operator=(U64Array &&other) noexcept {
    if (this != &other) {
        data_ptr = std::move(other.data_ptr);
        shape_ = std::move(other.shape_);
        device_type = other.device_type;
    }
    return *this;
}

U64Array::~U64Array() {
    if (data_ptr && device_type == device::DeviceType::GPU) {
        std::cerr << "Error: GPU data found in destructor but CUDA not available!" << std::endl;
    }
}

uint64_t &U64Array::operator[](size_t index) {
    if (index >= size()) {
        throw std::out_of_range("Array index out of range");
    }
    return data_ptr[index];
}

const uint64_t &U64Array::operator[](size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Array index out of range");
    }
    return data_ptr[index];
}

uint64_t &U64Array::at(const std::vector<size_t> &indices) {
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

const uint64_t &U64Array::at(const std::vector<size_t> &indices) const {
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

U64Array U64Array::to(device::DeviceType target_device) const {
    if (device_type == target_device) {
        return *this;
    }

    U64Array result(shape_, target_device);
    if (size() > 0) {
        size_t bytes = size() * sizeof(uint64_t);

        if (device_type == device::DeviceType::CPU && target_device == device::DeviceType::GPU) {

            std::cout << "Copying from CPU to GPU: " << bytes << " bytes" << std::endl;
            std::cout << "Source (CPU): " << static_cast<const void *>(data()) << std::endl;
            std::cout << "Dest (GPU): " << static_cast<void *>(result.data()) << std::endl;

            if (data() == nullptr) {
                throw std::runtime_error("Source CPU data is null");
            }
            if (result.data() == nullptr) {
                throw std::runtime_error("Destination GPU memory is null");
            }

            throw std::runtime_error("GPU device requested but CUDA not available");

        } else if (device_type == device::DeviceType::GPU && target_device == device::DeviceType::CPU) {
            std::cout << "Copying from GPU to CPU: " << bytes << " bytes" << std::endl;
            throw std::runtime_error("GPU device requested but CUDA not available");
        } else {
            std::copy(data(), data() + size(), result.data());
            throw std::runtime_error("GPU device requested but CUDA not available");
        }
    }
    return result;
}

U64Array U64Array::cpu() const { return to(device::DeviceType::CPU); }

U64Array U64Array::gpu() const { return to(device::DeviceType::GPU); }

U64Array U64Array::cuda() const { return to(device::DeviceType::GPU); }

void U64Array::fill(uint64_t value) { std::fill(data_ptr.get(), data_ptr.get() + size(), value); }

void U64Array::zeros() { fill(0); }

void U64Array::ones() { fill(1); }

U64Array U64Array::operator+(const U64Array &other) const { return add(*this, other); }

U64Array U64Array::operator+(uint64_t scalar) const { return add(*this, scalar); }

U64Array U64Array::operator*(const U64Array &other) const { return multiply(*this, other); }

U64Array U64Array::operator*(uint64_t scalar) const { return multiply(*this, scalar); }

void U64Array::print(std::string name, bool show_all) const { print_array(*this, name, show_all); }

U64Array operator+(uint64_t scalar, const U64Array &array) { return add(array, scalar); }

U64Array operator*(uint64_t scalar, const U64Array &array) { return multiply(array, scalar); }

U64Array add(const U64Array &a, const U64Array &b) {
    DeviceType target_device = DeviceType::CPU;
    if (a.device() == DeviceType::GPU || b.device() == DeviceType::GPU) {
        target_device = DeviceType::GPU;
    }

    auto result_shape = a.shape();
    U64Array result(result_shape, target_device);

    if (a.shape().dims == b.shape().dims && a.shape().dims == result_shape.dims) {
        if (target_device == DeviceType::GPU) {
            throw std::runtime_error("GPU operations not available - compiled without CUDA support");
        } else {
            auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
            auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
            u64comp::op.vec_add(cpu_a.data(), cpu_b.data(), result.data(), result.size());
        }
    } else {
        auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
        auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
        U64Array cpu_result(result_shape, DeviceType::CPU);

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

U64Array add(const U64Array &a, uint64_t scalar) {
    U64Array result(a.shape());
    u64comp::op.vec_add_scalar(a.data(), scalar, result.data(), result.size());
    return result;
}

U64Array multiply(const U64Array &a, const U64Array &b) {

    Shape result_shape = a.shape();
    DeviceType target_device = DeviceType::CPU;
    if (a.device() == DeviceType::GPU || b.device() == DeviceType::GPU) {
        target_device = DeviceType::GPU;
    }

    U64Array result(result_shape, target_device);

    if (a.shape().dims == b.shape().dims && a.shape().dims == result_shape.dims) {

        if (target_device == DeviceType::GPU) {

            throw std::runtime_error("GPU operations not available - compiled without CUDA support");

        } else {
            auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
            auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
            u64comp::op.vec_mul(cpu_a.data(), cpu_b.data(), result.data(), result.size());
        }
    } else {
        auto cpu_a = (a.device() == DeviceType::CPU) ? a : a.to(DeviceType::CPU);
        auto cpu_b = (b.device() == DeviceType::CPU) ? b : b.to(DeviceType::CPU);
        U64Array cpu_result(result_shape, DeviceType::CPU);

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

U64Array multiply(const U64Array &a, uint64_t scalar) {
    U64Array result(a.shape());
    u64comp::op.vec_mul_scalar(a.data(), scalar, result.data(), result.size());
    return result;
}

U64Array matmul(const U64Array &a, const U64Array &b) {
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

    U64Array result({a_rows, b_cols});

    // 使用SIMD优化的矩阵乘法
    // A: M×K (a_rows × a_cols), B: K×N (a_cols × b_cols), C: M×N (a_rows ×
    // b_cols) Leading dimensions: lda=K, ldb=N, ldc=N
    u64comp::op.mat_dot(a.data(), b.data(), result.data(), a_rows, b_cols, a_cols, // M, N, K
                        a_cols, b_cols, b_cols                                     // lda=K, ldb=N, ldc=N
    );

    return result;
}

U64Array arange(uint64_t start, uint64_t stop) {
    if (stop <= start) {
        return U64Array({0});
    }

    size_t count = stop - start;
    U64Array result({count});

    for (size_t i = 0; i < count; ++i) {
        result[i] = start + i;
    }

    return result;
}

U64Array zeros(const std::vector<size_t> &shape) {
    U64Array result(shape);
    std::memset(result.data(), 0, result.size() * sizeof(uint64_t));
    return result;
}

U64Array ones(const std::vector<size_t> &shape) {
    U64Array result(shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = 1;
    }
    return result;
}

U64Array full(const std::vector<size_t> &shape, uint64_t value) {
    U64Array result(shape);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = value;
    }
    return result;
}

U64Array concatenate(const std::vector<U64Array> &arrays, int axis) {
    if (arrays.empty()) {
        return U64Array({0});
    }

    if (axis != 0) {
        throw std::invalid_argument("Only axis=0 is supported for concatenation");
    }

    size_t total_size = 0;
    for (const auto &arr : arrays) {
        if (arr.shape().ndim() != 1) {
            throw std::invalid_argument("All arrays must be 1D for concatenation");
        }
        total_size += arr.size();
    }

    U64Array result({total_size});

    size_t offset = 0;
    for (const auto &arr : arrays) {
        std::memcpy(result.data() + offset, arr.data(), arr.size() * sizeof(uint64_t));
        offset += arr.size();
    }

    return result;
}

} // namespace np
} // namespace u64comp