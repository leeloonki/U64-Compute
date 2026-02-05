#pragma once
#include "op/ops.h"
#include <cstdint>
#include <memory>
#include <vector>
namespace u64comp {

namespace device {
enum class DeviceType { CPU, GPU };
constexpr const char *CPU = "CPU";
constexpr const char *GPU = "GPU";
constexpr const char *CUDA = "CUDA";

} // namespace device

namespace np {

class Shape {
public:
    std::vector<size_t> dims;

    Shape() = default;
    Shape(const std::vector<size_t> &dimensions) : dims(dimensions) {}
    Shape(std::initializer_list<size_t> dimensions) : dims(dimensions) {}

    size_t ndim() const { return dims.size(); }
    size_t size() const;
    size_t stride(size_t dim) const;

    bool operator==(const Shape &other) const { return dims == other.dims; }
    bool operator!=(const Shape &other) const { return dims != other.dims; }
};

class U64Array {
private:
    std::unique_ptr<uint64_t[]> data_ptr;
    Shape shape_;
    device::DeviceType device_type;

public:
    U64Array() : shape_({}), device_type(device::DeviceType::CPU) {}
    U64Array(const Shape &shape, device::DeviceType device = device::DeviceType::CPU);
    U64Array(const std::vector<size_t> &shape, device::DeviceType device = device::DeviceType::CPU);
    U64Array(std::initializer_list<size_t> shape, device::DeviceType device = device::DeviceType::CPU);

    U64Array(const U64Array &other);
    U64Array(U64Array &&other) noexcept;
    U64Array &operator=(const U64Array &other);
    U64Array &operator=(U64Array &&other) noexcept;

    ~U64Array();

    const Shape &shape() const { return shape_; }
    size_t size() const { return shape_.size(); }
    size_t ndim() const { return shape_.ndim(); }
    device::DeviceType device() const { return device_type; }

    uint64_t *data() { return data_ptr.get(); }
    const uint64_t *data() const { return data_ptr.get(); }

    uint64_t &operator[](size_t index);
    const uint64_t &operator[](size_t index) const;
    uint64_t &at(const std::vector<size_t> &indices);
    const uint64_t &at(const std::vector<size_t> &indices) const;
    U64Array to(device::DeviceType target_device) const;
    U64Array cpu() const;
    U64Array gpu() const;
    U64Array cuda() const;

    void fill(uint64_t value);
    void zeros();
    void ones();

    U64Array operator+(const U64Array &other) const;
    U64Array operator+(uint64_t scalar) const;
    U64Array operator*(const U64Array &other) const;
    U64Array operator*(uint64_t scalar) const;

    void print(std::string name = "", bool show_all = false) const;
};

// scalar + U64Array -> U64Array
U64Array operator+(uint64_t scalar, const U64Array &array);
U64Array operator*(uint64_t scalar, const U64Array &array);

U64Array add(const U64Array &a, const U64Array &b);
U64Array add(const U64Array &a, uint64_t scalar);
U64Array multiply(const U64Array &a, const U64Array &b);
U64Array multiply(const U64Array &a, uint64_t scalar);
U64Array matmul(const U64Array &a, const U64Array &b);

U64Array arange(uint64_t start, uint64_t stop);
U64Array zeros(const std::vector<size_t> &shape);
U64Array ones(const std::vector<size_t> &shape);
U64Array full(const std::vector<size_t> &shape, uint64_t value);
U64Array concatenate(const std::vector<U64Array> &arrays, int axis = 0);

} // namespace np

} // namespace u64comp