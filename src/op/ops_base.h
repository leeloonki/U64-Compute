#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
using vec_add_func = std::function<void(const uint64_t *, const uint64_t *, uint64_t *, size_t)>;
using vec_mul_func = std::function<void(const uint64_t *, const uint64_t *, uint64_t *, size_t)>;
using vec_add_scalar_func = std::function<void(const uint64_t *, uint64_t, uint64_t *, size_t)>;
using vec_mul_scalar_func = std::function<void(const uint64_t *, uint64_t, uint64_t *, size_t)>;
using mat_dot_func = std::function<void(const uint64_t *, const uint64_t *, uint64_t *, size_t, size_t, size_t, size_t, size_t, size_t)>;

class ops_base {
public:
    virtual void init() = 0;
    virtual void tostring() = 0;

    virtual void vec_add(const uint64_t *, const uint64_t *, uint64_t *, size_t) = 0;

    virtual void vec_mul(const uint64_t *, const uint64_t *, uint64_t *, size_t) = 0;

    virtual void vec_add_scalar(const uint64_t *, uint64_t, uint64_t *, size_t) = 0;

    virtual void vec_mul_scalar(const uint64_t *, uint64_t, uint64_t *, size_t) = 0;

    virtual void mat_dot(const uint64_t *, const uint64_t *, uint64_t *, size_t, size_t, size_t, size_t, size_t, size_t) = 0;

protected:
    vec_add_func best_vec_add;
    vec_mul_func best_vec_mul;
    vec_add_scalar_func best_vec_add_scalar;
    vec_mul_scalar_func best_vec_mul_scalar;
    mat_dot_func best_mat_dot;
    std::string selected_impl;
    uint32_t capabilities = 0;
};