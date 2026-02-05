#pragma once
#include "ops.h"

class ops_simd_x64 : public ops_base {
public:
    ops_simd_x64(uint32_t cap) {
        capabilities = cap;
        init();
    }
    void init() override {}

    void vec_add(const uint64_t *a, const uint64_t *b, uint64_t *res, size_t count) override { return best_vec_add(a, b, res, count); }

    void vec_mul(const uint64_t *a, const uint64_t *b, uint64_t *res, size_t count) override { return best_vec_mul(a, b, res, count); }

    void vec_add_scalar(const uint64_t *a, uint64_t scalar, uint64_t *res, size_t count) override {
        return best_vec_add_scalar(a, scalar, res, count);
    }

    void vec_mul_scalar(const uint64_t *a, uint64_t scalar, uint64_t *res, size_t count) override {
        return best_vec_mul_scalar(a, scalar, res, count);
    }

    void mat_dot(const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) override {
        return best_mat_dot(A, B, C, M, N, K, lda, ldb, ldc);
    }

    void tostring() override { std::cout << "X64 Ops" << std::endl; }
};