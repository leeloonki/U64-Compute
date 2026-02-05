#pragma once
#include "ops_base.h"
#include <iostream>
#include <memory>
class ops {
public:
    void init();
    ops() {
        init();
        std::cout << "select best ops" << std::endl;
    }

    void vec_add(const uint64_t *a, const uint64_t *b, uint64_t *res, size_t count) { return ops_ptr->vec_add(a, b, res, count); }

    void vec_mul(const uint64_t *a, const uint64_t *b, uint64_t *res, size_t count) { return ops_ptr->vec_mul(a, b, res, count); }

    void vec_add_scalar(const uint64_t *a, uint64_t scalar, uint64_t *res, size_t count) { return ops_ptr->vec_add_scalar(a, scalar, res, count); }

    void vec_mul_scalar(const uint64_t *a, uint64_t scalar, uint64_t *res, size_t count) { return ops_ptr->vec_mul_scalar(a, scalar, res, count); }

    void mat_dot(const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
        return ops_ptr->mat_dot(A, B, C, M, N, K, lda, ldb, ldc);
    }

private:
    std::unique_ptr<ops_base> ops_ptr;
};