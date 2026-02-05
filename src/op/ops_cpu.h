#pragma once
#include "ops_base.h"

void plain_vec_add(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count);
void plain_vec_mul(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count);
void plain_vec_add_scalar(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count);
void plain_vec_mul_scalar(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count);
void plain_mat_dot(const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);

class ops_cpu : public ops_base {
public:
    ops_cpu(uint32_t cap) { capabilities = cap; }

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

    void init() override {
        best_vec_add = plain_vec_add;
        best_vec_mul = plain_vec_mul;
        best_vec_add_scalar = plain_vec_add_scalar;
        best_vec_mul_scalar = plain_vec_mul_scalar;
        best_mat_dot = plain_mat_dot;
    }

    void tostring() override { std::cout << "CPU Ops" << std::endl; }
};