#pragma once
#include "ops_base.h"
#include <iostream>
void add_u64(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count);
void multiply_u64(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count);
void add_scalar_u64(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count);
void multiply_scalar_u64(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count);
void matmul_block_u64(const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);

class ops_simd_arm : public ops_base {
private:
public:
    ops_simd_arm(uint32_t cap) {
        capabilities = cap;
        init();
    }

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
        best_vec_add = [](const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count) { add_u64(a, b, result, count); };
        best_vec_mul = [](const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count) { multiply_u64(a, b, result, count); };
        best_vec_add_scalar = [](const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count) { add_scalar_u64(a, scalar, result, count); };
        best_vec_mul_scalar = [](const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count) {
            multiply_scalar_u64(a, scalar, result, count);
        };

        best_mat_dot = [](const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
            matmul_block_u64(A, B, C, M, N, K, lda, ldb, ldc);
        };

        selected_impl = "NEON";
        std::cout << "Selected ARM SIMD: NEON" << std::endl;
        return;
    }

    void tostring() override { std::cout << "ARM Ops" << std::endl; }
};