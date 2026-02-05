#include "ops_cpu.h"

void plain_vec_add(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

void plain_vec_mul(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

void plain_vec_add_scalar(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}

void plain_vec_mul_scalar(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}

void plain_mat_dot(const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            uint64_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}
