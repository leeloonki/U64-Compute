#include "ops_simd_arm.h"
#include <arm_neon.h>
void add_u64(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count) {
    size_t simd_count = (count / 2) * 2;
    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vr = vaddq_u64(va, vb);
        vst1q_u64(result + i, vr);
    }

    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

inline uint64x2_t vmulq_u64_emulated(uint64x2_t a, uint64x2_t b) {
    uint32x2_t a_lo = vmovn_u64(a);       // 低32位
    uint32x2_t a_hi = vshrn_n_u64(a, 32); // 高32位
    uint32x2_t b_lo = vmovn_u64(b);       // 低32位
    uint32x2_t b_hi = vshrn_n_u64(b, 32); // 高32位

    // 计算 (a_hi * b_lo + a_lo * b_hi) << 32 + a_lo * b_lo
    uint64x2_t lo_lo = vmull_u32(a_lo, b_lo); // a_lo * b_lo
    uint64x2_t hi_lo = vmull_u32(a_hi, b_lo); // a_hi * b_lo
    uint64x2_t lo_hi = vmull_u32(a_lo, b_hi); // a_lo * b_hi

    // (a_hi * b_lo + a_lo * b_hi) << 32
    uint64x2_t mid = vaddq_u64(hi_lo, lo_hi);
    uint64x2_t mid_shifted = vshlq_n_u64(mid, 32);

    return vaddq_u64(lo_lo, mid_shifted);
}

void multiply_u64(const uint64_t *a, const uint64_t *b, uint64_t *result, size_t count) {
    size_t simd_count = (count / 2) * 2;

    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vb = vld1q_u64(b + i);
        uint64x2_t vr = vmulq_u64_emulated(va, vb);
        vst1q_u64(result + i, vr);
    }

    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}
void add_scalar_u64(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count) {
    size_t simd_count = (count / 2) * 2;
    uint64x2_t vs = vdupq_n_u64(scalar);

    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vr = vaddq_u64(va, vs);
        vst1q_u64(result + i, vr);
    }

    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scalar;
    }
}
void multiply_scalar_u64(const uint64_t *a, uint64_t scalar, uint64_t *result, size_t count) {
    size_t simd_count = (count / 2) * 2;
    uint64x2_t vs = vdupq_n_u64(scalar);

    for (size_t i = 0; i < simd_count; i += 2) {
        uint64x2_t va = vld1q_u64(a + i);
        uint64x2_t vr = vmulq_u64_emulated(va, vs);
        vst1q_u64(result + i, vr);
    }

    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] * scalar;
    }
}
void matmul_block_u64(const uint64_t *A, const uint64_t *B, uint64_t *C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) {
    constexpr size_t BLOCK_SIZE = 32;

    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += BLOCK_SIZE) {
                size_t i_end = std::min(i + BLOCK_SIZE, M);
                size_t j_end = std::min(j + BLOCK_SIZE, N);
                size_t k_end = std::min(k + BLOCK_SIZE, K);

                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t jj = j; jj < j_end; ++jj) {
                        uint64x2_t sum_vec = vdupq_n_u64(0);
                        size_t kk;

                        for (kk = k; kk + 1 < k_end; kk += 2) {
                            uint64x2_t a_vec = vld1q_u64(&A[ii * lda + kk]);
                            uint64x2_t b_vec = {B[kk * ldb + jj], B[(kk + 1) * ldb + jj]};
                            uint64x2_t prod = vmulq_u64_emulated(a_vec, b_vec);
                            sum_vec = vaddq_u64(sum_vec, prod);
                        }

                        uint64_t sum = vgetq_lane_u64(sum_vec, 0) + vgetq_lane_u64(sum_vec, 1);

                        for (; kk < k_end; ++kk) {
                            sum += A[ii * lda + kk] * B[kk * ldb + jj];
                        }

                        C[ii * ldc + jj] += sum;
                    }
                }
            }
        }
    }
}
