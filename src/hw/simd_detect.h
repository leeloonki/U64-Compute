#pragma once
#include <iostream>
#include <string>
#include "types.h"
#include "micro.h"

// 3. simd
#ifdef ARCH_X86_64
#include <immintrin.h>
#ifdef __SSE2__
#define SIMD_HAS_SSE2 // 128bit
#endif
#ifdef __SSE4_1__
#define SIMD_HAS_SSE41 // 128bit
#endif
#ifdef __AVX2__
#define SIMD_HAS_AVX2 // 256bit
#endif
#ifdef __AVX512F__
#define SIMD_HAS_AVX512F // 512bit
#endif
#endif

#if defined(ARCH_ARM64)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SIMD_HAS_NEON
#include <arm_neon.h>
#endif
#endif

bool detect_simd(hwInfo *);

enum class SIMDCapability {
    NONE = 0,
    SSE2 = 1 << 0,
    SSE41 = 1 << 1,
    AVX2 = 1 << 2,
    AVX512 = 1 << 3,
    NEON = 1 << 4,
};

enum class ArchType { UNKNOWN, X64, ARM64 };

class SIMDDetector {
public:
    static ArchType detected_arch;
    static uint32_t detected_capabilities;

#if defined(ARCH_ARM64)
    static void detect_arm64_features();
#endif

#if defined(ARCH_X86_64)
    static void cpuid(int info[4], int func_id);
    static void detect_x64_features();
#endif
    static bool has_capability(SIMDCapability cap);
    static int get_best_vec_width();

public:
    static void init();
    static std::string simd2string();
    static ArchType get_arch() { return detected_arch; }
};