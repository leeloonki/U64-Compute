#pragma once

#include <cstdint>
#include <string>

// 架构检测宏
#if defined(__x86_64__)
    #ifndef SIMD_ARCH_X86_64
        #define SIMD_ARCH_X86_64
    #endif
#elif defined(__aarch64__)
    #ifndef SIMD_ARCH_ARM64
        #define SIMD_ARCH_ARM64
    #endif
#else
    #ifndef SIMD_ARCH_UNKNOWN
        #define SIMD_ARCH_UNKNOWN
    #endif
#endif

// SIMD指令集检测
#ifdef SIMD_ARCH_X86_64
    #include <immintrin.h>
    #ifdef __SSE2__
        #ifndef SIMD_HAS_SSE2
            #define SIMD_HAS_SSE2
        #endif
    #endif
    #ifdef __SSE4_1__
        #ifndef SIMD_HAS_SSE41
            #define SIMD_HAS_SSE41
        #endif
    #endif
    #ifdef __AVX__
        #ifndef SIMD_HAS_AVX
            #define SIMD_HAS_AVX
        #endif
    #endif
    #ifdef __AVX2__
        #ifndef SIMD_HAS_AVX2
            #define SIMD_HAS_AVX2
        #endif
    #endif
    #ifdef __AVX512F__
        #ifndef SIMD_HAS_AVX512F
            #define SIMD_HAS_AVX512F
        #endif
    #endif
#endif

#if defined(SIMD_ARCH_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #ifndef SIMD_HAS_NEON
            #define SIMD_HAS_NEON
        #endif
        #include <arm_neon.h>
    #endif
    #ifdef __ARM_FEATURE_SVE
        #ifndef SIMD_HAS_SVE
            #define SIMD_HAS_SVE
        #endif
        #include <arm_sve.h>
    #endif
#endif

namespace uint64_compute {

// SIMD能力枚举
enum class SIMDCapability {
    NONE = 0,
    // x86 指令集
    SSE2 = 1 << 0,
    SSE41 = 1 << 1,
    AVX = 1 << 2,
    AVX2 = 1 << 3,
    AVX512F = 1 << 4,
    // ARM 指令集
    NEON = 1 << 8,
    SVE = 1 << 9
};

// 架构类型
enum class ArchType {
    UNKNOWN,
    X86_64,
    ARM64
};

// 跨平台SIMD检测器
class CrossPlatformSIMD {
private:
    static uint32_t detected_capabilities;
    static ArchType detected_arch;
    static bool detection_done;
    
    // x86 CPUID检测
    #ifdef SIMD_ARCH_X86_64
    static void detect_x86_features();
    static bool has_cpuid();
    static void cpuid(int info[4], int function_id);
    #endif
    
    // ARM特性检测
    #if defined(SIMD_ARCH_ARM64)
    static void detect_arm_features();
    #endif
    
public:
    // 执行检测
    static void detect();
    
    // 查询能力
    static bool has_capability(SIMDCapability cap);
    static ArchType get_architecture();
    static std::string get_feature_string();
    static int get_best_simd_width(); // 返回最佳SIMD宽度(位)
    
    // 具体指令集检测
    static bool has_sse2() { detect(); return has_capability(SIMDCapability::SSE2); }
    static bool has_sse41() { detect(); return has_capability(SIMDCapability::SSE41); }
    static bool has_avx() { detect(); return has_capability(SIMDCapability::AVX); }
    static bool has_avx2() { detect(); return has_capability(SIMDCapability::AVX2); }
    static bool has_avx512f() { detect(); return has_capability(SIMDCapability::AVX512F); }
    static bool has_neon() { detect(); return has_capability(SIMDCapability::NEON); }
    static bool has_sve() { detect(); return has_capability(SIMDCapability::SVE); }
};

// SIMD调度器 - 在运行时选择最佳实现
template<typename T>
class SIMDDispatcher {
public:
    // 函数指针类型定义
    using AddFunc = void(*)(const T*, const T*, T*, size_t);
    using MulFunc = void(*)(const T*, const T*, T*, size_t);
    
private:
    static AddFunc best_add_func;
    static MulFunc best_mul_func;
    static bool dispatched;
    
public:
    static void dispatch();
    static void add(const T* a, const T* b, T* result, size_t count);
    static void multiply(const T* a, const T* b, T* result, size_t count);
};

} // namespace uint64_compute