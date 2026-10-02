#pragma once

#include <cstdint>
#include <string>

// 编译器
#if defined(__clang__)
    #define COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
    #define COMPILER_GCC
#elif defined(_MSC_VER)
    #define COMPILER_MSVC
#endif

// 架构检测宏
#if defined(__x86_64__) || defined(_M_X64)
    #define SIMD_ARCH_X86_64
#elif defined(__aarch64__)|| defined(_M_ARM64)
    #define SIMD_ARCH_ARM64
#else
    #define SIMD_ARCH_UNKNOWN
#endif

// SIMD指令集检测


// 指令集,核心能力
// SSE2/4.1 128 位向量化
// AVX2,256 位整数运算, FMA
// AVX512,512 位超长向量

#ifdef SIMD_ARCH_X86_64
    #include <immintrin.h>
    #ifdef __SSE2__
        #define SIMD_HAS_SSE2
    #endif
    #ifdef __SSE4_1__
        #define SIMD_HAS_SSE41
    #endif
    #ifdef __AVX2__
        #define SIMD_HAS_AVX2
    #endif
    #ifdef __AVX512F__
        #define SIMD_HAS_AVX512F
    #endif
#endif

#if defined(SIMD_ARCH_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #define SIMD_HAS_NEON
        #include <arm_neon.h>
    #endif
    // #ifdef __ARM_FEATURE_SVE 
    //     #define SIMD_HAS_SVE
    //     #include <arm_sve.h>
    // #endif
#endif

namespace uint64_compute {

// SIMD能力枚举
enum class SIMDCapability {
    NONE = 0,
    // x86 指令集
    SSE2 = 1 << 0,
    SSE41 = 1 << 1,
    AVX2 = 1 << 3,
    AVX512F = 1 << 4,
    // ARM 指令集
    NEON = 1 << 8,
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
    #if defined(SIMD_ARCH_X86_64)
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
    static bool has_avx2() { detect(); return has_capability(SIMDCapability::AVX2); }
    static bool has_avx512f() { detect(); return has_capability(SIMDCapability::AVX512F); }
    static bool has_neon() { detect(); return has_capability(SIMDCapability::NEON); }
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