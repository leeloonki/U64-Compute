#pragma once

// 平台和架构检测
#if defined(__x86_64__) || defined(_M_X64)
    #define UINT64COMPUTE_ARCH_X86_64
    #define UINT64COMPUTE_PLATFORM_X64
#elif defined(__aarch64__)
    #define UINT64COMPUTE_ARCH_ARM64
    #define UINT64COMPUTE_PLATFORM_ARM64
#else
    #define UINT64COMPUTE_ARCH_UNKNOWN
#endif

// 操作系统检测
#if defined(__linux__)
    #define UINT64COMPUTE_PLATFORM_LINUX
#elif defined(_WIN32) || defined(_WIN64)
    #define UINT64COMPUTE_PLATFORM_WINDOWS
#elif defined(__APPLE__)
    #define UINT64COMPUTE_PLATFORM_MACOS
#else
    #define UINT64COMPUTE_PLATFORM_UNKNOWN
#endif

// 缓存行大小
#if defined(UINT64COMPUTE_ARCH_X86_64)
    #define UINT64COMPUTE_CACHE_LINE_SIZE 64
#elif defined(UINT64COMPUTE_ARCH_ARM64)
    #define UINT64COMPUTE_CACHE_LINE_SIZE 64
#else
    #define UINT64COMPUTE_CACHE_LINE_SIZE 64
#endif

// 编译器特性
#ifdef __GNUC__
    #define UINT64COMPUTE_COMPILER_GCC
    #define UINT64COMPUTE_INLINE __inline__
    #define UINT64COMPUTE_FORCE_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define UINT64COMPUTE_COMPILER_MSVC
    #define UINT64COMPUTE_INLINE __inline
    #define UINT64COMPUTE_FORCE_INLINE __forceinline
#else
    #define UINT64COMPUTE_INLINE inline
    #define UINT64COMPUTE_FORCE_INLINE inline
#endif

// 版本信息
#define UINT64COMPUTE_VERSION_MAJOR 1
#define UINT64COMPUTE_VERSION_MINOR 0
#define UINT64COMPUTE_VERSION_PATCH 0