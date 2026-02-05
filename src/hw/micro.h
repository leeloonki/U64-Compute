// 1. compiler
#if defined(__clang__)
#define COMPILER_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
#define COMPILER_GCC
#elif defined(_MSC_VER)
#define COMPILER_MSVC
#endif

// 2. arch
#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X86_64
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ARCH_ARM64
#else
#define ARCH_UNKNOWN
#endif