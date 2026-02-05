#include "simd_detect.h"
#include <iostream>
#include <string>

ArchType SIMDDetector::detected_arch = ArchType::UNKNOWN;
uint32_t SIMDDetector::detected_capabilities = 0;

bool detect_simd(hwInfo *info) {
    SIMDDetector::init();
    info->type = DetectedType::ARM_SIMD;
    info->capabilities = SIMDDetector::detected_capabilities;
    SIMDDetector::simd2string();
    return true;
}

#if defined(ARCH_ARM64)
void SIMDDetector::detect_arm64_features() { detected_capabilities |= static_cast<uint32_t>(SIMDCapability::NEON); }
#endif

#if defined(ARCH_X86_64)
void SIMDDetector::cpuid(int info[4], int function_id) {
#ifdef COMPILER_GCC
    int success = __get_cpuid(function_id, reinterpret_cast<unsigned int *>(&info[0]), reinterpret_cast<unsigned int *>(&info[1]),
                              reinterpret_cast<unsigned int *>(&info[2]), reinterpret_cast<unsigned int *>(&info[3]));
    if (!success) {
        info[0] = info[1] = info[2] = info[3] = 0;
    }
#elif defined(COMPILER_MSVC)
    __cpuid(info, function_id);
#else
    // Fallback: set all to 0
    info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

void SIMDDetector::detect_x64_features() {
    int info[4];
    cpuid(info, 0);
    int max_func = info[0];
    if (max_func >= 1) {
        cpuid(info, 1);
        // ECX and EDX contain feature flags
        int ecx = info[2];
        int edx = info[3];

        // SSE2 (EDX bit 26)
        if (edx & (1 << 26)) {
            detected_capabilities |= static_cast<uint32_t>(SIMDCapability::SSE2);
        }

        // SSE4.1 (ECX bit 19)
        if (ecx & (1 << 19)) {
            detected_capabilities |= static_cast<uint32_t>(SIMDCapability::SSE41);
        }
    }
    if (max_function >= 7) {
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            // AVX2 (EBX bit 5)
            if (ebx & (1 << 5)) {
                detected_capabilities |= static_cast<uint32_t>(SIMDCapability::AVX2);
            }

            // AVX-512F (EBX bit 16)
            if (ebx & (1 << 16)) {
                detected_capabilities |= static_cast<uint32_t>(SIMDCapability::AVX512);
            }
        }
    }
}
#endif

bool SIMDDetector::has_capability(SIMDCapability cap) { return (detected_capabilities & static_cast<uint32_t>(cap)) != 0; }
int SIMDDetector::get_best_vec_width() {

    if (has_capability(SIMDCapability::AVX512))
        return 512;
    if (has_capability(SIMDCapability::AVX2))
        return 256;
    if (has_capability(SIMDCapability::SSE41) || has_capability(SIMDCapability::SSE2) || has_capability(SIMDCapability::NEON))
        return 128;
    return 64; // 标量
}

void SIMDDetector::init() {
#if defined(ARCH_X86_64)
    detected_arch = ArchType::X86_64;
    detect_x64_features();
#elif defined(ARCH_ARM64)
    detected_arch = ArchType::ARM64;
    detect_arm64_features();
#else
    detected_arch = ArchType::UNKNOWN;
#endif
}

std::string SIMDDetector::simd2string() {
    std::string features;

    if (has_capability(SIMDCapability::SSE2))
        features += "SSE2 ";
    if (has_capability(SIMDCapability::SSE41))
        features += "SSE4.1 ";
    if (has_capability(SIMDCapability::AVX2))
        features += "AVX2 ";
    if (has_capability(SIMDCapability::AVX512))
        features += "AVX512F ";
    if (has_capability(SIMDCapability::NEON))
        features += "NEON ";
    std::cout << "simd features: " << features << std::endl;
    return features.empty() ? "None" : features;
}
