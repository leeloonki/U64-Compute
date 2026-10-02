#include "simd_detect.h"

#if defined(SIMD_ARCH_X86_64)
    #include <cpuid.h>
#endif

#if defined(SIMD_ARCH_ARM64)
    #ifdef __linux__
        #include <unistd.h>
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
        #ifndef AT_HWCAP
            #define AT_HWCAP 16
        #endif
        #ifndef HWCAP_NEON
            #define HWCAP_NEON (1 << 12)
        #endif
    #endif
#endif

namespace uint64_compute {

// 静态成员初始化
uint32_t CrossPlatformSIMD::detected_capabilities = 0;
ArchType CrossPlatformSIMD::detected_arch = ArchType::UNKNOWN;
bool CrossPlatformSIMD::detection_done = false;

void CrossPlatformSIMD::detect() {
    if (detection_done) return;
    
    // 检测架构类型
    #if defined(SIMD_ARCH_X86_64)
        detected_arch = ArchType::X86_64;
        detect_x86_features();
    #elif defined(SIMD_ARCH_ARM64)
        detected_arch = ArchType::ARM64;
        detect_arm_features();
    #else
        detected_arch = ArchType::UNKNOWN;
    #endif
    
    detection_done = true;
}

#if defined(SIMD_ARCH_X86_64)
void CrossPlatformSIMD::detect_x86_features() {
    if (!has_cpuid()) return;
    
    int info[4];
    
    // 检查基本CPUID支持
    cpuid(info, 0);
    int max_function = info[0];
    
    if (max_function >= 1) {
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
        
        // AVX (ECX bit 28)
        if (ecx & (1 << 28)) {
            detected_capabilities |= static_cast<uint32_t>(SIMDCapability::AVX);
        }
    }
    
    // 检查扩展特性 - 使用与测试程序相同的方法
    if (max_function >= 7) {
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            // AVX2 (EBX bit 5)
            if (ebx & (1 << 5)) {
                detected_capabilities |= static_cast<uint32_t>(SIMDCapability::AVX2);
            }
            
            // AVX-512F (EBX bit 16)
            if (ebx & (1 << 16)) {
                detected_capabilities |= static_cast<uint32_t>(SIMDCapability::AVX512F);
            }
        }
    }
}

bool CrossPlatformSIMD::has_cpuid() {
    #if defined(__x86_64__) || defined(_M_X64)
        // x86_64 架构总是支持 CPUID
        return true;
    #else
        // 其他架构不支持 CPUID
        return false;
    #endif
}

void CrossPlatformSIMD::cpuid(int info[4], int function_id) {
    #ifdef __GNUC__
        int success = __get_cpuid(function_id, 
                   reinterpret_cast<unsigned int*>(&info[0]),
                   reinterpret_cast<unsigned int*>(&info[1]), 
                   reinterpret_cast<unsigned int*>(&info[2]),
                   reinterpret_cast<unsigned int*>(&info[3]));
        if (!success) {
            info[0] = info[1] = info[2] = info[3] = 0;
        }
    #elif defined(_MSC_VER)
        __cpuid(info, function_id);
    #else
        // Fallback: set all to 0
        info[0] = info[1] = info[2] = info[3] = 0;
    #endif
}
#endif

#if defined(SIMD_ARCH_ARM64) || defined(SIMD_ARCH_ARM32)
void CrossPlatformSIMD::detect_arm_features() {
    // 编译时检测
    #ifdef SIMD_HAS_NEON
        detected_capabilities |= static_cast<uint32_t>(SIMDCapability::NEON);
    #endif
    
    
    // 运行时检测(仅Linux)
    #if defined(__linux__) && defined(SIMD_ARCH_ARM64)
        unsigned long hwcap = getauxval(AT_HWCAP);
        if (hwcap & HWCAP_NEON) {
            detected_capabilities |= static_cast<uint32_t>(SIMDCapability::NEON);
        }
        
        #ifdef AT_HWCAP2
            unsigned long hwcap2 = getauxval(AT_HWCAP2);
            #ifdef HWCAP2_SVE
                if (hwcap2 & HWCAP2_SVE) {
                    detected_capabilities |= static_cast<uint32_t>(SIMDCapability::SVE);
                }
            #endif
        #endif
    #endif
}
#endif

bool CrossPlatformSIMD::has_capability(SIMDCapability cap) {
    detect();
    return (detected_capabilities & static_cast<uint32_t>(cap)) != 0;
}

ArchType CrossPlatformSIMD::get_architecture() {
    detect();
    return detected_arch;
}

std::string CrossPlatformSIMD::get_feature_string() {
    detect();
    std::string features;
    
    if (has_capability(SIMDCapability::SSE2)) features += "SSE2 ";
    if (has_capability(SIMDCapability::SSE41)) features += "SSE4.1 ";
    if (has_capability(SIMDCapability::AVX2)) features += "AVX2 ";
    if (has_capability(SIMDCapability::AVX512F)) features += "AVX512F ";
    if (has_capability(SIMDCapability::NEON)) features += "NEON ";
    
    return features.empty() ? "None" : features;
}

int CrossPlatformSIMD::get_best_simd_width() {
    detect();
    
    if (has_capability(SIMDCapability::AVX512F)) return 512;
    if (has_capability(SIMDCapability::AVX2)) return 256;
    if (has_capability(SIMDCapability::SSE2) || has_capability(SIMDCapability::NEON)) return 128;
    
    return 64; // 标量
}

} // namespace uint64_compute