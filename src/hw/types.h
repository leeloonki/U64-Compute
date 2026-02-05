#pragma once
#include <cstdint>
enum class DetectedType { PLAIN_CPU, ARM_SIMD, X64_SIMD, NVIDIA_GPU };

struct hwInfo {
    DetectedType type;
    uint32_t capabilities;
};
