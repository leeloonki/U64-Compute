#include "ops.h"
#include "hw/detect.h"
#include "ops_cpu.h"
#include "ops_gpu_cuda.h"
#include "ops_simd_arm.h"
#include "ops_simd_x64.h"
#include <memory>
void ops::init() {
    auto det = detect();
    switch (det.type) {
    case DetectedType::NVIDIA_GPU:
        this->ops_ptr = std::make_unique<ops_cuda>(det.capabilities);
        break;
    case DetectedType::ARM_SIMD:
        this->ops_ptr = std::make_unique<ops_simd_arm>(det.capabilities);
        break;
    case DetectedType::X64_SIMD:
        this->ops_ptr = std::make_unique<ops_simd_x64>(det.capabilities);
        break;
    case DetectedType::PLAIN_CPU:
        this->ops_ptr = std::make_unique<ops_cpu>(det.capabilities);
        break;
    default:
        break;
    }
}