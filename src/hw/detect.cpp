#include "gpu_detect.h"
#include "simd_detect.h"

hwInfo detect() {
    hwInfo info = {};
    if (detect_cuda(&info)) {
        return info;
    }
    if (detect_simd(&info)) {
        return info;
    } else {
        return info;
    }
}