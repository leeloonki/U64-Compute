#include "hardware_detector.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstring>
#include <random>
#include <iostream>
#include <sstream>
#include <set>
#include <fstream>

// Platform-specific includes
#ifdef UINT64COMPUTE_PLATFORM_LINUX
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #include <cpuid.h>
    #include <fstream>
#elif defined(UINT64COMPUTE_PLATFORM_WINDOWS)
    #include <windows.h>
    #include <intrin.h>
#elif defined(UINT64COMPUTE_PLATFORM_MACOS)
    #include <sys/sysctl.h>
    #include <sys/types.h>
    #include <mach/mach.h>
    #include <mach/vm_statistics.h>
    #include <mach/mach_host.h>
    #ifdef UINT64COMPUTE_ARCH_X86_64
        #include <cpuid.h>
    #endif
#endif

// CUDA headers if available
#ifdef UINT64COMPUTE_HAS_CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#endif

namespace uint64_compute {
namespace hardware {

// ================================
// HardwareDetector Implementation
// ================================

HardwareDetector& HardwareDetector::instance() {
    static HardwareDetector instance;
    return instance;
}

const SystemInfo& HardwareDetector::detect_system() const {
    if (!detection_done) {
        detect_cpu_info();
        detect_memory_info();
        detect_gpu_info();
        detect_parallel_features();
        detection_done = true;
    }
    return system_info_;
}

void HardwareDetector::detect_cpu_info() const {
    system_info_.cpu = CPUInfo{};
    auto& cpu = system_info_.cpu;
    
    // Detect architecture
#ifdef UINT64COMPUTE_ARCH_X86_64
    cpu.architecture = "x86_64";
    detect_x86_features();
#elif defined(UINT64COMPUTE_ARCH_X86_32)
    cpu.architecture = "x86_32";
    detect_x86_features();
#elif defined(UINT64COMPUTE_ARCH_ARM64)
    cpu.architecture = "arm64";
    detect_arm_features();
#elif defined(UINT64COMPUTE_ARCH_ARM32)
    cpu.architecture = "arm32";
    detect_arm_features();
#else
    cpu.architecture = "unknown";
#endif

    // Detect core counts
    cpu.cores_logical = std::thread::hardware_concurrency();
    
#ifdef UINT64COMPUTE_PLATFORM_LINUX
    cpu.cores_physical = cpu.cores_logical; // Default assumption
    
    // Try to get more accurate information from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        std::string model_name;
        int physical_id = -1;
        int core_id = -1;
        std::set<std::pair<int, int>> unique_cores;
        
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    model_name = line.substr(colon + 2);
                    if (cpu.brand.empty()) {
                        cpu.brand = model_name;
                    }
                }
            } else if (line.find("physical id") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    physical_id = std::stoi(line.substr(colon + 2));
                }
            } else if (line.find("core id") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    core_id = std::stoi(line.substr(colon + 2));
                }
            }
            
            if (physical_id >= 0 && core_id >= 0) {
                unique_cores.insert({physical_id, core_id});
                physical_id = core_id = -1;
            }
        }
        
        if (!unique_cores.empty()) {
            cpu.cores_physical = unique_cores.size();
        }
    }
    
#elif defined(UINT64COMPUTE_PLATFORM_WINDOWS)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    cpu.cores_logical = sysinfo.dwNumberOfProcessors;
    cpu.cores_physical = cpu.cores_logical; // Simplified
    
#elif defined(UINT64COMPUTE_PLATFORM_MACOS)
    size_t size = sizeof(cpu.cores_physical);
    sysctlbyname("hw.physicalcpu", &cpu.cores_physical, &size, NULL, 0);
    
    size = sizeof(cpu.cores_logical);
    sysctlbyname("hw.logicalcpu", &cpu.cores_logical, &size, NULL, 0);
    
    // Get brand string
    size = 0;
    sysctlbyname("machdep.cpu.brand_string", NULL, &size, NULL, 0);
    if (size > 0) {
        std::vector<char> brand(size);
        sysctlbyname("machdep.cpu.brand_string", brand.data(), &size, NULL, 0);
        cpu.brand = std::string(brand.data());
    }
#endif

    // Default cache sizes (will be refined by platform-specific code)
    cpu.cache_l1_size = 32;     // KB
    cpu.cache_l2_size = 256;    // KB
    cpu.cache_l3_size = 8192;   // KB
    cpu.cache_line_size = UINT64COMPUTE_CACHE_LINE_SIZE;
}

#ifdef UINT64COMPUTE_ARCH_X86_64
void HardwareDetector::detect_x86_features() const {
    auto& cpu = system_info_.cpu;
    cpu.vendor = "Unknown";
    cpu.features = 0;
    
    uint32_t eax, ebx, ecx, edx;
    
    // Check if CPUID is available
    run_cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    if (eax == 0) return;
    
    // Get vendor string
    char vendor[13] = {0};
    *reinterpret_cast<uint32_t*>(vendor) = ebx;
    *reinterpret_cast<uint32_t*>(vendor + 4) = edx;
    *reinterpret_cast<uint32_t*>(vendor + 8) = ecx;
    cpu.vendor = vendor;
    
    // Feature detection from CPUID(1)
    run_cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    
    // Check features in EDX
    if (edx & (1 << 26)) cpu.features |= static_cast<uint32_t>(CPUFeature::SSE2);
    
    // Check features in ECX
    if (ecx & (1 << 19)) cpu.features |= static_cast<uint32_t>(CPUFeature::SSE41);
    if (ecx & (1 << 28)) cpu.features |= static_cast<uint32_t>(CPUFeature::AVX);
    if (ecx & (1 << 12)) cpu.features |= static_cast<uint32_t>(CPUFeature::FMA);
    
    // Extended features from CPUID(7,0)
    run_cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    if (ebx & (1 << 5))  cpu.features |= static_cast<uint32_t>(CPUFeature::AVX2);
    if (ebx & (1 << 16)) cpu.features |= static_cast<uint32_t>(CPUFeature::AVX512F);
    if (ebx & (1 << 30)) cpu.features |= static_cast<uint32_t>(CPUFeature::AVX512BW);
    if (ebx & (1 << 31)) cpu.features |= static_cast<uint32_t>(CPUFeature::AVX512VL);
    if (ebx & (1 << 3))  cpu.features |= static_cast<uint32_t>(CPUFeature::BMI1);
    if (ebx & (1 << 8))  cpu.features |= static_cast<uint32_t>(CPUFeature::BMI2);
    
    // Cache information from CPUID(4)
    for (int i = 0; i < 4; ++i) {
        run_cpuid(4, i, &eax, &ebx, &ecx, &edx);
        uint32_t cache_type = eax & 0x1F;
        if (cache_type == 0) break;
        
        uint32_t cache_level = (eax >> 5) & 0x7;
        uint32_t cache_size = ((ebx >> 22) + 1) * (((ebx >> 12) & 0x3FF) + 1) * 
                             ((ebx & 0xFFF) + 1) * (ecx + 1);
        cache_size /= 1024; // Convert to KB
        
        if (cache_level == 1) {
            cpu.cache_l1_size = cache_size;
        } else if (cache_level == 2) {
            cpu.cache_l2_size = cache_size;
        } else if (cache_level == 3) {
            cpu.cache_l3_size = cache_size;
        }
    }
}

void HardwareDetector::run_cpuid(uint32_t function, uint32_t subfunction, 
                                uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) const {
#if defined(__GNUC__) || defined(__clang__)
    __cpuid_count(function, subfunction, *eax, *ebx, *ecx, *edx);
#elif defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, function, subfunction);
    *eax = regs[0];
    *ebx = regs[1];
    *ecx = regs[2];
    *edx = regs[3];
#else
    *eax = *ebx = *ecx = *edx = 0;
#endif
}
#endif // UINT64COMPUTE_ARCH_X86_64

#ifdef UINT64COMPUTE_ARCH_ARM64
void HardwareDetector::detect_arm_features() const {
    auto& cpu = system_info_.cpu;
    cpu.vendor = "ARM";
    cpu.features = 0;
    
#ifdef UINT64COMPUTE_ARCH_ARM64
    // ARM64 always has NEON
    cpu.features |= static_cast<uint32_t>(CPUFeature::NEON);
#endif

#ifdef UINT64COMPUTE_PLATFORM_LINUX
    // Try to detect features from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("Features") != std::string::npos) {
                if (line.find("neon") != std::string::npos) {
                    cpu.features |= static_cast<uint32_t>(CPUFeature::NEON);
                }
                if (line.find("sve") != std::string::npos) {
                    cpu.features |= static_cast<uint32_t>(CPUFeature::SVE);
                }
                if (line.find("sve2") != std::string::npos) {
                    cpu.features |= static_cast<uint32_t>(CPUFeature::SVE2);
                }
                break;
            }
        }
    }
#endif
}
#endif // UINT64COMPUTE_ARCH_ARM64

void HardwareDetector::detect_memory_info() const {
    system_info_.memory = MemoryInfo{};
    auto& mem = system_info_.memory;
    
#ifdef UINT64COMPUTE_PLATFORM_LINUX
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        mem.total_ram = info.totalram * info.mem_unit;
        mem.available_ram = info.freeram * info.mem_unit;
    }
    
    // Check for NUMA
    mem.numa_nodes = 1; // Default to 1
    std::ifstream numa_nodes("/sys/devices/system/node/possible");
    if (numa_nodes.is_open()) {
        std::string line;
        if (std::getline(numa_nodes, line)) {
            // Parse range like "0-1" or single number "0"
            size_t dash = line.find('-');
            if (dash != std::string::npos) {
                int max_node = std::stoi(line.substr(dash + 1));
                mem.numa_nodes = max_node + 1;
            }
        }
    }
    
#elif defined(UINT64COMPUTE_PLATFORM_WINDOWS)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        mem.total_ram = status.ullTotalPhys;
        mem.available_ram = status.ullAvailPhys;
    }
    
#elif defined(UINT64COMPUTE_PLATFORM_MACOS)
    size_t size = sizeof(mem.total_ram);
    sysctlbyname("hw.memsize", &mem.total_ram, &size, NULL, 0);
    
    // Available RAM is harder to get on macOS
    vm_size_t page_size;
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t host_size = sizeof(vm_stats) / sizeof(natural_t);
    
    host_page_size(mach_host_self(), &page_size);
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, 
                         (host_info64_t)&vm_stats, &host_size) == KERN_SUCCESS) {
        mem.available_ram = (vm_stats.free_count + vm_stats.inactive_count) * page_size;
    }
#endif

    // Set feature flags
    mem.features = 0;
    mem.features |= static_cast<uint32_t>(MemoryFeature::ALIGNED_MALLOC);
    
    if (mem.numa_nodes > 1) {
        mem.features |= static_cast<uint32_t>(MemoryFeature::NUMA_AWARE);
    }
}

void HardwareDetector::detect_gpu_info() const {
    system_info_.gpu = GPUInfo{};
    auto& gpu = system_info_.gpu;
    gpu.available = false;
    
#ifdef UINT64COMPUTE_HAS_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0) {
        gpu.available = true;
        
        // Get properties of the first device
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            gpu.name = prop.name;
            gpu.memory_total = prop.totalGlobalMem;
            gpu.compute_units = prop.multiProcessorCount;
            gpu.max_threads_per_block = prop.maxThreadsPerBlock;
            
            // Get available memory
            size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                gpu.memory_available = free_mem;
            }
        }
        
        // Get driver version
        int driver_version;
        if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
            int major = driver_version / 1000;
            int minor = (driver_version % 1000) / 10;
            gpu.driver_version = std::to_string(major) + "." + std::to_string(minor);
        }
    }
#endif
}

void HardwareDetector::detect_parallel_features() const {
    system_info_.parallel_features = 0;
    
#ifdef UINT64COMPUTE_HAS_OPENMP
    system_info_.parallel_features |= static_cast<uint32_t>(ParallelFeature::OPENMP);
#endif

#ifdef UINT64COMPUTE_HAS_STD_THREAD
    system_info_.parallel_features |= static_cast<uint32_t>(ParallelFeature::STD_THREAD);
#endif

    if (system_info_.gpu.available) {
        system_info_.parallel_features |= static_cast<uint32_t>(ParallelFeature::CUDA);
    }
    
    // Detect OS information
#ifdef UINT64COMPUTE_PLATFORM_LINUX
    system_info_.os_name = "Linux";
#elif defined(UINT64COMPUTE_PLATFORM_WINDOWS)
    system_info_.os_name = "Windows";
#elif defined(UINT64COMPUTE_PLATFORM_MACOS)
    system_info_.os_name = "macOS";
#else
    system_info_.os_name = "Unknown";
#endif
}

// Query methods
bool HardwareDetector::has_feature(CPUFeature feature) const {
    detect_system();
    return (system_info_.cpu.features & static_cast<uint32_t>(feature)) != 0;
}

bool HardwareDetector::has_feature(MemoryFeature feature) const {
    detect_system();
    return (system_info_.memory.features & static_cast<uint32_t>(feature)) != 0;
}

bool HardwareDetector::has_feature(ParallelFeature feature) const {
    detect_system();
    return (system_info_.parallel_features & static_cast<uint32_t>(feature)) != 0;
}

bool HardwareDetector::is_x86() const {
#ifdef UINT64COMPUTE_ARCH_X86_64
    return true;
#else
    return false;
#endif
}

bool HardwareDetector::is_arm() const {
#ifdef UINT64COMPUTE_ARCH_ARM64
    return true;
#else
    return false;
#endif
}

bool HardwareDetector::is_64bit() const {
#if defined(UINT64COMPUTE_ARCH_X86_64) || defined(UINT64COMPUTE_ARCH_ARM64)
    return true;
#else
    return false;
#endif
}

int HardwareDetector::get_simd_width() const {
    detect_system();
    
    if (has_feature(CPUFeature::AVX512F)) return 512;
    if (has_feature(CPUFeature::AVX2) || has_feature(CPUFeature::AVX)) return 256;
    if (has_feature(CPUFeature::SSE2) || has_feature(CPUFeature::NEON)) return 128;
    
    return 64; // Scalar
}

int HardwareDetector::get_optimal_thread_count() const {
    detect_system();
    
    // Use physical cores, but cap at reasonable limit
    int cores = system_info_.cpu.cores_physical;
    return std::min(cores, 64); // Cap at 64 to avoid excessive overhead
}

size_t HardwareDetector::get_cache_line_size() const {
    detect_system();
    return system_info_.cpu.cache_line_size;
}

size_t HardwareDetector::get_optimal_alignment() const {
    return get_simd_width() / 8; // Convert bits to bytes
}

size_t HardwareDetector::get_optimal_block_size(size_t data_size) const {
    detect_system();
    
    // Target L2 cache size for block processing
    size_t l2_size = system_info_.cpu.cache_l2_size * 1024; // Convert KB to bytes
    size_t element_size = sizeof(uint64_t);
    
    // Use 1/4 of L2 cache to leave room for other data
    size_t optimal_elements = (l2_size / 4) / element_size;
    
    // Ensure it's a multiple of SIMD width
    size_t simd_elements = get_simd_width() / 64; // uint64_t elements per SIMD register
    optimal_elements = (optimal_elements / simd_elements) * simd_elements;
    
    return std::min(optimal_elements, data_size);
}

bool HardwareDetector::should_use_parallel(size_t data_size) const {
    // Use parallel processing if we have multiple cores and enough data
    return get_optimal_thread_count() > 1 && data_size >= 1000;
}

std::string HardwareDetector::get_system_summary() const {
    detect_system();
    
    std::ostringstream ss;
    ss << "CPU: " << system_info_.cpu.brand << " (" << system_info_.cpu.architecture << ")\n";
    ss << "Cores: " << system_info_.cpu.cores_physical << " physical, " 
       << system_info_.cpu.cores_logical << " logical\n";
    ss << "Memory: " << (system_info_.memory.total_ram / (1024*1024*1024)) << " GB total\n";
    ss << "SIMD: " << get_simd_width() << "-bit\n";
    
    if (system_info_.gpu.available) {
        ss << "GPU: " << system_info_.gpu.name << " ("
           << (system_info_.gpu.memory_total / (1024*1024*1024)) << " GB)\n";
    }
    
    return ss.str();
}

void HardwareDetector::print_system_info() const {
    std::cout << "=== UInt64Compute Hardware Detection ===" << std::endl;
    std::cout << get_system_summary();
    std::cout << "=========================================" << std::endl;
}

// ================================
// OptimizationSelector Implementation
// ================================

OptimizationSelector::OptimizationSelector() : detector_(HardwareDetector::instance()) {
    // Auto-tune thresholds based on hardware
    const auto& sys = detector_.detect_system();
    
    // Adjust parallel threshold based on core count
    if (sys.cpu.cores_physical >= 8) {
        parallel_threshold_ = 512;  // Lower threshold for many-core systems
    } else if (sys.cpu.cores_physical >= 4) {
        parallel_threshold_ = 1024;
    } else {
        parallel_threshold_ = 2048; // Higher threshold for few-core systems
    }
    
    // Adjust GPU threshold based on available GPU memory
    if (sys.gpu.available && sys.gpu.memory_total > 4ULL * 1024 * 1024 * 1024) {
        gpu_threshold_ = 50000;  // Lower threshold for high-memory GPUs
    }
}

OptimizationStrategy OptimizationSelector::select_strategy(size_t data_size, 
                                                          const std::string& operation) const {
    (void)operation; // Suppress unused parameter warning for now
    if (forced_strategy_ != OptimizationStrategy::ADAPTIVE) {
        return forced_strategy_;
    }
    
    // Strategy selection logic
    bool has_gpu = detector_.has_feature(ParallelFeature::CUDA);
    bool has_parallel = detector_.has_feature(ParallelFeature::OPENMP) || 
                       detector_.has_feature(ParallelFeature::STD_THREAD);
    bool has_simd = detector_.get_simd_width() > 64;
    
    // For very large datasets, prefer GPU if available
    if (has_gpu && data_size >= gpu_threshold_) {
        return OptimizationStrategy::GPU_OFFLOAD;
    }
    
    // For medium to large datasets, use SIMD + parallel
    if (has_simd && has_parallel && data_size >= parallel_threshold_) {
        return OptimizationStrategy::SIMD_PARALLEL;
    }
    
    // For medium datasets, use SIMD only
    if (has_simd && data_size >= simd_threshold_) {
        return OptimizationStrategy::SIMD_ONLY;
    }
    
    // For small datasets or limited hardware, use scalar
    return OptimizationStrategy::SCALAR;
}

bool OptimizationSelector::should_use_gpu(size_t data_size, const std::string& operation) const {
    auto strategy = select_strategy(data_size, operation);
    return strategy == OptimizationStrategy::GPU_OFFLOAD;
}

bool OptimizationSelector::should_use_parallel(size_t data_size, const std::string& operation) const {
    auto strategy = select_strategy(data_size, operation);
    return strategy == OptimizationStrategy::PARALLEL_ONLY || 
           strategy == OptimizationStrategy::SIMD_PARALLEL;
}

std::string OptimizationSelector::get_best_simd_implementation() const {
    if (detector_.has_feature(CPUFeature::AVX512F)) return "AVX512";
    if (detector_.has_feature(CPUFeature::AVX2)) return "AVX2";
    if (detector_.has_feature(CPUFeature::AVX)) return "AVX";
    if (detector_.has_feature(CPUFeature::SSE2)) return "SSE2";
    if (detector_.has_feature(CPUFeature::NEON)) return "NEON";
    if (detector_.has_feature(CPUFeature::SVE)) return "SVE";
    
    return "Scalar";
}

// ================================
// AlignedAllocator Implementation
// ================================

void* AlignedAllocator::allocate(size_t size, size_t alignment) {
    if (alignment == 0) {
        alignment = get_default_alignment();
    }
    
    size = round_up_to_alignment(size, alignment);
    
#ifdef UINT64COMPUTE_PLATFORM_WINDOWS
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void AlignedAllocator::deallocate(void* ptr) {
    if (ptr) {
#ifdef UINT64COMPUTE_PLATFORM_WINDOWS
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

size_t AlignedAllocator::get_default_alignment() {
    return HardwareDetector::instance().get_optimal_alignment();
}

uint64_t* AlignedAllocator::allocate_uint64_array(size_t count, size_t alignment) {
    return static_cast<uint64_t*>(allocate(count * sizeof(uint64_t), alignment));
}

void AlignedAllocator::deallocate_uint64_array(uint64_t* ptr) {
    deallocate(ptr);
}

bool AlignedAllocator::is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

size_t AlignedAllocator::round_up_to_alignment(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ================================
// GlobalConfig Implementation
// ================================

GlobalConfig& GlobalConfig::instance() {
    static GlobalConfig instance;
    return instance;
}

OptimizationSelector& GlobalConfig::get_optimizer() {
    if (!optimizer_) {
        optimizer_ = std::unique_ptr<OptimizationSelector>(new OptimizationSelector());
    }
    return *optimizer_;
}

const OptimizationSelector& GlobalConfig::get_optimizer() const {
    return const_cast<GlobalConfig*>(this)->get_optimizer();
}

// ================================
// Utility Functions Implementation
// ================================

namespace utils {

double measure_memory_bandwidth() {
    const size_t test_size = 64 * 1024 * 1024; // 64 MB
    const int iterations = 10;
    
    auto src = std::unique_ptr<uint8_t[]>(new uint8_t[test_size]{});
    auto dst = std::unique_ptr<uint8_t[]>(new uint8_t[test_size]{});
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::generate(src.get(), src.get() + test_size, [&gen]() {
        return static_cast<uint8_t>(gen());
    });
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::memcpy(dst.get(), src.get(), test_size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    double bytes_per_second = (test_size * iterations * 1e9) / duration.count();
    return bytes_per_second / (1024 * 1024); // Convert to MB/s
}

double get_time_nanoseconds() {
    auto now = std::chrono::high_resolution_clock::now();
    auto epoch = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
}

void warm_cache(const void* data, size_t size) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    volatile uint8_t dummy = 0;
    
    for (size_t i = 0; i < size; i += UINT64COMPUTE_CACHE_LINE_SIZE) {
        dummy += ptr[i];
    }
    
    (void)dummy; // Prevent optimization
}

} // namespace utils

} // namespace hardware
} // namespace uint64_compute