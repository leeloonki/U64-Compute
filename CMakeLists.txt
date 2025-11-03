cmake_minimum_required(VERSION 3.18)
project(U64Compute VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(ENABLE_CUDA "Enable CUDA support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")

    # Platform-specific flags for macOS
    if(APPLE)
        execute_process(
            COMMAND xcrun --show-sdk-path
            OUTPUT_VARIABLE MACOS_SDK_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(MACOS_SDK_PATH)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${MACOS_SDK_PATH}")
        endif()
    endif()
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA support (optional)
set(CUDA_ENABLED FALSE)
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_60")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            set(CUDA_ENABLED TRUE)
            add_compile_definitions(CUDA_ENABLED)
            message(STATUS "CUDA support enabled")
        endif()
    else()
        message(STATUS "CUDA compiler not found, building without CUDA support")
    endif()
else()
    message(STATUS "CUDA support disabled")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
file(GLOB CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# CUDA source files (if CUDA is enabled)
if(CUDA_ENABLED)
    file(GLOB CU_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
    set(ALL_SOURCES ${CPP_SOURCES} ${CU_SOURCES})
else()
    set(ALL_SOURCES ${CPP_SOURCES})
endif()

# Build static library
if(BUILD_STATIC_LIBS)
    add_library(uint64compute_static STATIC ${ALL_SOURCES})
    set_target_properties(uint64compute_static PROPERTIES OUTPUT_NAME uint64compute)
    target_link_libraries(uint64compute_static PUBLIC OpenMP::OpenMP_CXX)

    if(CUDA_ENABLED)
        target_link_libraries(uint64compute_static PUBLIC CUDA::cudart CUDA::cublas)
    endif()

    message(STATUS "Static library will be built")
endif()

# Build shared library
if(BUILD_SHARED_LIBS)
    add_library(uint64compute_shared SHARED ${ALL_SOURCES})
    set_target_properties(uint64compute_shared PROPERTIES OUTPUT_NAME uint64compute)
    target_link_libraries(uint64compute_shared PUBLIC OpenMP::OpenMP_CXX)

    if(CUDA_ENABLED)
        target_link_libraries(uint64compute_shared PUBLIC CUDA::cudart CUDA::cublas)
    endif()

    message(STATUS "Shared library will be built")
endif()

# Determine which library to use for executables
if(BUILD_STATIC_LIBS)
    set(LINK_LIBRARY uint64compute_static)
else()
    set(LINK_LIBRARY uint64compute_shared)
endif()

# Build functional test executable
if(BUILD_TESTS)
    add_executable(functional_test ${CMAKE_SOURCE_DIR}/functional_test.cpp)
    target_link_libraries(functional_test PRIVATE ${LINK_LIBRARY})

    if(CUDA_ENABLED)
        target_link_libraries(functional_test PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    message(STATUS "Functional test will be built")
endif()

# Build benchmark executable
if(BUILD_BENCHMARKS)
    add_executable(benchmark ${CMAKE_SOURCE_DIR}/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE ${LINK_LIBRARY})

    if(CUDA_ENABLED)
        target_link_libraries(benchmark PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    message(STATUS "Benchmark will be built")
endif()

# Install targets
if(BUILD_STATIC_LIBS)
    install(TARGETS uint64compute_static
            ARCHIVE DESTINATION lib)
endif()

if(BUILD_SHARED_LIBS)
    install(TARGETS uint64compute_shared
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endif()

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
        DESTINATION include/uint64_compute
        FILES_MATCHING PATTERN "*.h")

# Print build information
message(STATUS "=== UInt64 Compute Library Build Configuration ===")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "CUDA Support: ${CUDA_ENABLED}")
if(CUDA_ENABLED)
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
endif()
message(STATUS "Build Static Library: ${BUILD_STATIC_LIBS}")
message(STATUS "Build Shared Library: ${BUILD_SHARED_LIBS}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "================================================")

# Custom targets for convenience
add_custom_target(info
    COMMAND ${CMAKE_COMMAND} -E echo "Build configuration information printed above"
    COMMENT "Displaying build information"
)

# Test target
if(BUILD_TESTS)
    add_custom_target(test_run
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/functional_test
        DEPENDS functional_test
        COMMENT "Running functional tests"
    )
endif()

# Benchmark target
if(BUILD_BENCHMARKS)
    add_custom_target(benchmark_run
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/benchmark
        DEPENDS benchmark
        COMMENT "Running benchmarks"
    )
endif()
