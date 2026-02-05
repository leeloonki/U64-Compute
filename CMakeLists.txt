cmake_minimum_required(VERSION 3.18)
project(
    U64Compute
    VERSION 1.0.0
    LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "Debug"
        CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(src)
add_subdirectory(demo)
