file(GLOB ALL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(op STATIC ${ALL_FILES})
target_include_directories(op PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(STATUS "Project Source Dir: ${CMAKE_CURRENT_SOURCE_DIR}")
