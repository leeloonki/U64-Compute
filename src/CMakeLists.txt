add_subdirectory(hw)
add_subdirectory(op)
add_subdirectory(np)

add_library(u64comp INTERFACE)
target_include_directories(u64comp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(u64comp INTERFACE hw np op)
