add_executable(helloU64 ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(helloU64 PRIVATE u64comp)
message(STATUS "hello U64compute")
