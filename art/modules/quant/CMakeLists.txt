cmake_minimum_required(VERSION 3.10)
project(module_quant)

include(module)
set(CMAKE_CXX_STANDARD 11)
file(GLOB_RECURSE C_SRC src/*.c)
file(GLOB_RECURSE CXX_SRC src/*.cpp)

nart_case_module_add(
    CASE_MODULE quant
    SRC ${C_SRC} ${CXX_SRC}
    INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
