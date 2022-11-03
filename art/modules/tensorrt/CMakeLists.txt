cmake_minimum_required(VERSION 3.10)
project(module_tensorrt)

include(module)

find_package(CUDA REQUIRED)

# set(CMAKE_CXX_STANDARD 11)
file(GLOB_RECURSE C_SRCS src/*.c)
file(GLOB_RECURSE CXX_SRCS src/*.cpp)

if (NOT TENSORRT_ROOT_DIR)
    set(TENSORRT_ROOT_DIR "/usr/local/cuda/")
endif()
set(TENSORRT_ROOT_DIR ${TENSORRT_ROOT_DIR} CACHE PATH "the path of tensorrt root directory" FORCE)

nart_case_module_add(
    CASE_MODULE tensorrt
    SRC ${C_SRCS} ${CXX_SRCS}
    INCLUDE ${CUDA_TOOLKIT_INCLUDE} include
    LINK cudart cudnn nvinfer
)

target_include_directories(nart_case_module_tensorrt_static SYSTEM PRIVATE ${TENSORRT_ROOT_DIR}/include)
target_include_directories(nart_case_module_tensorrt_shared SYSTEM PRIVATE ${TENSORRT_ROOT_DIR}/include)
target_link_directories(nart_case_module_tensorrt_static PUBLIC ${TENSORRT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
target_link_directories(nart_case_module_tensorrt_shared PUBLIC ${TENSORRT_ROOT_DIR}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
