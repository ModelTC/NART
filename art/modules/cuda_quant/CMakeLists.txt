cmake_minimum_required(VERSION 3.10)
project(module_cuda)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

include(module)

file(GLOB_RECURSE C_SRC src/*.c)
file(GLOB_RECURSE CXX_SRC src/*.cpp)
file(GLOB_RECURSE CU_SRC src/*.cu)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CUDA_NVCC_FLAGS
    -Xcompiler -fPIC -lineinfo
    -gencode arch=compute_75,code=sm_75
    -std=c++11 -Xptxas -O0,-v -maxrregcount=255)

if (NOT CUDART_LD_DIR)
    get_filename_component(CUDART_LD_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
endif()

cuda_add_library(nart_case_module_cuda_quant_cuda
    STATIC
    ${CU_SRC}
)
set_target_properties(
    nart_case_module_cuda_quant_cuda PROPERTIES OUTPUT_NAME art_module_cuda_quant_cuda)
target_link_libraries(nart_case_module_cuda_quant_cuda nart_case_include)
target_include_directories(nart_case_module_cuda_quant_cuda PRIVATE include)
target_compile_options(nart_case_module_cuda_quant_cuda PUBLIC -fPIC)
install(TARGETS nart_case_module_cuda_quant_cuda EXPORT nart_case_targets)

nart_case_module_add(
    CASE_MODULE cuda_quant
    SRC ${C_SRC} ${CXX_SRC}
    INCLUDE ${CUDA_TOOLKIT_INCLUDE} include
    LINK cudnn cublas
    LINK_STATIC nart_case_module_cuda_quant_cuda
    LINK_SHARED -Wl,--push-state,--whole-archive nart_case_module_cuda_quant_cuda -Wl,--pop-state
)

link_directories(${CUDART_LD_DIR})
target_link_directories(nart_case_module_cuda_quant_static PUBLIC ${CUDART_LD_DIR})
target_link_directories(nart_case_module_cuda_quant_shared PUBLIC ${CUDART_LD_DIR})
