cmake_minimum_required(VERSION 3.13)
project(nart_switch)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
include(version)

# is this required?
add_compile_options(-fPIC)
set(CMAKE_CXX_STANDARD 11)
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

    # Add -O0 to remove optimizations when debugging
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
endif()
# needed by nart-case.
add_definitions(-DART_CONSTRAINT_WITH_NAME)
# let nart-case know we are building in nart-switch.
set(BUILD_IN_SWITCH TRUE)
add_subdirectory(art)

if (NOT DISABLE_SWITCH)
    # compile caffe.proto
    add_subdirectory(proto)

    file(GLOB_RECURSE _NART_SWITCH_C_SRCS src/*.c)
    list(APPEND NART_SWITCH_SRCS ${_NART_SWITCH_C_SRCS})
    file(GLOB_RECURSE _NART_SWITCH_CXX_SRCS src/*.cpp)
    list(APPEND NART_SWITCH_SRCS ${_NART_SWITCH_CXX_SRCS})

    add_library(nart_switch_static STATIC ${NART_SWITCH_SRCS})
    add_library(nart_switch_shared SHARED ${NART_SWITCH_SRCS})
    set_target_properties(
        nart_switch_static nart_switch_shared
        PROPERTIES
        OUTPUT_NAME nart
    )

    # sources uses #include "include/xxx.h", we we have to add the whole source dir to include dir.
    target_include_directories(nart_switch_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(nart_switch_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

    # link to nart_case interface
    target_link_libraries(nart_switch_static PRIVATE nart_case_include)
    target_link_libraries(nart_switch_shared PRIVATE nart_case_include)

    target_link_libraries(nart_switch_static PUBLIC nart_case_static nart_proto)
    # TODO: use static or shared dependency on nart_case?
    target_link_libraries(nart_switch_shared PUBLIC nart_case_static nart_proto)

    add_subdirectory(python)
endif()

set(ENABLE_NART_TOOLS ${ENABLE_NART_TOOLS} CACHE STRING "Enable building of nart tools (nart_promark)")
if (${ENABLE_NART_TOOLS})
    add_subdirectory(tools)
endif()
