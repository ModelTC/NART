cmake_minimum_required(VERSION 3.10)
include(cmake/SomeTools.cmake)

project(nart_case)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)
set(NART_CASE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_compile_options(-fPIC)
# if we are building nart-case in nart-switch, add the -DBUILD_PROMARK flag, which defines BUILD_PROMARK macro.
if (${BUILD_IN_SWITCH})
    add_definitions("-DBUILD_PROMARK")
endif()

if (NOT NART_LOG_LEVEL)
    if (DEFINED CMAKE_BUILD_TYPE)
        string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE)
    else()
        set(BUILD_TYPE "not_specified")
    endif()
    if (x${BUILD_TYPE} STREQUAL "xdebug")
        message(STATUS "debug log enabled")
        set(NART_LOG_LEVEL "0")
    else()
        set(NART_LOG_LEVEL "1")
    endif()
endif() 
add_compile_definitions("NART_LOG_LEVEL=${NART_LOG_LEVEL}")

file(GLOB_RECURSE _NART_CASE_C_SRC src/*.c src/*.cpp)
list(APPEND NART_CASE_SRC ${_NART_CASE_C_SRC})
file(GLOB_RECURSE _NART_CASE_C_SRC modules/default/*.c)
file(GLOB_RECURSE _NART_CASE_CXX_SRC modules/default/*.cpp)
list(APPEND NART_CASE_SRC ${_NART_CASE_C_SRC} ${_NART_CASE_CXX_SRC})

add_library(nart_case_include INTERFACE)
set(NART_CASE_PUBLIC_INCLUDES include)
list(APPEND NART_CASE_PUBLIC_INCLUDES modules/default/include)

add_library(nart_case_static STATIC ${NART_CASE_SRC})
add_library(nart_case_shared SHARED ${NART_CASE_SRC})

set_property(TARGET nart_case_static PROPERTY C_STANDARD 99)
set_property(TARGET nart_case_shared PROPERTY C_STANDARD 99)

# An interface dummy library, which collects all libraries (nart_case and modules).
add_library(nart_case_all_static INTERFACE)
add_library(nart_case_all_shared INTERFACE)

target_link_libraries(nart_case_all_static INTERFACE nart_case_static)
target_link_libraries(nart_case_all_shared INTERFACE nart_case_shared)

set_target_properties(
    nart_case_static nart_case_shared
    PROPERTIES
    OUTPUT_NAME art
)

target_link_libraries(nart_case_static PUBLIC nart_case_include)
target_link_libraries(nart_case_shared PUBLIC nart_case_include)

set(NART_CASE_MODULES ${NART_CASE_MODULES} CACHE STRING "modules")
list(REMOVE_DUPLICATES NART_CASE_MODULES)
foreach(mod ${NART_CASE_MODULES})
    add_subdirectory(modules/${mod})

    target_link_libraries(nart_case_all_static INTERFACE nart_case_module_${mod}_static)
    target_link_libraries(nart_case_all_shared INTERFACE nart_case_module_${mod}_shared)

    list(APPEND NART_CASE_PUBLIC_INCLUDES modules/${mod}/include)

    # target_link_libraries(nart_case_static PUBLIC nart_case_module_${mod}_static)
    # target_link_libraries(nart_case_shared PRIVATE -Wl,--push-state,--whole-archive nart_case_module_${mod}_static -Wl,--pop-state)
endforeach(mod ${NART_CASE_MODULES})

# install
install(
    TARGETS nart_case_static nart_case_shared
    EXPORT nart_case_targets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
)

# headers
list(TRANSFORM NART_CASE_PUBLIC_INCLUDES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM NART_CASE_PUBLIC_INCLUDES APPEND "/")
target_include_directories(
    nart_case_include INTERFACE
    "$<BUILD_INTERFACE:${NART_CASE_PUBLIC_INCLUDES}>"
    $<INSTALL_INTERFACE:include>
)

install(TARGETS nart_case_include EXPORT nart_case_targets)
install(
    DIRECTORY ${NART_CASE_PUBLIC_INCLUDES}
    DESTINATION include
    COMPONENT Development
)

# export
install(
    EXPORT nart_case_targets
    DESTINATION lib/cmake/nart_case
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nart_case-config.cmake"
    INSTALL_DESTINATION lib/cmake/nart_case
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/nart_case-config.cmake"
    DESTINATION lib/cmake/nart_case
)
