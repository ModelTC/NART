cmake_minimum_required(VERSION 3.13)
project(pynart)

# the python executable
if (NOT PYTHON_EXE)
    set(PYTHON_EXE "python3")
endif()
set(PYTHON_EXE ${PYTHON_EXE} CACHE STRING "the python executable to which to build with")

# install dependencies first
message(STATUS "Installing requirements" )
execute_process(
    COMMAND ${PYTHON_EXE} -m pip install --user -r requirements.txt
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE result)
if (NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Cannot install requirements")
endif()

# proto
add_subdirectory(nart/proto/)

file(GLOB_RECURSE _PYNART_C_SRCS src/nart/*.c)
list(APPEND PYNART_SRCS ${_PYNART_C_SRCS})
file(GLOB_RECURSE _PYNART_CXX_SRCS src/nart/*.cpp)
list(APPEND PYNART_SRCS ${_PYNART_CXX_SRCS})
add_library(pynart_shared SHARED ${PYNART_SRCS})

# the output library name of pynart
execute_process(COMMAND ${PYTHON_EXE}-config "--extension-suffix" OUTPUT_VARIABLE PYTHON_EXT_SUFFIX RESULT_VARIABLE result)
if (NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Cannot get python extension suffix: " ${PYTHON_EXT_SUFFIX})
endif()
string(STRIP ${PYTHON_EXT_SUFFIX} PYTHON_EXT_SUFFIX)
set_target_properties(pynart_shared PROPERTIES PREFIX "")
set_target_properties(pynart_shared PROPERTIES OUTPUT_NAME "_nart")
set_target_properties(pynart_shared PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})
# python library dependencies
# pybind11
execute_process(COMMAND ${PYTHON_EXE} "-m" "pybind11" "--includes" OUTPUT_VARIABLE PYBIND11_INCLUDES RESULT_VARIABLE result)
if (NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Cannot get pybind11 includes: " ${PYBIND11_INCLUDES})
endif()
string(STRIP ${PYBIND11_INCLUDES} PYBIND11_INCLUDES)
# NOTE: replace space to semiconlon (string -> list), this requires NO WHITE SPACE IN PATH.
string(REPLACE " " ";" PYBIND11_INCLUDES ${PYBIND11_INCLUDES})
target_compile_options(pynart_shared PRIVATE ${PYBIND11_INCLUDES})

# python library path
execute_process(COMMAND ${PYTHON_EXE}-config "--prefix" OUTPUT_VARIABLE PYTHON_PREFIX RESULT_VARIABLE result)
if (NOT ${result} EQUAL 0)
    message(FATAL_ERROR "Cannot get python prefix")
endif()
string(STRIP ${PYTHON_PREFIX} PYTHON_PREFIX)
target_link_directories(pynart_shared PUBLIC ${PYTHON_PREFIX}/lib)

# nart/art dependency
target_link_libraries(pynart_shared PUBLIC nart_switch_static)
target_link_libraries(pynart_shared PRIVATE nart_case_include)

# create link to generated shared library.
# TODO(hujian): Eliminate the hardcode for output library file name.
# get the relative path of _nart.so
file(RELATIVE_PATH
     PYNART_LIB_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/nart/
     ${CMAKE_CURRENT_BINARY_DIR}/_nart${PYTHON_EXT_SUFFIX})
add_custom_command(
    TARGET pynart_shared
    POST_BUILD
    COMMAND ln -s -f ${PYNART_LIB_PATH} ./
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nart/
    COMMENT "Create link to _nart shared library"
    VERBATIM
)

# pyart definition.
file(GLOB_RECURSE PYART_CXX_SRCS src/art/*.cpp)
add_library(pyart_shared SHARED ${PYART_CXX_SRCS})
# set output library name of pyart.
set_target_properties(pyart_shared PROPERTIES PREFIX "")
set_target_properties(pyart_shared PROPERTIES OUTPUT_NAME "_art")
set_target_properties(pyart_shared PROPERTIES SUFFIX ${PYTHON_EXT_SUFFIX})
# pybind11 dependency
target_compile_options(pyart_shared PRIVATE ${PYBIND11_INCLUDES})
# python library path
target_link_directories(pyart_shared PUBLIC ${PYTHON_PREFIX}/lib)
# dependencies of art & art modules.
# We use shared because the nart_case_shared are required by module_xxx_shared.
target_link_libraries(pyart_shared PRIVATE nart_case_include nart_case_shared)

# create link to generated shared library.
# TODO(hujian): Eliminate the hardcode for output library file name.
# get the relative path of _nart.so
file(RELATIVE_PATH
     PYART_LIB_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/nart/art/
     ${CMAKE_CURRENT_BINARY_DIR}/_art${PYTHON_EXT_SUFFIX})
add_custom_command(
    TARGET pyart_shared
    POST_BUILD
    COMMAND ln -s -f ${PYART_LIB_PATH} ./
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nart/art/
    COMMENT "Create link to _art shared library"
    VERBATIM
)
# link module libraries to nart/art/libmodules
foreach(module ${NART_CASE_MODULES})
    list(APPEND LIBS_TO_PACK nart_case_module_${module}_shared)
endforeach(module)
list(APPEND LIBS_TO_PACK nart_case_shared)
foreach(TARGET_NAME ${LIBS_TO_PACK})
    add_custom_command(
        TARGET pyart_shared
        POST_BUILD
        COMMAND ln -s -f $<TARGET_FILE:${TARGET_NAME}> ./libmodules/
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nart/art/
        COMMENT "Create link to ${TARGET_NAME} library in nart/art/"
        VERBATIM
    )
endforeach(TARGET_NAME)

# generate document
add_custom_target(
    document
    COMMAND make html
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs/
    DEPENDS pynart_shared
    COMMENT "Generate nart switch document"
)
