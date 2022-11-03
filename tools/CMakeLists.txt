cmake_minimum_required(VERSION 3.10)
project(nart_tools)

set(PROMARK_SRCS "promark.cpp" "json11/json11.cpp")
add_executable(nart_promark ${PROMARK_SRCS})

target_link_libraries(nart_promark nart_case_shared)
target_link_libraries(nart_promark nart_case_include)
# TODO: eliminate hardcode art default module include
# target_include_directories(nart_promark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../art/default/)
# TODO: eliminate this hardcode for including art/src/parade_impl.h
target_include_directories(nart_promark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(nart_promark dl)

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_libraries(nart_promark pthread)
endif()

add_dependencies(nart_promark nart_case_all_shared)

install(
    TARGETS nart_promark
    RUNTIME DESTINATION bin
)
