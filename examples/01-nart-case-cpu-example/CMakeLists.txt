cmake_minimum_required(VERSION 3.13)
project(cpu_example)

find_package(nart_case REQUIRED)

add_executable(cpu_example src/main.c)
target_link_libraries(cpu_example PRIVATE nart_case_static)
target_link_libraries(cpu_example INTERFACE nart_case_include)
