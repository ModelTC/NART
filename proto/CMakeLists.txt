cmake_minimum_required(VERSION 3.10)
project(nart_proto)

find_package(Protobuf REQUIRED)
file(GLOB NART_PROTOS *.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${NART_PROTOS})
add_library(nart_proto STATIC ${PROTO_SRCS})
target_include_directories(nart_proto PUBLIC ${Protobuf_INCLUDE_DIRS})
target_include_directories(nart_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})