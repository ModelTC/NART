cmake_minimum_required(VERSION 3.10)
project(pynart_proto)

find_package(Protobuf REQUIRED)
file(GLOB NART_PROTOS *.proto)
protobuf_generate_python(PROTO_PY_SRCS ${NART_PROTOS})
# trigger proto generation process
add_custom_target(
    pynart_proto ALL 
    DEPENDS ${PROTO_PY_SRCS})
# message(${PROTO_PY_SRCS})
foreach(PB2_PY_SRC ${PROTO_PY_SRCS})
    file(RELATIVE_PATH PB2_PY_SRC_REL ${CMAKE_CURRENT_SOURCE_DIR} ${PB2_PY_SRC})
    list(APPEND PROTO_PY_SRCS_REL ${PB2_PY_SRC_REL})
endforeach(PB2_PY_SRC)

add_custom_command(
    TARGET pynart_proto 
    POST_BUILD
    COMMAND ln -s -f ${PROTO_PY_SRCS_REL} -t ./
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "create link to python pb2 sources generated from proto"
    VERBATIM
)
