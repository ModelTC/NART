PROJECT_NAME := module_cuda_quant
C_SRCS += $(wildcard src/*.c)
CXX_SRCS += $(wildcard src/*.cpp)
CUDA_SRCS += $(wildcard src/*.cu)

C_SRCS += $(wildcard src/op/*.c)
CXX_SRCS += $(wildcard src/op/*.cpp)
CUDA_SRCS += $(wildcard src/op/*.cu)

CUDA_SRCS += $(wildcard src/gemm/*.cu)
CUDA_SRCS += $(wildcard src/gemm/int8/*.cu)
CUDA_SRCS += $(wildcard src/gemm/int4/*.cu)

C_SRCS += $(wildcard src/utils/*.c)
CXX_SRCS += $(wildcard src/utils/*.cpp)
CUDA_SRCS += $(wildcard src/utils/*.cu)

MODULES := cuda_quant
#CUDA_DIR := /usr/local/cuda/
NVCCFLAGS := -lineinfo -gencode arch=compute_75,code=sm_75 -std=c++11 -Xptxas -O3, -maxrregcount=255 #
$(warning "[WARNING!!!!!]cuda quant only support sm_75")
export INCLUDE_DIRS
export LD_DIRS
export NVCCFLAGS
ALL_LIBS += -lcudart -lcudnn -lcublas

.PHONY: all

all: static_lib shared_lib

include makefiles/common.mk
