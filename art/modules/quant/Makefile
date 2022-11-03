PROJECT_NAME := module_quant
C_SRCS += $(wildcard src/*.c)
CXX_SRCS += $(wildcard src/*.cpp)

C_SRCS += $(wildcard src/op/*.c)
C_SRCS += $(wildcard src/utils/*.c)
C_SRCS += $(shell find src/gemm/ -name *.c)
CXX_SRCS += $(wildcard src/op/*.cpp)

MODULES := quant
#CUDA_DIR := /usr/local/cuda/

#ALL_LIBS += -lcudart

.PHONY: all

all: static_lib shared_lib

.SECONDEXPANSION:
.PHONY:
print-%:
	@echo $($*)

include $(ART_ROOT)/makefiles/common.mk
