PROJECT_NAME := module_cuda
C_SRCS += $(wildcard src/*.c)
CXX_SRCS += $(wildcard src/*.cpp)
CUDA_SRCS += $(wildcard src/*.cu)

C_SRCS += $(wildcard src/op/*.c)
CXX_SRCS += $(wildcard src/op/*.cpp)
CUDA_SRCS += $(wildcard src/op/*.cu)

MODULES := cuda
#CUDA_DIR := /usr/local/cuda/

export INCLUDE_DIRS
export LD_DIRS
CUDA_LIBS = -lcudart -lcudnn -lcublas
ALL_LIBS += $(CUDA_LIBS)

.PHONY: all

all: static_lib shared_lib

.SECONDEXPANSION:
.PHONY:
print-%:
	@echo $($*)

include $(ART_ROOT)/makefiles/common.mk
