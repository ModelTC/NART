PROJECT_NAME := module_tensorrt

TRT_HOME ?= /usr/local/cuda/

C_SRCS += $(wildcard src/*.c)
CXX_SRCS += $(wildcard src/*.cpp)

C_SRCS += $(wildcard src/op/*.c)
CXX_SRCS += $(wildcard src/op/*.cpp)
INCLUDE_DIRS += $(abspath include/)
INCLUDE_DIRS += /usr/local/cuda/include
INCLUDE_DIRS += $(TRT_HOME)/include

ifeq ($(TRT_PROFILING), TRUE)
CXXFLAGS += -D TRT_PROFILING
endif

MODULE := tensorrt
export INCLUDE_DIRS

LD_DIRS +=/usr/local/cuda/lib64/
LD_DIRS += $(TRT_HOME)/lib
export LD_DIRS


TENSORRT_LIBS = -lcudart -lcudnn -lnvinfer
ALL_LIBS += $(TENSORRT_LIBS)
.PHONY: all

all: static_lib shared_lib

.SECONDEXPANSION:
.PHONY:
print-%:
	@echo $($*)

include $(ART_ROOT)/makefiles/common.mk
