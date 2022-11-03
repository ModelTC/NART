PROJECT_NAME ?= $(error PROJECT_NAME must be specified)
ART_ROOT ?= $(error ART_ROOT must be specified)
BUILD_DIR ?= $(error BUILD_DIR must be specified)
BUILD_DIR := $(abspath $(BUILD_DIR))
PROTOC ?= protoc

STATIC_LIB ?= lib$(PROJECT_NAME).a
SHARED_LIB ?= lib$(PROJECT_NAME).so
EXECUTABLE ?= $(PROJECT_NAME)

ALL_OBJS += $(patsubst %, $(BUILD_DIR)/%.o, $(C_SRCS))
ALL_OBJS += $(patsubst %, $(BUILD_DIR)/%.o, $(CUDA_SRCS))
ALL_OBJS += $(patsubst %, $(BUILD_DIR)/%.o, $(CXX_SRCS))

ifneq ($(BUILD_TYPE),RELEASE)
CFLAGS += -g
NVCCFLAGS += -g
else
CFLAGS += -O3 -DNDEBUG
NVCCFLAGS += -O3 -DNDEBUG
endif

.PHONY: clean bin static_lib shared_lib install

include $(ART_ROOT)/makefiles/toolchain.mk

clean:
	@rm $(BUILD_DIR) $(CLEAN_OBJS) -rf

static_lib: $(BUILD_DIR)/lib/$(STATIC_LIB)

shared_lib: $(BUILD_DIR)/lib/$(SHARED_LIB)

bin: $(BUILD_DIR)/bin/$(EXECUTABLE)

INCLUDE_DIRS += $(ART_ROOT)/
LD_DIRS += $(ART_ROOT)/lib

ifneq (,$(findstring cuda, $(MODULES)))
CUDA_CALL_DIR := $(shell nvcc --version | grep release)
RELEASE_VERSION := $(strip $(shell echo $(CUDA_CALL_DIR) | awk -F ',' '{print $$2}'))
CUDA_VERSION := $(shell echo $(RELEASE_VERSION) | awk -F ' ' '{print $$2}')
MAJOR_VERSION := $(shell echo $(CUDA_VERSION) | awk -F '.' '{print $$1}')
MINOR_VERSION := $(shell echo $(CUDA_VERSION) | awk -F '.' '{print $$2}')

#NVCCFLAGS += -gencode arch=compute_53,code=sm_53 \
#             -gencode arch=compute_62,code=sm_62 \
#             -gencode arch=compute_60,code=sm_60 \
#             -gencode arch=compute_61,code=sm_61

#ifeq ($(MAJOR_VERSION),10)
#    NVCCFLAGS += -gencode arch=compute_75,code=sm_75 \
#                 -gencode arch=compute_72,code=sm_72 \
#                 -gencode arch=compute_70,code=sm_70
#else ifeq ($(MAJOR_VERSION),9)
#    NVCCFLAGS += -gencode arch=compute_70,code=sm70
#    ifeq ($(MINOR_VERSION),2)
#        NVCCFLAGS += -gencode arch=compute_72,code=sm72
#    endif
#endif

CUDA_DIR ?= /usr/local/cuda
NVCC ?= $(CUDA_DIR)/bin/nvcc
CUDART_LD_DIR ?= $(CUDA_DIR)/lib64
LD_DIRS += $(CUDART_LD_DIR)
INCLUDE_DIRS += $(CUDA_DIR)/include
NVCCFLAGS += -Wno-deprecated-gpu-targets
NVCCFLAGS += $(addprefix -L, $(LD_DIRS))
NVCCFLAGS += $(addprefix -I, $(INCLUDE_DIRS))
endif

CXXFLAGS += -std=c++11
LDFLAGS += $(addprefix -L, $(LD_DIRS))
CFLAGS += $(addprefix -I, $(INCLUDE_DIRS))

$(BUILD_DIR)/%.c.o: %.c
	@mkdir -p $(@D)
	@echo "CC	" $@
	@$(CC) $(CFLAGS) -fPIC -Wall -Wextra -c $< -MMD -MT $@ -o $@

ifneq (,$(findstring cuda, $(MODULES)))
$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(@D)
	@echo "NVCC	" $@
	@$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -Xcompiler -fpic -M -MT $@ $< > $(patsubst %.o, %.d, $@)
	@$(NVCC) -ccbin $(CXX) $(NVCCFLAGS) -Xcompiler -fpic,-Wall,-Wextra -c $< -o $@
endif

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(@D)
	@echo "CXX	" $@
	@$(CXX) $(CFLAGS) $(CXXFLAGS) -fPIC -Wall -Wextra -c $< -MMD -MT $@ -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	@mkdir -p $(@D)
	@echo "CXX	" $@
	@$(CXX) $(CFLAGS) $(CXXFLAGS) -fPIC -Wall -Wextra -c $< -MMD -MT $@ -o $@

$(BUILD_DIR)/lib/$(STATIC_LIB): $(ALL_OBJS)
	@mkdir -p $(@D)
	@echo "AR	" $@
	@$(AR) rs $@ $^

$(BUILD_DIR)/lib/$(SHARED_LIB): $(ALL_OBJS)
	@mkdir -p $(@D)
	@echo "LD	" $@
	@$(CC) -shared -o $@ $^ $(LDFLAGS) $(addprefix -Wl$(comma)-rpath$(comma), $(abspath $(LD_DIRS))) -Wl$(comma)-rpath$(comma)'$$ORIGIN' $(ALL_LIBS)

comma := ,
$(BUILD_DIR)/bin/$(EXECUTABLE): $(ALL_OBJS)
	@mkdir -p $(@D)
	@echo "LD	" $@
	@$(CXX) -o $@ $^ $(LDFLAGS) $(addprefix -Wl$(comma)-rpath$(comma), $(abspath $(LD_DIRS))) -Wl$(comma)-rpath$(comma)'$$ORIGIN' $(ALL_LIBS)

%.pb.cc %.pb.h: %.proto
	@echo PROTOC-CC "\t" $<
	@$(PROTOC) -I$(<D) --cpp_out=$(<D) $<

%_pb2.py: %.proto
	@echo PROTOC-PYTHON "\t" $<
	@$(PROTOC) -I$(<D) --python_out=$(<D) $<

-include $(shell find $(BUILD_DIR) -name "*.d" 2>/dev/null)

.DEFAULT_GOAL := all
