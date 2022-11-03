ifneq ($(CROSS_COMPILE),)
CC := $(CROSS_COMPILE)gcc
CXX := $(CROSS_COMPILE)g++
AR := $(CROSS_COMPILE)ar
LD := $(CROSS_COMPILE)ld
DUMPMACHINE ?= $(shell $(CC) -dumpmachine)
SYSTEM_NAME ?= $(word 2,$(subst -, ,$(DUMPMACHINE)))
SYSTEM_PROCESSOR ?= $(word 1,$(subst -, ,$(DUMPMACHINE)))
ifneq ($(SYSROOT),)
CFLAGS += --sysroot=${SYSROOT}
endif
endif
