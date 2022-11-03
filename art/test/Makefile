PROJECT_NAME := test
C_SRCS += $(wildcard *.c)
ART_ROOT ?= ..

ALL_LIBS += -lart -lm
BUILD_DIR ?= build

INCLUDE_DIRS += ../include ../modules/default/include
LD_DIRS += ../lib

.PHONY: test run_test
test: bin
	ln -sf $(BUILD_DIR)/bin/$(EXECUTABLE) ./test

include ../makefiles/common.mk

.DEFAULT_GOAL := test
