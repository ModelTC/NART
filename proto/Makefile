PROTOC ?= protoc

.PHONY: all all_cpp all_python

all: all_cpp all_python

all_cpp: $(patsubst %.proto, %.pb.cc, $(wildcard *.proto))

all_python: $(patsubst %.proto, %_pb2.py, $(wildcard *.proto))

%.pb.cc %.pb.h: %.proto
	@echo PROTOC-CC "\t" $<
	@$(PROTOC) --cpp_out=$(CURDIR) $<

%_pb2.py: %.proto
	@echo PROTOC-PYTHON "\t" $<
	@$(PROTOC) --python_out=$(CURDIR) $<
