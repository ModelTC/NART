#!/bin/sh

SCRIPT=`readlink -f "$0"`
BASE_DIR=`dirname "$SCRIPT"`

if [ $# -ge 1 ]; then
  BUILD_DIR="$1"
else
  BUILD_DIR="build"
fi
if [ ! -f "$BUILD_DIR/cpu_example" ]; then
  echo "Usage: $0 <build_dir>"
  exit 1
fi

mkdir -p data/input data/output && \
  "$BASE_DIR/src/gen_data.py" && \
  "$BUILD_DIR/cpu_example" && \
  "$BASE_DIR/src/compare.py" && \
  echo "test passed!"
