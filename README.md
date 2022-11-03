# NART

## Introduction

> What is NART?

**NART** = **N**ART is not **A** **R**un**T**ime

NART is a deep learning inference framework.

NART supports multiple types of deep learning models and multiple back-ends.

## License

NART is licensed under the [Apache-2.0](LICENSE) license.

## Requirements

- pybind11
- numpy
- onnx>=1.4.0
- sphinx (for doc)
- sphinxcontrib-contentui (for doc)
- protobuf>=3.5.1 (python package and compiler)

### Model conversion

The below are versions supported by the model conversion module:

- PyTorch: 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8

Note: for Caffe target, some layers may be invalid for official Caffe,
which will be fixed in later releases.

## Build

```sh
# update git submodules
git submodule update --init --recursive

# install python requirements
pip install -r python/requirements.txt

cmake -B build \
    -DNART_CASE_MODULES='quant;cuda;tensorrt' \ # enable nart case modules in art/modules
    -DENABLE_NART_TOOLS=ON                      # enable nart tools (e.g. promark)

cmake --build build -j16
```

## Install

```sh
# set install prefix, change it to virtual env or some other desired path
export _NART_INSTALL_PREFIX=`pwd`/install

# ensure libart.so can be found by nart tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$_NART_INSTALL_PREFIX/lib

# install built modules
cmake --install build --prefix $_NART_INSTALL_PREFIX

# install python modules
cd python
python setup.py install --prefix $_NART_INSTALL_PREFIX
```

## Usage

### Model inference deployment

Please refer to [this example](./examples/00-model-conversion-and-inference).

### NART case runtime

Please refer to [this example](./examples/01-nart-case-cpu-example) and [`nart_promark`](./tools/promark.cpp).
