..  Copyright 2022 SenseTime Group Limited

..  Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

..  http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

.. _tutorial-switch:

NART Switch
=============

switch是nart进行模型部署的统一入口，旨在对各平台所需参数进行抽象统一，并通过config中各后端独有的配置段进行差异化配置


Usage
-----------------

.. code-block:: bash

    python -m nart.switch --help

Example
-----------------

**onnx model**

使用onnx模型作为输入

.. code-block:: bash

    python -m nart.switch -t default --onnx model.onnx

**caffe model**

使用caffe模型作为输入

.. code-block:: bash

    python -m nart.switch -t default --prototxt caffe.prototxt --model caffe.caffemodel

Advanced Usage
-------------------

使用config来指定量化信息，网络拆分方式等配置

.. code-block:: bash

    python -m nart.switch -t default -c config.json --onnx model.onnx

**config.json**

.. code-block:: json

    {
        "默认网络类型":"[default]",
        "default_net_type_token":"default",

        "是否使用随机输入":"bool",
        "rand_input":false,

        "calibration数据数量": "int",
        "data_num": 100,

        "calibration数据路径": "dict",
        "input_path_map": {
            "blob_name": "path/to/blob_data/"
        },

        "设置某一层的后端":"dict",
        "set_net_type": {
            "layer_name": "module_name[default,cuda]"
        },

        "各个module的配置": "dict",
        "cuda": {
        }
    }

Platform
-----------------

.. toctree::
   :maxdepth: 3

   switch/tensorrt
   switch/cuda_quant

