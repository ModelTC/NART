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

.. _tutorial-switch-tensorrt:

tensorrt
==================

**概述**

NVIDIA tensorrt 7.0。
我们在nart.switch.tensorrt模块中，提供从caffe模型到nart.case.tensorrt模型的转化帮助。

**example**

简单用法(通过-b指定最大batch size，此时要求所有输入的batch size维度一致；所有计算使用fp32）：

.. literalinclude:: examples/tensorrt.sh
    :name: tensorrt_sh
    :language: bash

如果需要更细致的配置，则需要使用配置文件，此时需要在配置文件里指定详细尺寸，不能再用-b参数：

.. literalinclude:: examples/tensorrt2.sh
    :name: tensorrt_sh2
    :language: bash

**带动态尺寸的配置模板**

.. literalinclude:: examples/tensorrt_cfg.json
    :caption: tensorrt_config.json
    :name: tensorrt_config
    :language: json

**离线量化配置模板**

.. literalinclude:: examples/tensorrt_quant_cfg.json
    :caption: tensorrt_config_quant.json
    :name: tensorrt_config_quant
    :language: json