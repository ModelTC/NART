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

.. _tutorial-switch-cuda_quant:

cuda_quant
==================

**概述**

nart.switch.cuda_quant模块能够将dirichlet导出的量化模型和参数，转换为nart.case.cuda_quant模型，目前支持4-bit、8-bit和4/8-bit混合模型。

nart.case.cuda_quant模块提供了Turing GPU(Tensor Core)上4-bit/8-bit常见算子的支持，部分shape下conv2d速度超过TensorRT 7.0。

**使用说明**

量化参数转换脚本：http://gitlab.bj.sensetime.com/source-admin/liubing_dirichlet_2_switch

通过该脚本，我们可以将dirichlet导出的量化参数转为nart.switch.cuda_quant可以解析的格式。示例如下，其中out_qparams.json即为switch接收的量化参数文件。

.. code-block:: bash

    python dirichlet_2_nart_json.py -i in_qparams.json -o out_qparams.json test.prototxt test.caffemodel

转换后，需要在config.json->"cuda_quant"->"qparams_file"关键字中定义该量化参数文件。

**能力**

目前cuda_quant支持4/8-bit的conv2d、relu、pool、eltwise sum和quantize、dequantize等layer。对于cuda_quant不支持的layer，用户可以在config.json的set_net_type中自定义后端，如使用cuda端运行。

另外，nart的默认layout是NCHW，cuda_quant模块中的算子都使用NHWC布局，NCHW到NHWC的转换由nart自动完成。

**example**

.. literalinclude:: examples/cuda_quant.sh
    :name: cuda_quant_sh
    :language: bash

**Sample cuda_quant_config.json**

.. literalinclude:: examples/cuda_quant_config.json
    :name: cuda_quant_config
    :language: json

**Sample for cuda_quant_qparams.json**

cuda_quant模块接收的量化参数示例如下所示。每组量化参数包括"alpha", "zero_point", "bit", "type"等四个字段。以"_param_1"和"_param_2"结尾的分别代表weight和bias的量化参数。

注：此处bias的alpha做了预处理：alpha = input_scale * weight_scale。这样在nart.case计算卷积时可以节省计算量。

.. literalinclude:: examples/cuda_quant_qparams.json
    :name: cuda_quant_qparams
    :language: json

