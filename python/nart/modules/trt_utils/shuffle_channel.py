# Copyright 2022 SenseTime Group Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Contains parser(s) for a ONNX op, different class may be defined based on the tensorrt version.
    NOTE: This file will not be imported by default, call trt_utils.init_parsers() to import.
          This also means, when adding a new parser file, you should add `import xxx` to trt_utils.init_parsers
"""
from ..tensorrt import ONNXNodeParser, TensorrtParser
from ...ops.op import OpDesc
from .parse_utils import (
    get_nb_dims,
    ShapeTensor,
    reduce_mul,
    concat_shape,
    concat_shapes,
    get_shape,
    add_shuffle,
    flatten_tensor,
    to_const_tensor,
)
from tensorrt import tensorrt as trt
import numpy as np
import logging

LOGGER = logging.getLogger("nart.modules.tensorrt")

real_parser = TensorrtParser.get_class()


class ShuffleChannel(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = cls.get_itensor_by_name(node.input[0])
        get_attr = cls.def_get_attr(node)
        group = get_attr("group")
        # first reshape
        data_shape = get_shape(data)
        # C = data_shape[1]

        split_shape = concat_shapes(
            [
                data_shape.slice(0),
                ShapeTensor(
                    [group, -1], to_const_tensor(np.array([group, -1], np.int32))
                ),
                data_shape.slice(slice(2, data_shape.nb_dim)),
            ]
        )
        reshape_layer1 = add_shuffle(data, split_shape)
        reshape_layer1.name = f"{node.name}::reshape1"

        # transpose
        data = reshape_layer1.get_output(0)
        transpose_layer = network.add_shuffle(data)
        transpose_layer.first_transpose = [0, 2, 1, 3, 4]
        transpose_layer.name = f"{node.name}::transpose"

        # second reshape
        data = transpose_layer.get_output(0)
        reshape_layer2 = add_shuffle(data, data_shape)  # simply reshape back
        reshape_layer2.name = f"{node.name}::reshape2"

        outputs = {node.output[0]: reshape_layer2.get_output(0)}
        return reshape_layer2, outputs
