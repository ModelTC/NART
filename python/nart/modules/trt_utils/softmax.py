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
)
from tensorrt import tensorrt as trt
import numpy as np
import logging

LOGGER = logging.getLogger("nart.modules.tensorrt")

real_parser = TensorrtParser.get_class()


class Softmax(ONNXNodeParser, OpDesc, op_type="Softmax", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        input = itensor_by_name[node.input[0]]
        get_attribute = cls.def_get_attr(node)
        axis = get_attribute("axis", 1)
        axis = axis % get_nb_dims(input)

        # softmax layer in tensorrt can only be done on one axis, so
        # if {axis} is not the last axis, flatten is needed
        need_flatten = axis != get_nb_dims(input) - 1
        if need_flatten:
            # first flatten input
            flattened = flatten_tensor(input, axis)

            softmax = network.add_softmax(input=flattened)
            softmax.name = node.name
            # softmax.axes is a bit flag, after flattening, always do softmax on axis 1.
            # the batch dimension is NOT EXCLUDED in EXPLICIT_BATCH mode.
            softmax.axes = 1 << 1

            # then reshape it back
            reshape = add_shuffle(softmax.get_output(0), get_shape(input))
            op = reshape
            outputs = {node.output[0]: reshape.get_output(0)}
        else:
            softmax_op = network.add_softmax(input=input)
            softmax_op.name = node.name
            softmax_op.axes = 1 << axis
            op = softmax_op
            outputs = {node.output[0]: softmax_op.get_output(0)}

        return op, outputs


class Softmax_13(
    ONNXNodeParser, OpDesc, parser=real_parser, op_type="Softmax", version=13
):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = cls.get_itensor_by_name(node.input[0])
        nb_dim = get_nb_dims(data)
        axis = node.get_attribute_value("axis", 1)
        axis = axis % nb_dim

        softmax_layer = network.add_softmax(input=data)
        softmax_layer.name = node.name
        softmax_layer.axes = 1 << axis
        op = softmax_layer
        outputs = {node.output[0]: softmax_layer.get_output(0)}

        return op, outputs


class CaffeSoftmax(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = itensor_by_name[node.input[0]]
        get_attribute = cls.def_get_attr(node)
        axis = get_attribute("axis", 1)

        softmax_layer = network.add_softmax(input=data)
        softmax_layer.name = node.name
        softmax_layer.axes = 1 << axis
        op = softmax_layer
        outputs = {node.output[0]: softmax_layer.get_output(0)}

        return op, outputs
