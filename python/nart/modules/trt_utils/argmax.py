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
from ...ops.op import OpDesc, Op
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


class ArgMaxBase(ONNXNodeParser, OpDesc, is_abstract=True, parser=real_parser):
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    @classmethod
    def _parse_(cls, node: Op, network, _):
        data = cls.get_itensor_by_name(node.input[0])
        axis = node.get_attribute_value("axis")
        keepdims = node.get_attribute_value("keepdims")

        topk_op = network.add_topk(data, op=trt.TopKOperation.MAX, k=1, axes=1 << axis)
        topk_op.name = node.name

        if keepdims:
            outputs = {node.output[0]: topk_op.get_output(1)}
        else:
            # if keepdims is false, we have to unsqueeze the output.
            indices = topk_op.get_output(1)
            shape = get_shape(indices)
            nb_dims = get_nb_dims(indices)
            parts = list()
            if axis != 0:
                parts.append(shape.slice(slice(0, axis)))
            if axis + 1 != nb_dims:
                parts.append(shape.slice(slice(axis + 1, nb_dims)))
            out_shape = concat_shapes(parts)
            reshape_layer = add_shuffle(indices, out_shape)
            reshape_layer.name = f"{node.name}_unsqueeze"
            outputs = {node.output[0]: reshape_layer.get_output(0)}

        return topk_op, outputs


class ArgMax(ArgMaxBase, OpDesc, parser=real_parser):
    pass


class ArgMax_11(ArgMaxBase, OpDesc, parser=real_parser, op_type="ArgMax", version=11):
    pass
