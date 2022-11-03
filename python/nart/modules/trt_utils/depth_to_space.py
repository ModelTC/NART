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
from .environment import ParserContext
from numpy.core.shape_base import block
from ..tensorrt import ONNXNodeParser, TensorrtParser
from ...ops.op import OpDesc, Shape
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


class DepthToSpace(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = cls.get_itensor_by_name(node.input[0])
        assert (
            data.dtype != trt.DataType.BOOL
        ), "TensorRT does not support BOOL input type for the DepthToSpace operator."
        get_attr = cls.def_get_attr(node)
        blocksize = get_attr("blocksize")
        mode = get_attr("mode")
        assert mode == "CRD"
        data_shape = get_shape(data)
        assert (
            len(data_shape.shape_vals) == 4
        ), "The input tensor must be in NCHW format."
        c, h, w = data_shape.shape_vals[-3:]
        c_2 = c // blocksize**2
        assert c % blocksize**2 == 0
        _is_dynamic = h == -1 and w == -1
        # dynamic shape
        if _is_dynamic:
            _first_shape = [c_2, blocksize, blocksize]
            hw_shape = data_shape.slice(slice(2, data_shape.nb_dim))
            first_shape = concat_shapes(
                [
                    data_shape.slice(0),
                    ShapeTensor(
                        _first_shape, to_const_tensor(np.array(_first_shape, np.int32))
                    ),
                    hw_shape,
                ]
            )
        else:
            _first_shape = [c_2, blocksize, blocksize, h, w]
            first_shape = concat_shape(
                data_shape.slice(0),
                ShapeTensor(
                    _first_shape, to_const_tensor(np.array(_first_shape, np.int32))
                ),
            )
        reshape_layer1 = add_shuffle(data, first_shape)
        reshape_layer1.second_transpose = [0, 1, 4, 2, 5, 3]
        reshape_layer1.name = f"{node.name}::reshape1"

        data = reshape_layer1.get_output(0)
        if _is_dynamic:
            _second_shape = [c_2]
            mul_op = network.add_elementwise(
                hw_shape.shape_tensor,
                to_const_tensor(np.array([blocksize], np.int32)),
                trt.ElementWiseOperation.PROD,
            )
            mul_op.name = f"{node.name}::mul"
            before_shape = concat_shape(
                data_shape.slice(0),
                ShapeTensor(
                    _second_shape, to_const_tensor(np.array(_second_shape, np.int32))
                ),
            )
            concat_output = concat_shape(
                before_shape.shape_tensor, mul_op.get_output(0)
            )
            reshape_layer2 = add_shuffle(data, concat_output)
        else:
            _second_shape = [c_2, h * blocksize, w * blocksize]
            second_shape = concat_shape(
                data_shape.slice(0),
                ShapeTensor(
                    _second_shape, to_const_tensor(np.array(_second_shape, np.int32))
                ),
            )
            reshape_layer2 = add_shuffle(data, second_shape)
        reshape_layer2.name = f"{node.name}::reshape2"

        if _is_dynamic:
            shuffle_layer = network.add_shuffle(reshape_layer2.get_output(0))
            shuffle_layer.reshape_dims = (
                data_shape.shape_vals[:1] + _second_shape + [0, 0]
            )
            shuffle_layer.name = f"{node.name}::shuffle"
            outputs = {node.output[0]: shuffle_layer.get_output(0)}
            return shuffle_layer, outputs

        outputs = {node.output[0]: reshape_layer2.get_output(0)}
        return reshape_layer2, outputs
