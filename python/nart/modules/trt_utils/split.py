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

from itertools import accumulate

real_parser = TensorrtParser.get_class()


class Split(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        inp = cls.get_itensor_by_name(node.input[0])
        inp_shape = get_shape(inp)
        inp_dim = get_nb_dims(inp)
        axis = node.get_attribute_value("axis")
        axis = axis if axis >= 0 else axis + inp_dim
        splits = node.get_attribute_value("split")
        assert len(splits) == len(
            node.output
        ), "invalid Split node, the length of split differs from the number of output"
        shape_before = inp_shape.slice(slice(0, axis))
        shape_after = inp_shape.slice(slice(axis + 1, inp_dim))
        accum_size = [0] + list(accumulate(splits))
        outputs = dict()
        layers = []
        for out, split_size, start in zip(node.output, splits, accum_size):
            start_offset = [0] * inp_dim
            start_offset[axis] = start
            out_shape = concat_shapes(
                [shape_before, ShapeTensor([split_size]), shape_after]
            )
            slice_layer = network.add_slice(
                inp, start_offset, out_shape.shape_vals, stride=[1] * inp_dim
            )
            slice_layer.name = f"{node.name}::slice_{out}"
            if out_shape.is_dynamic:
                slice_layer.set_input(2, out_shape.shape_tensor)

            outputs[out] = slice_layer.get_output(0)
            layers.append(slice_layer)

        return layers, outputs
