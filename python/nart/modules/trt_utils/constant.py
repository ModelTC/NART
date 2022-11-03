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


class Constant(ONNXNodeParser, OpDesc, layertype="Constant", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        # get_attr = cls.def_get_attr(node)
        values = node.to_ndarray()
        if values.dtype == np.int64:
            # tensorrt doesn't support int64, cast int64 to int32.
            values = values.astype(np.int32)
        constant = network.add_constant(values.shape, values)
        constant.name = node.name

        # ctx = ParserContext.get_current()
        # ctx.constants[constant.get_output(0)] = values

        outputs = {node.output[0]: constant.get_output(0)}
        return constant, outputs
