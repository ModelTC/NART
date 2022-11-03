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
    cast_to,
)
from tensorrt import tensorrt as trt

import logging

logger = logging.getLogger("nart.modules.tensorrt")

real_parser = TensorrtParser.get_class()


class Gather(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = cls.get_itensor_by_name(node.input[0])
        indices = cls.get_itensor_by_name(node.input[1])
        if indices.dtype != trt.DataType.INT32:
            logger.warning(
                f"expecting the input indices tensor of gather layer to be INT32, "
                f"got {indices.dtype}, cast added automatically"
            )
            indices = cast_to(indices, trt.DataType.INT32)
        axis = node.get_attribute_value("axis")
        gather_op = network.add_gather(ipt, indices, axis)
        gather_op.name = node.name

        outputs = {node.output[0]: gather_op.get_output(0)}
        return gather_op, outputs
