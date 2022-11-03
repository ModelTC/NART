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


class Cast(ONNXNodeParser, OpDesc, parser=real_parser):
    from ...ops import DataType

    type_dict = {
        DataType.FLOAT: trt.DataType.FLOAT,
        DataType.FLOAT16: trt.DataType.HALF,
        DataType.INT32: trt.DataType.INT32,
        DataType.INT8: trt.DataType.INT8,
        DataType.BOOL: trt.DataType.BOOL,
        DataType.INT64: trt.DataType.INT32,
    }

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = cls.get_itensor_by_name(node.input[0])
        get_attr = cls.def_get_attr(node)
        to = get_attr("to")
        from ...ops import DataType

        to = DataType(to)
        assert to in cls.type_dict, "unsupported data type"
        if to == DataType.INT64:
            LOGGER.warning(
                "tensorrt doesn't support int64, changed to int32 automatically"
            )

        identity_layer = network.add_identity(data)
        identity_layer.name = node.name
        dtype = Cast.type_dict[to]
        identity_layer.set_output_type(0, dtype)

        outputs = {node.output[0]: identity_layer.get_output(0)}
        return identity_layer, outputs

    @OpDesc.attr(OpDesc.AttrType.INT)
    def to():
        from ..ops import DataType

        return to in {
            DataType.FLOAT,
            DataType.FLOAT16,
            DataType.INT32,
            DataType.INT8,
            DataType.BOOL,
            DataType.INT64,
        }
        # return True
