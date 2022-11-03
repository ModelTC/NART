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

# element wise operations


class ElementWiseParser(ONNXNodeParser, is_abstract=True):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        input0 = node.input[0]
        input0 = cls.get_itensor_by_name(input0)
        input1 = node.input[1]
        input1 = cls.get_itensor_by_name(input1)
        # assert get_nb_dims(input0) == get_nb_dims(input1), 'tensorrt requires inputs of elementwise operation have same number of dimensions'
        if cls.get_operation() in [trt.ElementWiseOperation.AND]:
            ipt_cast_layer = network.add_identity(input0)
            ipt_cast_layer.set_output_type(0, trt.DataType.BOOL)
            input0 = ipt_cast_layer.get_output(0)

            ipt_cast_layer = network.add_identity(input1)
            ipt_cast_layer.set_output_type(0, trt.DataType.BOOL)
            input1 = ipt_cast_layer.get_output(0)

        layer = network.add_elementwise(
            input1=input0, input2=input1, op=cls.get_operation()
        )
        layer.name = node.name

        cast_layer = None
        if cls.get_operation() in [
            trt.ElementWiseOperation.GREATER,
            trt.ElementWiseOperation.LESS,
            trt.ElementWiseOperation.AND,
        ]:
            cast_layer = network.add_identity(layer.get_output(0))
            cast_layer.name = node.name + "::cast"
            cast_layer.set_output_type(0, trt.DataType.FLOAT)

        outputs = {
            node.output[0]: cast_layer.get_output(0)
            if cast_layer
            else layer.get_output(0)
        }
        return layer, outputs

    @classmethod
    def get_operation(cls):
        raise RuntimeError("get_operation should be overridden")


class Mul(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.PROD


class Add(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.SUM


class Sub(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.SUB


class Max(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.MAX


class Div(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.DIV


class FloorDiv(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.FLOOR_DIV


class Greater(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.GREATER


class Pow(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.POW


class Less(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.LESS


class And(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.AND


class Min(ElementWiseParser, OpDesc, parser=real_parser):
    @classmethod
    def get_operation(cls):
        return trt.ElementWiseOperation.MIN
