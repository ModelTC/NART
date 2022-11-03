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


class UnaryParser(ONNXNodeParser, is_abstract=True):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = cls.get_itensor_by_name(node.input[0])
        op = cls.get_unary_op()
        layer = network.add_unary(data, op)
        layer.name = node.name
        outputs = {node.output[0]: layer.get_output(0)}
        return layer, outputs

    @classmethod
    def get_unary_op(cls):
        raise NotImplementedError(f"{cls.__name__} does not override get_unary_op")


class Floor(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.FLOOR


class Ceil(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.FLOOR


class Abs(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.ABS


class Sqrt(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.SQRT


class Exp(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.EXP


class Not(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.NOT


class Log(UnaryParser, OpDesc, parser=real_parser):
    @classmethod
    def get_unary_op(cls):
        return trt.UnaryOperation.LOG
