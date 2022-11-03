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


class ReduceBase(ONNXNodeParser, is_abstract=True):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = cls.get_itensor_by_name(node.input[0])
        nb_dims = get_nb_dims(ipt)
        axes = node.get_attribute_value("axes", list(range(nb_dims)))
        axes_bitset = 0
        for axis in axes:
            axis = axis if axis >= 0 else axis + get_nb_dims(ipt)
            axes_bitset = axes_bitset | (1 << axis)
        keep_dims = bool(node.get_attribute_value("keepdims"))

        reduce_layer = network.add_reduce(
            ipt, cls._get_reduce_op(), axes_bitset, keep_dims
        )
        reduce_layer.name = node.name
        outputs = {node.output[0]: reduce_layer.get_output(0)}
        return reduce_layer, outputs

    @classmethod
    def _get_reduce_op(cls):
        raise NotImplementedError(f"{cls.__name__} should override _get_reduce_op")


class ReduceMean(ReduceBase, OpDesc, parser=real_parser):
    @classmethod
    def _get_reduce_op(cls):
        return trt.ReduceOperation.AVG


class ReduceMin(ReduceBase, OpDesc, parser=real_parser):
    @classmethod
    def _get_reduce_op(cls):
        return trt.ReduceOperation.MIN


class ReduceMax(ReduceBase, OpDesc, parser=real_parser):
    @classmethod
    def _get_reduce_op(cls):
        return trt.ReduceOperation.MAX


class ReduceSum(ReduceBase, OpDesc, parser=real_parser):
    @classmethod
    def _get_reduce_op(cls):
        return trt.ReduceOperation.SUM


class ReduceProd(ReduceBase, OpDesc, parser=real_parser):
    @classmethod
    def _get_reduce_op(cls):
        return trt.ReduceOperation.PROD
