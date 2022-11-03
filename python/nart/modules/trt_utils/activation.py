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

real_parser = TensorrtParser.get_class()


class ActivationParser(ONNXNodeParser, is_abstract=True):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        input = itensor_by_name[node.input[0]]
        act_op = network.add_activation(input=input, type=cls.get_activation_type())
        act_op.name = node.name
        alpha, beta = cls.get_alpha_beta(node)
        if alpha is not None:
            act_op.alpha = alpha
        if beta is not None:
            act_op.beta = beta
        outputs = {node.output[0]: act_op.get_output(0)}
        return act_op, outputs

    @classmethod
    def get_activation_type(cls):
        raise RuntimeError("get_activation_type should be overridden")

    @classmethod
    def get_alpha_beta(cls, node):
        """derivate classes should override this method to return activation parameter alpha and beta"""
        return None, None


class Relu(ActivationParser, OpDesc, layertype="Relu", parser=real_parser):
    @classmethod
    def get_activation_type(cls):
        return trt.ActivationType.RELU


class Sigmoid(ActivationParser, OpDesc, parser=real_parser):
    @classmethod
    def get_activation_type(cls):
        return trt.ActivationType.SIGMOID


class Tanh(ActivationParser, OpDesc, parser=real_parser):
    @classmethod
    def get_activation_type(cls):
        return trt.ActivationType.TANH


class Clip(ActivationParser, OpDesc, parser=real_parser):
    @classmethod
    def get_activation_type(cls):
        return trt.ActivationType.CLIP

    @classmethod
    def get_alpha_beta(cls, node):
        if len(node.input) > 1:
            # Clip-11
            min_const = cls.get_const_array(node.input[1])
            max_const = cls.get_const_array(node.input[2])
            assert min_const is not None and (
                np.ndim(min_const) == 0 or len(min_const) == 1
            ), "clip min must be a const scalar"
            assert max_const is not None and (
                np.ndim(max_const) == 0 or len(max_const) == 1
            ), "clip max must be a const scalar"
            min_const, max_const = min_const.item(0), max_const.item(0)
        else:
            # Clip-6
            min_const = node.get_attribute_value("min", -3.402823e38)
            max_const = node.get_attribute_value("max", 3.402823e38)
        return min_const, max_const


class LeakyRelu(ActivationParser, OpDesc, parser=real_parser):
    @classmethod
    def get_activation_type(cls):
        return trt.ActivationType.LEAKY_RELU

    @classmethod
    def get_alpha_beta(cls, node):
        alpha = node.get_attribute_value("alpha", 0.01)
        return alpha, None


class Elu(ActivationParser, OpDesc, parser=real_parser):
    @classmethod
    def get_activation_type(cls):
        return trt.ActivationType.ELU

    @classmethod
    def get_alpha_beta(cls, node):
        alpha = node.get_attribute_value("alpha")
        return alpha, None
