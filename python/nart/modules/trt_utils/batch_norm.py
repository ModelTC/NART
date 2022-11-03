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


class BatchNormalization(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = cls.get_itensor_by_name(node.input[0])
        scale = cls.get_const_array(node.input[1])
        bias = cls.get_const_array(node.input[2])
        mean = cls.get_const_array(node.input[3])
        var = cls.get_const_array(node.input[4])
        assert all(
            x is not None for x in [scale, bias, mean, var]
        ), "scale, bias, mean and var should all be constant in inference"
        scale, bias, mean, var = [
            x.astype(np.float64) for x in [scale, bias, mean, var]
        ]
        get_attr = cls.def_get_attr(node)
        epsilon = get_attr("epsilon", 1e-5)

        assert (
            len(node.output) == 1
        ), "only one output is supported by tensorrt batch normalization"

        std_var = np.sqrt(var + epsilon)
        adjusted_scale = (scale / std_var).astype(np.float32)
        adjusted_bias = (bias - scale * mean / std_var).astype(np.float32)
        if get_nb_dims(ipt) < 4:
            # tensorrt scale layer requires input to have at least 4 dimensions,
            # expand dims of ipt to 4 if not.
            # nb_dims is the original number of dimensions.
            nb_dims = get_nb_dims(ipt)
            reshape0 = add_shuffle(
                ipt, ShapeTensor([0] * nb_dims + [1] * (4 - nb_dims))
            )
            ipt = reshape0.get_output(0)
        else:
            nb_dims = None
        scale_layer = network.add_scale(
            input=ipt,
            mode=trt.ScaleMode.CHANNEL,
            shift=adjusted_bias,
            scale=adjusted_scale,
        )
        scale_layer.name = node.name
        # channel axis is always 1 in onnx batch normalization.
        assert scale_layer.channel_axis == 1
        if nb_dims is None:
            outputs = {node.output[0]: scale_layer.get_output(0)}
            return scale_layer, outputs
        else:
            # need to reshape back to nb_dims
            output = scale_layer.get_output(0)
            reshape1 = add_shuffle(output, ShapeTensor([0] * nb_dims))
            return scale_layer, {node.output[0]: reshape1.get_output(0)}


class CaffeBatchNorm(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = itensor_by_name[node.input[0]]
        mean = cls.get_const_array(node.input[1])
        var = cls.get_const_array(node.input[2])
        mvf = cls.get_const_array(node.input[3]).item(0)

        assert all(
            x is not None for x in [mean, var, mvf]
        ), "mean and var of BatchNorm should both be constant in inference"
        mvf = 1 / mvf if mvf != 0 else 1
        mean, var = [x.astype(np.float64) for x in [mean, var]]
        mean = mean * mvf
        var = var * mvf
        get_attr = cls.def_get_attr(node)
        epsilon = get_attr("eps", 1e-5)

        assert (
            len(node.output) == 1
        ), "only one output is supported by tensorrt batch normalization"

        std_var = np.sqrt(var + epsilon)
        adjusted_scale = (1 / std_var).astype(np.float32)
        adjusted_bias = np.negative(mean / std_var).astype(np.float32)

        if get_nb_dims(ipt) < 4:
            # tensorrt scale layer requires input to have at least 4 dimensions,
            # expand dims of ipt to 4 if not.
            # nb_dims is the original number of dimensions.
            nb_dims = get_nb_dims(ipt)
            reshape0 = add_shuffle(
                ipt, ShapeTensor([0] * nb_dims + [1] * (4 - nb_dims))
            )
            ipt = reshape0.get_output(0)
        else:
            nb_dims = None

        scale_layer = network.add_scale(
            input=ipt,
            mode=trt.ScaleMode.CHANNEL,
            shift=adjusted_bias,
            scale=adjusted_scale,
        )
        scale_layer.name = node.name
        # channel axis is always 1 in onnx batch normalization.
        assert scale_layer.channel_axis == 1

        if nb_dims is None:
            outputs = {node.output[0]: scale_layer.get_output(0)}
            return scale_layer, outputs
        else:
            # need to reshape back to nb_dims
            output = scale_layer.get_output(0)
            reshape1 = add_shuffle(output, ShapeTensor([0] * nb_dims))
            return scale_layer, {node.output[0]: reshape1.get_output(0)}
