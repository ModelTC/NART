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


class ConvTranspose(ONNXNodeParser, OpDesc, parser=real_parser):
    padding_mode_by_autopad = {
        "SAME_UPPER": trt.PaddingMode.SAME_UPPER,
        "SAME_LOWER": trt.PaddingMode.SAME_LOWER,
    }

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        input = itensor_by_name[node.input[0]]
        weight = cls.get_const_array(node.input[1])
        bias = None
        if len(node.input) >= 3:
            bias = cls.get_const_array(node.input[2])
        nb_spatial_dim = get_nb_dims(input) - 2
        # attributes
        get_attr = cls.def_get_attr(node)
        auto_pad = get_attr("auto_pad", "NOTSET")
        assert (
            get_attr("dilations", [1] * nb_spatial_dim) == [1] * nb_spatial_dim
        ), "tensorrt does not support dilation in deconvolution"
        group = get_attr("group", 1)
        kernel_shape = weight.shape[2:]
        kernel_shape = get_attr("kernel_shape", kernel_shape)
        assert (
            get_attr("output_padding", [0] * nb_spatial_dim) == [0] * nb_spatial_dim
        ), "tensorrt does not support output_padding"
        # assert get_attr('output_shape', None) is None, 'output_shape supportion in tensorrt is not implemented'
        pads = get_attr("pads", [0] * (2 * nb_spatial_dim))
        pre_padding = pads[:nb_spatial_dim]
        post_padding = pads[nb_spatial_dim:]
        strides = get_attr("strides", [1] * nb_spatial_dim)
        num_output_maps = weight.shape[1] * group

        if bias is not None:
            assert (
                bias.shape[0] == num_output_maps
            ), f"the shape of bias differs from number of output feature map, {bias.shape[0]} vs {num_output_maps}"
        # add layer and set attributes
        deconv_layer = network.add_deconvolution_nd(
            input, num_output_maps, kernel_shape, weight, bias
        )
        deconv_layer.name = node.name
        if auto_pad == "NOTSET":
            # use explicit padding
            deconv_layer.pre_padding = pre_padding
            deconv_layer.post_padding = post_padding
        elif auto_pad != "VALID":
            padding_mode = cls.padding_mode_by_autopad.get(auto_pad, None)
            assert padding_mode is not None, f"unhandled auto_pad mode: {auto_pad}"
            deconv_layer.padding_mode = padding_mode
        deconv_layer.num_groups = group
        deconv_layer.stride_nd = strides

        outputs = {node.output[0]: deconv_layer.get_output(0)}
        return deconv_layer, outputs

    @OpDesc.attr(OpDesc.AttrType.INTS)
    def dilations():
        return all(x == 1 for x in dilations)

    @OpDesc.attr(OpDesc.AttrType.INTS)
    def output_padding():
        return False

    @OpDesc.attr(OpDesc.AttrType.INTS)
    def output_shape():
        return False
