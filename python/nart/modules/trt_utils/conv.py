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


class Conv(ONNXNodeParser, OpDesc, layertype="Conv", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        # get inputs and weights
        # input = itensor_by_name[node.input[0]]
        input = cls.get_itensor_by_name(node.input[0])
        # NOTE: what if the kernel or bias is not in weights?
        kernel = cls.get_const_array(node.input[1])
        bias = (
            cls.get_const_array(node.input[2])
            if (len(node.input) == 3 and node.input[2])
            else None
        )
        # get attributes
        get_attribute = cls.def_get_attr(node)
        wgt_shape = get_shape(kernel)
        num_out_maps = wgt_shape[0]

        nb_spatial_dim = get_nb_dims(input) - 2
        assert nb_spatial_dim != 1, "1D convolution not supported"
        assert nb_spatial_dim == len(wgt_shape) - 2
        if bias is not None:
            assert get_nb_dims(bias) == 1 and get_shape(bias)[0] == num_out_maps
        kernel_shape = get_attribute("kernel_shape", wgt_shape[2:])
        strides = get_attribute("strides", [1] * nb_spatial_dim)
        pads = get_attribute("pads", [0] * (2 * nb_spatial_dim))
        pre_padding = pads[:nb_spatial_dim]
        post_padding = pads[nb_spatial_dim:]
        dilation = get_attribute("dilations", [1] * nb_spatial_dim)
        # padding mode
        padding_mode = None
        if get_attribute("auto_pad", "NOTSET") != "NOTSET":
            auto_pad = get_attribute("auto_pad", "s")
            if auto_pad == "SAME_UPPER":
                padding_mode = trt.PaddingMode.SAME_UPPER
            elif auto_pad == "SAME_LOWER":
                padding_mode = trt.PaddingMode.SAME_LOWER
            elif auto_pad == "VALID":
                # do nothing
                pass
            else:
                raise RuntimeError("the auto_pad type if not handled")

        conv = network.add_convolution_nd(
            input=input,
            num_output_maps=num_out_maps,
            kernel_shape=kernel_shape,
            kernel=kernel,
            bias=bias,
        )
        conv.name = node.name
        conv.stride_nd = strides
        if padding_mode:
            # only set padding_mode when needed, since it supresses explicit padding
            conv.padding_mode = padding_mode
        else:
            conv.pre_padding = pre_padding
            conv.post_padding = post_padding
        conv.dilation_nd = dilation

        group = get_attribute("group", 1)
        try:
            # in trt8, relu + group conv will cause bug when check tensor shape
            # assert wgt_shape[1] * group == get_shape(input)[1]
            pass
        except AssertionError:
            LOGGER.info("can not check the convolution kernel channel dim size")
        conv.num_groups = group

        outputs = {node.output[0]: conv.get_output(0)}

        return conv, outputs
