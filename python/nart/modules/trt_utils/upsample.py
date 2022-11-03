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
from ...core import Node
import math
from .environment import tensorrt_version
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
)
from tensorrt import tensorrt as trt
import numpy as np
import logging

LOGGER = logging.getLogger("nart.modules.tensorrt")


real_parser = TensorrtParser.get_class()

trt_ver = tensorrt_version()

mode_dict = {
    "nearest": trt.ResizeMode.NEAREST,
    "bilinear": trt.ResizeMode.LINEAR,
    "linear": trt.ResizeMode.LINEAR,
    "trilinear": trt.ResizeMode.LINEAR,
}
interp_dims_dict = {"bilinear": 2, "linear": 2, "trilinear": 3}

if trt_ver.major >= 8:
    # upsample implementation of tensorrt >= 8
    class Upsample(ONNXNodeParser, OpDesc, parser=real_parser):
        @classmethod
        def _parse_(cls, node: Node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            mode = node.get_attribute_value("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            mode = mode_dict[mode]
            resize_layer.resize_mode = mode

            HW_START = 2
            if node.has_input(1):
                LOGGER.info("using scales input of upsample to set")
                scales = cls.get_const_array(node.input[1])
                assert (
                    scales is not None
                ), "tensorrt resize in scales mode requires the scales to be build time constant"
                assert all(
                    math.isclose(scales[dim], 1.0) for dim in range(0, HW_START)
                ), "cannot upsample batch and channel axis"
                resize_layer.scales = list(scales)
            else:
                LOGGER.info("using height&with in upsample attributes")
                in_shape = get_shape(input)
                part1 = in_shape.slice(slice(0, HW_START))
                height = node.get_attribute_value("height")
                width = node.get_attribute_value("width")
                part2 = ShapeTensor([height, width])
                out_shape = concat_shape(part1, part2)

                if not out_shape.is_dynamic:
                    # static shape
                    resize_layer.shape = out_shape.shape_vals
                else:
                    resize_layer.set_input(1, out_shape.shape_tensor)
            # In the standard of nart, the coordinate transformation of linear and nearest mode is different.
            if mode == trt.ResizeMode.LINEAR:
                resize_layer.coordinate_transformation = (
                    trt.ResizeCoordinateTransformation.ALIGN_CORNERS
                )
            else:
                resize_layer.coordinate_transformation = (
                    trt.ResizeCoordinateTransformation.HALF_PIXEL
                )
                resize_layer.nearest_rounding = trt.ResizeRoundMode.HALF_UP

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

elif trt_ver.major >= 7 and trt_ver.minor >= 1:
    # upsample implementation of tensorrt >= 7.1
    class Upsample(ONNXNodeParser, OpDesc, parser=real_parser):
        @classmethod
        def _parse_(cls, node: Node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            mode = node.get_attribute_value("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"

            in_shape = get_shape(input)
            HW_START = 2
            part1 = in_shape.slice(slice(0, HW_START))
            if node.has_input(1):
                scales = cls.get_const_array(node.input[1])
                assert (
                    scales is not None
                ), "tensorrt resize in scales mode requires the scales to be build time constant"
                # N is the tensor dimension without batch
                N = get_nb_dims(input) - 1
                # check whether linear interpolation dimension is supported by tensorrt
                if mode in interp_dims_dict:
                    interp_dims = interp_dims_dict[mode]
                    assert all(
                        math.isclose(scales[dim], 1.0)
                        for dim in range(1, N + 1 - interp_dims)
                    ), f"scales for dimensions except the last {interp_dims} should be 1.0 for {mode}"

                # resize_layer.scales = list(scales)
                # NOTE: The resize layer in tensorrt 7.x has some bug,
                # when using scales parameter, the result is not correct.
                # For example, upsampling [3, 3] to [6, 6], when Align_corners=True, the `scales of tensor` are 2*2,
                # while the `scales of image` should really be 5/2, but tensorrt mixes the two, and producing
                # incorrect result.
                assert all(
                    math.isclose(scales[dim], 1.0) for dim in range(0, HW_START)
                ), "cannot upsample batch and channel axis"
                part2 = in_shape.slice(slice(HW_START, None))
                if not part2.is_dynamic:
                    vals = [
                        int(x * scale)
                        for x, scale in zip(part2.shape_vals, scales[HW_START:])
                    ]
                    assert len(vals) + HW_START == in_shape.nb_dim
                    part2 = ShapeTensor(vals)
                else:
                    raise NotImplementedError(
                        "upsampling input tensor with dynamic HW is not supported at present"
                    )
            else:
                LOGGER.info("[tensorrt] using height&with in upsample attributes")
                height = node.get_attribute_value("height")
                width = node.get_attribute_value("width")
                part2 = ShapeTensor([height, width])

            mode = mode_dict[mode]

            out_shape = concat_shape(part1, part2)

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name

            if not out_shape.is_dynamic:
                # static shape
                resize_layer.shape = out_shape.shape_vals
            else:
                resize_layer.set_input(1, out_shape.shape_tensor)
            resize_layer.resize_mode = mode
            # align_corners is True at linear mode, False at nearest mode.
            if mode == trt.ResizeMode.LINEAR:
                resize_layer.align_corners = True
            else:
                resize_layer.align_corners = False

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

else:
    # upsample implementation of tensorrt == 7.0.x
    class Upsample(ONNXNodeParser, OpDesc, parser=TensorrtParser):
        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            scales = cls.get_const_array(node.input[1])
            assert (
                scales is not None
            ), "tensorrt resize in scales mode requires the scales to be build time constant"
            get_attr = cls.def_get_attr(node)
            mode = get_attr("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"
            # N is the tensor dimension without batch
            N = get_nb_dims(input) - 1
            # check whether linear interpolation dimension is supported by tensorrt
            if mode in interp_dims_dict:
                interp_dims = interp_dims_dict[mode]
                assert all(
                    math.isclose(scales[dim], 1.0)
                    for dim in range(1, N + 1 - interp_dims)
                ), f"scales for dimensions except the last {interp_dims} should be 1.0 for {mode}"

            mode = mode_dict[mode]

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            resize_layer.scales = list(scales)
            resize_layer.resize_mode = mode
            # nearest Upsample means align_corners=False
            resize_layer.align_corners = False

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

        @OpDesc.attr(OpDesc.AttrType.STRING)
        def mode():
            # NOTE: until tensorrt 7.0, IResizeLayer of linear mode is not correct, so only nearest mode is enabled.
            return mode == "nearest"


if trt_ver.major >= 8:

    class DynamicUpsample(ONNXNodeParser, OpDesc, parser=real_parser):
        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            assert node.has_input(1)
            ref = cls.get_itensor_by_name(node.input[1])
            get_attr = cls.def_get_attr(node)

            mode = get_attr("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"
            mode = mode_dict[mode]

            # DynamicUpsample only changes the height and width, so we have to compose the output shape
            in_shape = get_shape(input)
            out_shape = get_shape(ref)
            out_shape = concat_shape(
                in_shape.slice(slice(0, 2)), out_shape.slice(slice(2, None))
            )

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            if not out_shape.is_dynamic:
                # static shape
                resize_layer.shape = out_shape.shape_vals
            else:
                resize_layer.set_input(1, out_shape.shape_tensor)
            resize_layer.resize_mode = mode
            # align_corners is True at linear mode, False at nearest mode.
            # align_corner is deleted in tensorrt 8.2, replace with coordinate_transformation
            if mode == trt.ResizeMode.LINEAR:
                resize_layer.coordinate_transformation = (
                    trt.ResizeCoordinateTransformation.ALIGN_CORNERS
                )
            else:
                pass

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

elif trt_ver.major >= 7 and trt_ver.minor >= 1:

    class DynamicUpsample(ONNXNodeParser, OpDesc, parser=real_parser):
        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            assert node.has_input(1)
            ref = cls.get_itensor_by_name(node.input[1])
            get_attr = cls.def_get_attr(node)

            mode = get_attr("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"
            mode = mode_dict[mode]

            # DynamicUpsample only changes the height and width, so we have to compose the output shape
            in_shape = get_shape(input)
            out_shape = get_shape(ref)
            out_shape = concat_shape(
                in_shape.slice(slice(0, 2)), out_shape.slice(slice(2, None))
            )

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            if not out_shape.is_dynamic:
                # static shape
                resize_layer.shape = out_shape.shape_vals
            else:
                resize_layer.set_input(1, out_shape.shape_tensor)
            resize_layer.resize_mode = mode
            # align_corners is True at linear mode, False at nearest mode.
            if mode == trt.ResizeMode.LINEAR:
                resize_layer.align_corners = True
            else:
                resize_layer.align_corners = False

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

else:

    class DynamicUpsample(ONNXNodeParser, OpDesc, parser=real_parser):
        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            assert node.has_input(1)
            ref = cls.get_itensor_by_name(node.input[1])
            get_attr = cls.def_get_attr(node)

            mode = get_attr("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"
            mode = mode_dict[mode]

            # DynamicUpsample only changes the height and width, so we have to compose the output shape
            in_shape = get_shape(input)
            out_shape = get_shape(ref)
            out_shape = concat_shape(
                in_shape.slice(slice(0, 2)), out_shape.slice(slice(2, None))
            )

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            if not out_shape.is_dynamic:
                # static shape
                resize_layer.shape = out_shape.shape_vals
            else:
                resize_layer.set_input(1, out_shape.shape_tensor)
            resize_layer.resize_mode = mode
            # nearest Upsample means align_corners=False
            resize_layer.align_corners = False

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

        @OpDesc.attr(OpDesc.AttrType.STRING)
        def mode():
            # NOTE: until tensorrt 7, IResizeLayer of linear mode is not correct, so only nearest mode is enabled.
            return mode == "nearest"
