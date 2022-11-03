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

if trt_ver.major >= 8:
    # upsample implementation of tensorrt >= 8.0
    class Resize(ONNXNodeParser, OpDesc, parser=real_parser, version=11):
        """Resize on trt-8+.
        NOTE: the combination `nearest + [align_corner, half_pixel]` cannot be aligned with onnxruntime.
        """

        coordination_transformation_dict = {
            "half_pixel": trt.ResizeCoordinateTransformation.HALF_PIXEL,
            "pytorch_half_pixel": trt.ResizeCoordinateTransformation.HALF_PIXEL,
            "align_corners": trt.ResizeCoordinateTransformation.ALIGN_CORNERS,
            "asymmetric": trt.ResizeCoordinateTransformation.ASYMMETRIC,
        }
        nearest_mode_dict = {
            "round_prefer_floor": trt.ResizeRoundMode.HALF_DOWN,
            "round_prefer_ceil": trt.ResizeRoundMode.HALF_UP,
            "floor": trt.ResizeRoundMode.FLOOR,
            "ceil": trt.ResizeRoundMode.CEIL,
        }

        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            mode = node.get_attribute_value("mode", "nearest")
            assert mode in mode_dict, "unsupported resize mode {0}".format(mode)
            mode = mode_dict[mode]
            # N is the tensor dimension without batch
            N = get_nb_dims(input) - 1

            # the coordination transform mode.
            coordination_transform_mode = node.get_attribute_value(
                "coordinate_transformation_mode"
            )
            assert (
                coordination_transform_mode in cls.coordination_transformation_dict
            ), "unsupported coordination transformation mode {0}".format(
                coordination_transform_mode
            )
            coordination_transform_mode = cls.coordination_transformation_dict.get(
                coordination_transform_mode, None
            )

            HW_START = 2
            specify_size = False
            # NOTE: we must calculate output shape before adding resize layer itself.
            if node.has_input(3):
                # output size specified.
                size = cls.get_const_array(node.input[3])
                assert size is not None
                in_shape = get_shape(input)
                part1 = in_shape.slice(slice(0, HW_START))
                part2 = ShapeTensor(list(size)[HW_START:])
                out_shape = concat_shape(part1, part2)
                specify_size = True
            else:
                # use scales value.
                scales = cls.get_const_array(node.input[2])
                assert (
                    scales is not None
                ), "tensorrt resize in scales mode requires the scales to be build time constant"
                assert all(
                    math.isclose(scales[dim], 1.0) for dim in range(0, HW_START)
                ), "cannot upsample batch and channel axis"

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            resize_layer.resize_mode = mode
            resize_layer.coordinate_transformation = coordination_transform_mode
            if specify_size:
                if out_shape.is_dynamic:
                    resize_layer.set_input(1, out_shape.shape_tensor)
                else:
                    resize_layer.shape = out_shape.shape_vals
            else:
                resize_layer.scales = list(scales)

            # the nearest mode.
            if node.get_attribute_value("mode", "nearest") == "nearest":
                nearest_mode = node.get_attribute_value("nearest_mode")
                nearest_mode = cls.nearest_mode_dict[nearest_mode]
                resize_layer.nearest_rounding = nearest_mode

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

elif trt_ver.major >= 7 and trt_ver.minor >= 1:
    # upsample implementation of tensorrt >= 7.1
    class Resize(ONNXNodeParser, OpDesc, parser=real_parser, version=11):
        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            get_attr = cls.def_get_attr(node)
            mode = get_attr("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"
            # N is the tensor dimension without batch
            N = get_nb_dims(input) - 1

            mode = mode_dict[mode]

            nearest_mode = node.get_attribute_value("nearest_mode")
            if (
                node.get_attribute_value("mode", "nearest") == "nearest"
                and nearest_mode != "floor"
            ):
                LOGGER.warning(
                    "tensorrt 7.x only support `floor` nearest_mode, got {0}".format(
                        nearest_mode
                    )
                )

            # the coordination transform mode.
            coordination_transform_mode = node.get_attribute_value(
                "coordinate_transformation_mode"
            )
            assert coordination_transform_mode in [
                "half_pixel",
                "align_corners",
                "pytorch_half_pixel",
            ], (
                "trt 7.x only support half_pixel or align_corners coordination transform,"
                " got {0}".format(coordination_transform_mode)
            )

            HW_START = 2
            if node.has_input(3):
                # output size specified.
                size = cls.get_const_array(node.input[3])
                assert size is not None
                in_shape = get_shape(input)
                part1 = in_shape.slice(slice(0, HW_START))
                part2 = ShapeTensor(list(size)[HW_START:])
                out_shape = concat_shape(part1, part2)
            else:
                # use scales value.
                scales = cls.get_const_array(node.input[2])
                assert (
                    scales is not None
                ), "tensorrt resize in scales mode requires the scales to be build time constant"
                assert all(
                    math.isclose(scales[dim], 1.0) for dim in range(0, HW_START)
                ), "cannot upsample batch and channel axis"
                # NOTE: The resize layer in tensorrt 7.x has some bug,
                # when using scales parameter, the result is not correct.
                # For example, upsampling [3, 3] to [6, 6], when Align_corners=True, the `scales of tensor` are 2*2,
                # while the `scales of image` should really be 5/2, but tensorrt mixes the two, and producing
                # incorrect result.
                in_shape = get_shape(input)
                part1 = in_shape.slice(slice(0, HW_START))
                part2 = in_shape.slice(slice(HW_START, None))
                if not part2.is_dynamic:
                    vals = [
                        x * scale
                        for x, scale in zip(part2.shape_vals, scales[HW_START:])
                    ]
                    assert len(vals) + HW_START == in_shape.nb_dim
                    part2 = ShapeTensor(vals)
                else:
                    scale = scales[HW_START]
                    for _scale in scales[HW_START + 1 :]:
                        assert _scale == scale
                    assert np.floor(scale) == scale
                    mul_op = network.add_elementwise(
                        part2.shape_tensor,
                        to_const_tensor(np.array([scale], dtype=np.int32)),
                        op=trt.ElementWiseOperation.PROD,
                    )
                    mul_op.name = f"{node.name}::prod"
                    part2 = ShapeTensor(
                        [-1] * len(scales[HW_START:]), mul_op.get_output(0)
                    )
                out_shape = concat_shape(part1, part2)

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            if not out_shape.is_dynamic:
                # static shape
                resize_layer.shape = out_shape.shape_vals
            else:
                resize_layer.set_input(1, out_shape.shape_tensor)

            resize_layer.resize_mode = mode
            if coordination_transform_mode == "align_corners":
                resize_layer.align_corners = True
            else:
                resize_layer.align_corners = False

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

else:
    # upsample implementation of tensorrt == 7.0.x
    class Resize(ONNXNodeParser, OpDesc, parser=TensorrtParser, version=11):
        @classmethod
        def _parse_(cls, node, network, itensor_by_name):
            input = itensor_by_name[node.input[0]]
            scales = cls.get_const_array(node.input[2])
            assert (
                scales is not None
            ), "tensorrt resize in scales mode requires the scales to be build time constant"
            get_attr = cls.def_get_attr(node)
            mode = get_attr("mode", "nearest")
            assert mode in mode_dict, "unsupported upsample mode"
            # N is the tensor dimension without batch
            N = get_nb_dims(input) - 1

            mode = mode_dict[mode]

            nearest_mode = node.get_attribute_value("nearest_mode")
            if (
                node.get_attribute_value("mode", "nearest") == "nearest"
                and nearest_mode != "floor"
            ):
                LOGGER.warning(
                    "tensorrt 7.x only support `floor` nearest_mode, got {0}".format(
                        nearest_mode
                    )
                )

            resize_layer = network.add_resize(input)
            resize_layer.name = node.name
            resize_layer.scales = list(scales)
            resize_layer.resize_mode = mode
            # nearest Upsample means align_corners=False
            coordination_transform_mode = node.get_attribute_value(
                "coordinate_transformation_mode"
            )
            assert coordination_transform_mode in ["half_pixel"], (
                "trt 7.0 only support half_pixel coordination transform,"
                " got {0}".format(coordination_transform_mode)
            )
            resize_layer.align_corners = False

            outputs = {node.output[0]: resize_layer.get_output(0)}
            return resize_layer, outputs

        @OpDesc.attr(OpDesc.AttrType.STRING)
        def mode():
            # NOTE: until tensorrt 7, IResizeLayer of linear mode is not correct, so only nearest mode is enabled.
            return mode == "nearest"
