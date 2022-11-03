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


class Reshape(ONNXNodeParser, OpDesc, layertype="Reshape", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        input = cls.get_itensor_by_name(node.input[0])
        get_attr = cls.def_get_attr(node)
        # todo: maybe infer that the shape is output of Constant?
        if len(node.input) >= 2:
            shape = cls.get_const_array(node.input[1])
            if shape is not None:
                shape = list(shape)
            else:
                shape = itensor_by_name[node.input[1]]
        else:
            shape = get_attr("shape", "ints")
        reshape = network.add_shuffle(input)
        reshape.name = node.name
        # reshape.name = node.name
        if isinstance(shape, trt.ITensor):
            # use dynamic reshape
            reshape.set_input(1, shape)
        else:
            # use static reshape
            reshape.reshape_dims = shape

        outputs = {node.output[0]: reshape.get_output(0)}
        return reshape, outputs


class Unsqueeze(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = cls.get_itensor_by_name(node.input[0])

        get_attr = cls.def_get_attr(node)
        axes = get_attr("axes", "ints")
        # the number of dimension of output tensor.
        out_dims = get_nb_dims(ipt) + len(axes)

        class Part:
            def __init__(self, type, start=None):
                self.type = type
                self.ones_cnt = 1 if type == "ones" else None
                self.range = [start, start + 1] if type == "copy" else None

        # the next unused axis in input.
        it = iter(range(get_nb_dims(ipt)))
        # the parts of output shape
        parts = []
        axes = [axis if axis >= 0 else axis + out_dims for axis in axes]
        for axis in range(out_dims):
            if axis in axes:
                # fill a 1
                if parts and parts[-1].type == "ones":
                    # last part is sequence of ones, append the one to that part
                    parts[-1].ones_cnt += 1
                else:
                    # last part is copied dimensions, create new part
                    parts.append(Part("ones"))
            else:
                # copy the dimension
                if parts and parts[-1].type == "copy":
                    parts[-1].range[1] = next(it) + 1
                else:
                    parts.append(Part("copy", next(it)))
        # covnert from parts to read parts
        input_shape = get_shape(ipt)
        parts = [
            ShapeTensor([1] * item.ones_cnt)
            if item.type == "ones"
            else input_shape.slice(slice(item.range[0], item.range[1]))
            for item in parts
        ]
        out_shape = concat_shapes(parts)
        # TensorRT shape tensor must be 0D or 1D
        # cf.: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#overview
        # 8.5. Execution Tensors vs. Shape Tensors
        if (
            out_shape.shape_vals == [1, 1]
            and list(ipt.shape) == [1]
            and not ipt.is_execution_tensor
        ):
            out_shape = ShapeTensor([1])
            LOGGER.warning(
                "Skip Unsqueeze op for converting a non-execution tensor shape "
                "from [1] to [1, 1]."
            )
            out_shape = ShapeTensor([1])
        reshape_op = add_shuffle(ipt, out_shape)
        reshape_op.name = node.name

        outputs = {node.output[0]: reshape_op.get_output(0)}
        return reshape_op, outputs


class Squeeze(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = cls.get_itensor_by_name(node.input[0])
        graph = node.owning_graph
        ipt_shape = graph.get_tensor_shape(node.input[0])
        axes = [idx for idx, axis in enumerate(ipt_shape) if axis == 1]
        axes = node.get_attribute_value("axes", axes)
        ipt_shape = get_shape(ipt)
        shape_vals = ipt_shape.shape_vals
        nb_dims = get_nb_dims(ipt)
        axes = [x if x >= 0 else x + nb_dims for x in axes]
        # check the squeezed axes
        squeezed_dims = [shape_vals[axis] for axis in axes]
        if any(x > 0 and x != 1 for x in squeezed_dims):
            LOGGER.error(
                f"{node.name} tries to squeeze the {axes} axes of input whose shape is {shape_vals}"
            )
        kept_axes = [axis for axis in range(nb_dims) if axis not in axes]
        output_shape = ipt_shape.gather(kept_axes)
        reshape_op = add_shuffle(ipt, output_shape)
        reshape_op.name = node.name

        outputs = {node.output[0]: reshape_op.get_output(0)}
        return reshape_op, outputs


class Flatten(ONNXNodeParser, OpDesc, parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        data = cls.get_itensor_by_name(node.input[0])
        shape = get_shape(data)
        get_attr = cls.def_get_attr(node)
        axis = get_attr("axis", 1)

        d0 = reduce_mul(shape.slice(slice(axis)))
        d1 = reduce_mul(shape.slice(slice(axis, None)))
        layer = add_shuffle(data, concat_shape(d0, d1))
        layer.name = node.name

        outputs = {node.output[0]: layer.get_output(0)}
        return layer, outputs
