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

real_parser = TensorrtParser.get_class()


class PoolingParser(ONNXNodeParser, is_abstract=True):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    @staticmethod
    def get_pooling_attr(get_attribute, nspatial_dim, node):
        # auto_pad not supported
        assert get_attribute("auto_pad", "NOTSET") == "NOTSET"
        # dilations is not supported by tensorrt
        assert (
            "dilations" not in node.attr_dict
            or get_attribute("dilations", [1] * nspatial_dim) == [1] * nspatial_dim
        )
        # ceil_mode determines padding_mode
        if "ceil_mode" in node.attr_dict:
            ceil_mode = get_attribute("ceil_mode", 0)
        else:
            ceil_mode = 0
        padding_mode = (
            trt.PaddingMode.EXPLICIT_ROUND_UP
            if ceil_mode == 1
            else trt.PaddingMode.EXPLICIT_ROUND_DOWN
        )
        pads = get_attribute("pads", [0] * (2 * nspatial_dim))
        pre_padding = pads[:nspatial_dim]
        post_padding = pads[nspatial_dim:]
        strides = get_attribute("strides", [1] * nspatial_dim)
        kernel_shape = get_attribute(
            "kernel_shape", [0] * nspatial_dim
        )  # the default value is only used to provide dim information
        if "count_include_pad" in node.attr_dict:
            count_include_pad = get_attribute("count_include_pad", 0)
        else:
            count_include_pad = 0
        return (
            kernel_shape,
            pre_padding,
            post_padding,
            padding_mode,
            strides,
            count_include_pad,
        )

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        input = itensor_by_name[node.input[0]]
        get_attribute = cls.def_get_attr(node)
        nspatial_dim = get_nb_dims(input) - 2
        assert nspatial_dim in [2, 3], "only 2D/3D pooling supported at present"
        pool_type = cls.get_pool_type()

        (
            kernel_shape,
            pre_padding,
            post_padding,
            padding_mode,
            strides,
            count_include_pad,
        ) = PoolingParser.get_pooling_attr(get_attribute, nspatial_dim, node)

        pool_op = network.add_pooling_nd(
            input, type=pool_type, window_size=kernel_shape
        )
        pool_op.name = node.name
        pool_op.pre_padding = pre_padding
        pool_op.post_padding = post_padding
        pool_op.padding_mode = padding_mode
        pool_op.stride_nd = strides

        # some special operations related to certain pooling type.
        # this is not a good design, those operations should be done in derivate classes.
        if pool_type == trt.PoolingType.AVERAGE:
            pool_op.average_count_excludes_padding = (
                True if count_include_pad == 0 else False
            )

        if pool_type == trt.PoolingType.MAX:
            assert (
                len(node.output) == 1 or not node.output[1]
            ), "Indices output is not supported"

        outputs = {node.output[0]: pool_op.get_output(0)}
        return pool_op, outputs

    @classmethod
    def get_pool_type(cls):
        raise RuntimeError("get_pool_type should be overridden")


class MaxPool(PoolingParser, OpDesc, parser=real_parser):
    @classmethod
    def get_pool_type(cls):
        return trt.PoolingType.MAX

    @OpDesc.attr(tp=OpDesc.AttrType.STRING, default="NOTSET")
    def auto_pad():
        return auto_pad == "NOTSET"


class MaxPool_10(
    PoolingParser, OpDesc, parser=real_parser, op_type="MaxPool", version=10
):
    @classmethod
    def get_pool_type(cls):
        return trt.PoolingType.MAX

    @OpDesc.attr(tp=OpDesc.AttrType.STRING, default="NOTSET")
    def auto_pad():
        return auto_pad == "NOTSET"


class AveragePool(PoolingParser, OpDesc, parser=real_parser):
    @classmethod
    def get_pool_type(cls):
        return trt.PoolingType.AVERAGE

    @OpDesc.attr(tp=OpDesc.AttrType.STRING, default="NOTSET")
    def auto_pad():
        return auto_pad == "NOTSET"


class AveragePool_10(
    PoolingParser, OpDesc, parser=real_parser, op_type="AveragePool", version=10
):
    @classmethod
    def get_pool_type(cls):
        return trt.PoolingType.AVERAGE

    @OpDesc.attr(tp=OpDesc.AttrType.STRING, default="NOTSET")
    def auto_pad():
        return auto_pad == "NOTSET"
