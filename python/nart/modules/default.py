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

# -*- coding: utf-8 -*-
from numpy.core.fromnumeric import shape
from nart.passes.split_caffe_ops import SplitCaffeExp
from ..core.art import Proxy
from ..core.art import FakeParade
from ..core.art import FakeTensor
from ..core.art import Dtype
from .parser import Parser
from ..ops.op import Op, OpDesc
from ..utils.alter import current_context
from ..utils.onnx_utils import map_onnx_to_art_dtype

import inspect
import numpy
import logging
from functools import reduce
from onnx import helper

logger = logging.getLogger("nart.modules.default")

from ..ops import OpSetRegistry


class CaffeParser(Parser, module_name="default"):
    dispatch_dct = OpSetRegistry()

    @classmethod
    def get_support_layer_list(cls):
        return list(map(lambda x: x, CaffeParser.dispatch_dct))

    class LayerParser:
        @classmethod
        def __init_subclass__(
            cls, layertype=None, overwrite=False, is_abstract=False, **kwargs
        ):
            # call the Opdesc.__init_subclass__ first, since it will config domain & version
            super().__init_subclass__(**kwargs)
            if is_abstract:
                # abstract class, skip registration
                return
            if layertype is None:
                layertype = cls.__name__
            if cls.parse is CaffeParser.LayerParser.parse:
                raise
            domain = cls._domain
            version = cls._version
            key = OpSetRegistry.make_key(layertype, domain=domain, version=version)
            if overwrite is False and key in CaffeParser.dispatch_dct:
                raise
            CaffeParser.dispatch_dct[key] = cls

        @classmethod
        def parse(cls, parser, layerdef, inputs):
            pass

    def __init__(
        self,
        net,
        config={},
        data_generator=None,
        proxy=Proxy.ModuleProxy("default"),
        net_id=0,
    ):
        super(CaffeParser, self).__init__(net, config, data_generator)
        self.proxy = proxy

        if len(CaffeParser.dispatch_dct) is 0:
            raise RuntimeError(
                "No LayerParser registered, if you want to try out default LayerpParsers, call CaffeParser.register_defaults()"
            )

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        self.before_parse()
        for node in self.net.nodes:
            key = OpSetRegistry.make_key(node.op_type, node.domain, node.version)
            if key not in CaffeParser.dispatch_dct:
                raise RuntimeError(
                    f"LayerParser for type '{node.op_type}(domain={node.domain}, version={node.version})' not registered"
                )
        into_parade_outputs = {}
        for t in into_parade.tensors:
            if t.dtype == "float32":
                into_parade_outputs[t.name] = t
        dct_name_tensor = self.parse_input_dict()
        dct_name_tensor.update(into_parade_outputs)
        self.parse_layers(self.net, dct_name_tensor, into_parade)
        return into_parade

    def op_post(self, net, op):
        return op

    def parse_layers(self, net, dct_name_tensor, parade):
        for node in net.nodes:
            if node.op_type == "PRelu":
                # NOTE: this is a workaround for PRelu op.
                # TODO: update case op to meet onnx standard!!!
                slops_tensor_name = node.input[1]
                slops = dct_name_tensor[slops_tensor_name]
                slops_data = slops.data
                from functools import reduce
                import operator

                flatten_shape = [reduce(operator.mul, slops.shape.dims, 1)]
                import numpy as np

                slops = FakeTensor(
                    slops.dtype,
                    data=slops_data.flatten(),
                    shape=(flatten_shape, 1, 0),
                    name=slops_tensor_name,
                )
                dct_name_tensor[slops_tensor_name] = slops
            res = CaffeParser.dispatch_dct.find(
                node.op_type, node.domain, node.version
            ).parse(self, node, list(map(lambda x: dct_name_tensor[x], node.input)))
            if isinstance(res, (tuple, list)):
                op, outputs = res
            else:
                op = res
                outputs = node.output.copy()
            op = self.op_post(node, op)
            res_list = parade.append(op)
            assert len(res_list) == len(outputs)
            for name, tensor in zip(outputs, res_list):
                tensor.name = name
            dct_name_tensor.update({k: v for k, v in zip(outputs, res_list)})
        return parade

    def trans_onnx_to_caffe(self, node):
        layers = []
        alter_ctx = current_context()
        # logging.delogging.debug(f"[default] node name {node.name}")
        # for i in self.encoder.parse_layer(node):
        #     layers.append(i.params)
        parser_cls = alter_ctx.caffe_registry.find(
            node.op_type, domain=node.domain, version=node.version, default=None
        )
        if parser_cls is None:
            msg = (
                f"cannot parse op type `{node.op_type}(domain={node.domain}, "
                f"version={node.version})` by caffe alter"
            )
            logger.fatal(msg)
            raise RuntimeError(msg)
        parser = parser_cls(node)
        layers = parser.parse(alter_ctx.caffe_pb2)
        layers = layers if isinstance(layers, (list, tuple)) else [layers]
        return layers

    @classmethod
    def register_defaults(cls):
        pass

    def before_parse(self):
        for node in self.net.nodes:
            key = OpSetRegistry.make_key(node.op_type, node.domain, node.version)
            if key not in self.__class__.dispatch_dct:
                raise RuntimeError(
                    f"{self.__class__.__name__} for type '{node.op_type}("
                    f"domain={node.domain}, version={node.version})' not registered"
                )

    @classmethod
    def get_passes(cls):
        from .. import passes

        ret = [
            passes.RemoveCertainOp(["Dropout"]),
            passes.ConstantToInitializer(),
            passes.ConvertGlobalPool(),
            passes.SubToAdd(),
            # passes.ExtractEltwise(),
            passes.SoftmaxToCaffeSoftmax(),
            passes.SplitCaffeExp(),
            passes.SplitPixelShuffle(),
            passes.InsertUnsqueezeForBroadcastOp("MatMul"),
            passes.ConvertDualInputScale(),
            passes.ConstantToInitializer(),
            passes.SimplifyExpand(),
            passes.DeadCodeElimination(),
        ]
        return ret


def split_input_param(inputs):
    inp = []
    param = []
    for i in inputs:
        if i.data is not None:
            param.append(i)
        else:
            inp.append(i)
    return inp, param


class ConvolutionReLU(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        conv = layer.params.convolution_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.conv_2d(
            inputs,
            params,
            node.name,
            conv_2d_num_output=conv.num_output,
            conv_2d_kernel_h=conv.kernel_h,
            conv_2d_kernel_w=conv.kernel_w,
            conv_2d_pad_h=conv.pad_h,
            conv_2d_pad_w=conv.pad_w,
            conv_2d_stride_h=conv.stride_h,
            conv_2d_stride_w=conv.stride_w,
            conv_2d_group=conv.group,
            conv_2d_relu_flag=True,
            conv_2d_bias_flag=conv.bias_term,
        )


class Conv(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        n = len(inputs[1].shape.dims)
        if n == 4:
            num_output = inputs[1].shape.dims[0]
            kernel_h = inputs[1].shape.dims[2]
            kernel_w = inputs[1].shape.dims[3]
            params = list()
            conv_params_kernel_shape = node.get_attribute_value(
                "kernel_shape", [kernel_h, kernel_w]
            )
            conv_params_pads = node.get_attribute_value("pads", [0, 0])
            conv_params_strides = node.get_attribute_value("strides", [1, 1])
            conv_params_group = node.get_attribute_value("group", 1)
            dilation = node.get_attribute_value("dilations", [1, 1])

            return parser.proxy.conv_2d(
                inputs,
                params,
                node.name,
                conv_2d_num_output=num_output,
                conv_2d_kernel_h=conv_params_kernel_shape[0],
                conv_2d_kernel_w=conv_params_kernel_shape[1],
                conv_2d_pad_h=conv_params_pads[0],
                conv_2d_pad_w=conv_params_pads[1],
                conv_2d_stride_h=conv_params_strides[0],
                conv_2d_stride_w=conv_params_strides[1],
                conv_2d_hole_h=dilation[0],
                conv_2d_hole_w=dilation[1],
                conv_2d_group=conv_params_group,
                conv_2d_bias_flag=len(node.input) == 3,
            )
        else:
            num_output = inputs[1].shape.dims[0]
            kernel = inputs[1].shape.dims[2:]
            inputs, params = split_input_param(inputs)
            conv_params_kernel_shape = node.get_attribute_value("kernel_shape", kernel)
            conv_params_pad = node.get_attribute_value(
                "pads", [0 for i in range(2 * n - 4)]
            )
            conv_params_stride = node.get_attribute_value(
                "strides", [1 for i in range(n - 2)]
            )
            conv_params_group = node.get_attribute_value("group", 1)
            dilation = node.get_attribute_value("dilations", [1 for i in range(n - 2)])

            assert (
                len(conv_params_kernel_shape) == n - 2
            ), "conv_params_kernel_shape not match inputs_shape"
            assert len(conv_params_pad) == 2 * (n - 2) or len(conv_params_pad) == (
                n - 2
            ), "conv_params_pad_shape not match inputs_shape"
            assert (
                len(conv_params_stride) == n - 2
            ), "conv_params_stride_shape not match inputs_shape"
            if len(conv_params_pad) == 2 * (n - 2):
                assert (
                    conv_params_pad[: n - 2] == conv_params_pad[n - 2 :]
                ), "asymmetric pad have not been realized"

            return parser.proxy.conv_nd(
                inputs,
                params,
                node.name,
                conv_nd_num_output=num_output,
                conv_nd_kernel=conv_params_kernel_shape,
                conv_nd_pad=conv_params_pad[: n - 2],
                # conv_1d_pad=conv_params_pad,
                conv_nd_stride=conv_params_stride,
                conv_nd_hole=dilation,
                conv_nd_group=conv_params_group,
                conv_nd_bias_flag=len(node.input) == 3,
            )


class MatMul(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.matmul(inputs, layer_name=node.name)


class ReduceMax(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.reducemax(
            inputs,
            params,
            node.name,
            reduce_axes=node.get_attribute_value("axes"),
            reduce_keepdims=node.get_attribute_value("keepdims"),
        )


class ReduceMean(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.reducemean(
            inputs,
            params,
            node.name,
            reduce_axes=node.get_attribute_value("axes"),
            reduce_keepdims=node.get_attribute_value("keepdims"),
        )


class ReduceMin(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.reducemin(
            inputs,
            params,
            node.name,
            reduce_axes=node.get_attribute_value("axes"),
            reduce_keepdims=node.get_attribute_value("keepdims"),
        )


class ReduceSum(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.reducesum(
            inputs,
            params,
            node.name,
            reduce_axes=node.get_attribute_value("axes"),
            reduce_keepdims=node.get_attribute_value("keepdims"),
        )


class ReduceProd(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.reduceprod(
            inputs,
            params,
            node.name,
            reduce_axes=node.get_attribute_value("axes"),
            reduce_keepdims=node.get_attribute_value("keepdims"),
        )


class ReduceL2(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.reducel2(
            inputs,
            params,
            node.name,
            reduce_axes=node.get_attribute_value("axes"),
            reduce_keepdims=node.get_attribute_value("keepdims"),
        )


class BatchNormalization(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        bn = layer.bn_param
        inputs, params = split_input_param(inputs)

        return parser.proxy.bn(inputs, params, node.name, bn_eps=bn.var_eps)


class InstanceNormalization(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)

        return parser.proxy.instancenorm(
            inputs,
            params,
            node.name,
            instancenorm_eps=node.get_attribute_value("epsilon", 1e-5),
        )


class CaffeBatchNorm(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        scale_factor = layer.blobs.pop().data[0]
        scale_factor = 0 if abs(scale_factor) < 1e-6 else 1 / scale_factor
        layer.blobs[0].data[:] = (
            numpy.array(layer.blobs[0].data[:], dtype="float32") * scale_factor
        )
        layer.blobs[1].data[:] = (
            numpy.array(layer.blobs[1].data[:], dtype="float32") * scale_factor
        )

        batchnorm = layer.batch_norm_param
        batchnorm_eps = batchnorm.eps
        inputs, params = split_input_param(inputs)
        return parser.proxy.batchnorm(
            inputs, params, layer.name, batchnorm_eps=batchnorm_eps
        )


class CaffeScale(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        bias_term = True if len(layer.blobs) == 2 else False
        inputs, params = split_input_param(inputs)

        return parser.proxy.scale(inputs, params, layer.name, scale_bias_term=bias_term)


class ConvTranspose(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        kc = inputs[1].shape.dims[1]
        kernel_h = inputs[1].shape.dims[2]
        kernel_w = inputs[1].shape.dims[3]
        inputs, params = split_input_param(inputs)
        conv_params_kernel_shape = node.get_attribute_value(
            "kernel_shape", [kernel_h, kernel_w]
        )
        conv_params_pads = node.get_attribute_value("pads", [0, 0])
        conv_params_strides = node.get_attribute_value("strides", [1, 1])
        conv_params_group = node.get_attribute_value("group", 1)
        num_output = kc * conv_params_group

        return parser.proxy.deconv_2d(
            inputs,
            params,
            node.name,
            conv_2d_num_output=num_output,
            conv_2d_kernel_h=conv_params_kernel_shape[0],
            conv_2d_kernel_w=conv_params_kernel_shape[1],
            conv_2d_pad_h=conv_params_pads[0],
            conv_2d_pad_w=conv_params_pads[1],
            conv_2d_stride_h=conv_params_strides[0],
            conv_2d_stride_w=conv_params_strides[1],
            conv_2d_group=conv_params_group,
            conv_2d_bias_flag=len(node.input) == 3,
        )


class Corr(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        corr = layer.correlation_param
        return parser.proxy.correlation(
            [inputs[0], inputs[1]], layer_name=node.name, correlation_groups=corr.groups
        )


class Correlation1D(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.correlation1d(
            [inputs[0], inputs[1]],
            layer_name=node.name,
            correlation1d_max_displacement=node.get_attribute_value("max_displacement"),
            correlation1d_kernel_size=node.get_attribute_value("kernel_size"),
            correlation1d_pad=node.get_attribute_value("pad"),
            correlation1d_single_direction=node.get_attribute_value("single_direction"),
        )


# class Add(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):

#     @classmethod
#     def parse(cls, parser, node, inputs):

#         [layer,] = parser.trans_onnx_to_caffe(node)
#         eltwise = layer.eltwise_param
#         pm = {
#                 'eltwise_operation': 1,
#                 }
#         if len(eltwise.coeff) > 0:
#             pm['eltwise_coeff'] = eltwise.coeff
#         if (len(node.input) > len(eltwise.coeff)):
#             pm['eltwise_coeff'] = list(eltwise.coeff) + [1.0] * (len(node.input) - len(eltwise.coeff))
#         return parser.proxy.eltwise(inputs, layer_name=node.name,
#                 **pm
#                 )


# class Mul(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):

#     @classmethod
#     def parse(cls, parser, node, inputs):

#         [layer,] = parser.trans_onnx_to_caffe(node)
#         eltwise = layer.eltwise_param
#         pm = {
#                 'eltwise_operation': 0,
#                 }
#         if len(eltwise.coeff) > 0:
#             pm['eltwise_coeff'] = eltwise.coeff
#         if (len(node.input) > len(eltwise.coeff)):
#             pm['eltwise_coeff'] = list(eltwise.coeff) + [1.0] * (len(node.input) - len(eltwise.coeff))
#         return parser.proxy.eltwise(inputs, layer_name=node.name,
#                 **pm
#                 )

# class Max(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):

#     @classmethod
#     def parse(cls, parser, node, inputs):
#         [layer,] = parser.trans_onnx_to_caffe(node)
#         eltwise = layer.eltwise_param
#         pm = {
#                 'eltwise_operation': 2,
#                 }
#         if len(eltwise.coeff) > 0:
#             pm['eltwise_coeff'] = eltwise.coeff
#         if (len(node.input) > len(eltwise.coeff)):
#             pm['eltwise_coeff'] = list(eltwise.coeff) + [1.0] * (len(node.input) - len(eltwise.coeff))
#         return parser.proxy.eltwise(inputs, layer_name=node.name,
#                 **pm
#                 )


class Eltwise(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        eltwise = layer.eltwise_param
        pm = {
            "eltwise_operation": 1,
        }
        if len(eltwise.coeff) > 0:
            pm["eltwise_coeff"] = eltwise.coeff
        if len(node.input) > len(eltwise.coeff):
            pm["eltwise_coeff"] = list(eltwise.coeff) + [1.0] * (
                len(node.input) - len(eltwise.coeff)
            )
        return parser.proxy.eltwise(inputs, layer_name=node.name, **pm)


class Mul(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.mul(inputs, layer_name=node.name)


class Div(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.div(inputs, layer_name=node.name)


class Abs(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.abs(inputs, params, layer_name=node.name)


class Floor(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.floor(inputs, params, layer_name=node.name)


class Add(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.add(inputs, layer_name=node.name)


class Min(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.min(inputs, layer_name=node.name)


class ArgMax(CaffeParser.LayerParser, OpDesc, parser=CaffeParser, version=11):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.argmax(
            inputs,
            params,
            layer_name=node.name,
            argmax_axis=node.get_attribute_value("axis"),
            argmax_keepdims=node.get_attribute_value("keepdims"),
            argmax_select_last_index=node.get_attribute_value("select_last_index"),
        )


class Sub(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.sub(inputs, layer_name=node.name)


class Pow(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.pow(inputs, layer_name=node.name)


class Sqrt(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.sqrt(inputs, layer_name=node.name)


class Exp(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.exp(inputs, layer_name=node.name)


class Log(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.log(inputs, layer_name=node.name)


class LeakyRelu(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        import numpy as np

        alpha = node.get_attribute_value("alpha")
        slope = FakeTensor(
            shape=(
                [
                    1,
                ],
                1,
                0,
            ),
            data=np.array([alpha], dtype="float32"),
        )
        return parser.proxy.prelu(inputs, [slope], node.name, prelu_share=True)


class MaxPool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    MAX = 0
    AVE = 1

    @classmethod
    def parse(cls, parser, node, inputs):

        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.pooling_param

        return parser.proxy.pool(
            inputs,
            layer_name=node.name,
            pool_method=cls.MAX,
            pool_pad_h=pool.pad_h,
            pool_pad_w=pool.pad_w,
            pool_kernel_h=pool.kernel_h,
            pool_kernel_w=pool.kernel_w,
            pool_stride_h=pool.stride_h,
            pool_stride_w=pool.stride_w,
            pool_ceil_mode=pool.ceil_mode,
        )


# keep the class name same as Operator op_type although it duplicates with earlier defined class,
# which makes it easier to register parser, otherwise, the `layertype` is required to be set.
class MaxPool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser, version=10):
    MAX = 0
    AVE = 1

    @classmethod
    def parse(cls, parser, node, inputs):

        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.pooling_param

        return parser.proxy.pool(
            inputs,
            layer_name=node.name,
            pool_method=cls.MAX,
            pool_pad_h=pool.pad_h,
            pool_pad_w=pool.pad_w,
            pool_kernel_h=pool.kernel_h,
            pool_kernel_w=pool.kernel_w,
            pool_stride_h=pool.stride_h,
            pool_stride_w=pool.stride_w,
            pool_ceil_mode=pool.ceil_mode,
        )


class AveragePool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    MAX = 0
    AVE = 1

    @classmethod
    def parse(cls, parser, node, inputs):

        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.pooling_param

        return parser.proxy.pool(
            inputs,
            layer_name=node.name,
            pool_method=cls.AVE,
            pool_pad_h=pool.pad_h,
            pool_pad_w=pool.pad_w,
            pool_kernel_h=pool.kernel_h,
            pool_kernel_w=pool.kernel_w,
            pool_stride_h=pool.stride_h,
            pool_stride_w=pool.stride_w,
            pool_ceil_mode=pool.ceil_mode,
        )


# keep the class name same as Operator op_type although it duplicates with earlier defined class,
# which makes it easier to register parser, otherwise, the `layertype` is required to be set.
class AveragePool(
    CaffeParser.LayerParser,
    OpDesc,
    parser=CaffeParser,
    op_type="AveragePool",
    version=10,
):
    MAX = 0
    AVE = 1

    @classmethod
    def parse(cls, parser, node, inputs):

        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.pooling_param

        return parser.proxy.pool(
            inputs,
            layer_name=node.name,
            pool_method=cls.AVE,
            pool_pad_h=pool.pad_h,
            pool_pad_w=pool.pad_w,
            pool_kernel_h=pool.kernel_h,
            pool_kernel_w=pool.kernel_w,
            pool_stride_h=pool.stride_h,
            pool_stride_w=pool.stride_w,
            pool_ceil_mode=pool.ceil_mode,
        )


class InnerProductReLU(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe("InnerProductReLU", node)
        ip = layer.inner_product_param

        assert len(inputs[0].shape.dims) == 4 or len(inputs[0].shape.dims) == 2
        if len(inputs[0].shape.dims) == 4:
            layerdef.blobs[0].shape.ClearField("dim")
            assert (
                inputs[0].shape.channel_axis == 1 or inputs[0].shape.channel_axis == 3
            )
            if inputs[0].shape.channel_axis == 1:
                layerdef.blobs[0].shape.dim.append(ip.num_output)
                layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
            else:
                layerdef.blobs[0].shape.dim.append(ip.num_output)
                layerdef.blobs[0].shape.dim.extend(
                    map(lambda x: inputs[0].shape.dims[x], [3, 1, 2])
                )
        elif len(inputs[0].shape.dims) == 2:
            layerdef.blobs[0].shape.ClearField("dim")
            assert (
                inputs[0].shape.channel_axis == 1 or inputs[0].shape.channel_axis == 0
            )
            if inputs[0].shape.channel_axis == 1:
                layerdef.blobs[0].shape.dim.append(ip.num_output)
                layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
            else:
                layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
                layerdef.blobs[0].shape.dim.append(ip.num_output)

        inputs, params = split_input_param(inputs)
        return parser.proxy.ip(
            inputs,
            params,
            layerdef.name,
            ip_relu_flag=True,
            ip_num_output=ip.num_output,
        )


# todo


class Gemm(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        num_output = inputs[1].shape.dims[0]
        axis = inputs[1].shape.channel_axis
        inputs, params = split_input_param(inputs)
        return parser.proxy.ip(
            inputs, params, node.name, ip_num_output=num_output, ip_axis=axis
        )


class Gather(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        axis = node.get_attribute_value("axis", 0)
        return parser.proxy.gather(inputs, layer_name=node.name, gather_axis=axis)


class GridSample(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    # mode
    MODE_BILINEAR = 1
    MODE_NEAREST = 2

    # padding mode
    PADDING_ZEROS = 1
    PADDING_BORDER = 2
    PADDING_REFLECTION = 3

    @classmethod
    def parse(cls, parser, node, inputs):
        mode = node.get_attribute_value("mode")
        if mode == "bilinear":
            mode = GridSample.MODE_BILINEAR
        elif mode == "nearest":
            mode = GridSample.MODE_NEAREST
        else:
            logger.fatal(
                f"[GridSample] Unsupported mode `{mode}`, should be `bilinear` or `nearest`."
            )

        padding_mode = node.get_attribute_value("padding_mode")
        if padding_mode == "zeros":
            padding_mode = GridSample.PADDING_ZEROS
        elif padding_mode == "border":
            padding_mode = GridSample.PADDING_BORDER
        elif padding_mode == "reflection":
            padding_mode = GridSample.PADDING_REFLECTION
        else:
            logger.fatal(
                f"[GridSample] Unsupported mode `{mode}`, should be `bilinear` or `nearest`."
            )
        align_corners = node.get_attribute_value("align_corners")
        return parser.proxy.gridsample(
            inputs,
            layer_name=node.name,
            gridsample_mode=mode,
            gridsample_padding_mode=padding_mode,
            gridsample_align_corners=align_corners,
        )


class Relu(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        return parser.proxy.relu(inputs, layer_name=node.name)


class Flatten(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        shape = layer.reshape_param.shape.dim
        return parser.proxy.reshape(
            [inputs[0]],
            layer_name=node.name,
            reshape_dims=shape,
            reshape_dim_size=len(shape),
            reshape_axis=0,
            reshape_num_axes=-1,
        )


class Squeeze(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        shape = layer.reshape_param.shape.dim
        return parser.proxy.reshape(
            [inputs[0]],
            layer_name=node.name,
            reshape_dims=shape,
            reshape_dim_size=len(shape),
            reshape_axis=0,
            reshape_num_axes=-1,
        )


class Unsqueeze(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        shape = layer.reshape_param.shape.dim
        return parser.proxy.reshape(
            [inputs[0]],
            layer_name=node.name,
            reshape_dims=shape,
            reshape_dim_size=len(shape),
            reshape_axis=0,
            reshape_num_axes=-1,
        )


class Unfold(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        def _get_list(name, val):
            if isinstance(val, (list, tuple)):
                if len(val) == 2:
                    return tuple(val)
                elif len(val) == 1:
                    return tuple(val * 2)
                else:
                    logger.warning("Can only support 2-D {self.__class__.__name__}")
                    return tuple(val[:2])
            return (val, val)

        inputs, params = split_input_param(inputs)
        conv_params_kernel_shape = _get_list(
            "kernel_size", node.get_attribute_value("kernel_size")
        )
        conv_params_pads = _get_list(
            "padding", node.get_attribute_value("padding", [0, 0])
        )
        conv_params_strides = _get_list(
            "stride", node.get_attribute_value("stride", [1, 1])
        )
        dilation = _get_list("dilation", node.get_attribute_value("dilation", [1, 1]))

        return parser.proxy.unfold(
            inputs,
            params,
            node.name,
            unfold_kernel_h=conv_params_kernel_shape[0],
            unfold_kernel_w=conv_params_kernel_shape[1],
            unfold_pad_h=conv_params_pads[0],
            unfold_pad_w=conv_params_pads[1],
            unfold_stride_h=conv_params_strides[0],
            unfold_stride_w=conv_params_strides[1],
            unfold_hole_h=dilation[0],
            unfold_hole_w=dilation[1],
        )


class QuantDequant(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        from ..core.art import FakeTensor

        dummy = FakeTensor(
            shape=(inputs[0].shape.dims, 1, 0), name="fake_" + inputs[0].name
        )
        if inputs[0].data is None:
            inputs, params = [dummy, inputs[0]], None
        else:
            inputs, params = [dummy], [inputs[0]]
        return parser.proxy.quant_dequant(
            inputs,
            params,
            layer_name=node.name,
            quant_dequant_qmin=node.attributes["q_min"].ints,
            quant_dequant_qmax=node.attributes["q_max"].ints,
            quant_dequant_scale=node.attributes["scale"].floats,
        )


class Clip(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        if len(inputs) == 3:
            inputs, params = split_input_param(inputs)
            return parser.proxy.clip(inputs, params, layer_name=node.name)

        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        if layer.type == "ReLU6":
            return parser.proxy.relu6(inputs, layer_name=node.name)
        else:
            from ..core.art import FakeTensor

            t_min = FakeTensor(
                shape=(1,),
                name=inputs[0].name + "_clip_min",
                data=node.get_attribute_value("min"),
            )
            t_max = FakeTensor(
                shape=(1,),
                name=inputs[0].name + "_clip_max",
                data=node.get_attribute_value("max"),
            )
            return parser.proxy.clip([*inputs, t_min, t_max], layer_name=node.name)


class PRelu(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        shared = False
        if params[0].data.size == 1:
            shared = True
        return parser.proxy.prelu(inputs, params, node.name, prelu_share=shared)


class Tanh(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        return parser.proxy.tanh(inputs, layer_name=node.name)


class TopK(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.topk(
            inputs,
            layer_name=node.name,
            topk_axis=node.get_attribute_value("axis"),
            topk_k=node.get_attribute_value("k"),
        )


class Sigmoid(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        return parser.proxy.sigmoid(inputs, layer_name=node.name)


class ScatterND(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.scatternd(
            inputs,
            layer_name=node.name,
            reduction=node.get_attribute_value("reduction"),
        )


class Concat(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        concat = layer.concat_param

        dim_size = len(node.owning_graph.get_tensor_shape(node.input[0]))
        axis = concat.axis if concat.axis >= 0 else concat.axis + dim_size
        return parser.proxy.concat(inputs, layer_name=node.name, concat_axis=axis)


class Constant(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        pass


class Reshape(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        # FIXME:
        #   support 1-input implementation only
        from onnx import numpy_helper

        if node.input[1] in node.owning_graph.initializer:
            shape = numpy_helper.to_array(node.owning_graph.initializer[node.input[1]])
        else:
            constant_node = node.owning_graph.get_tensor_producer(node.input[1])[-1]
            shape = list(
                np.frombuffer(
                    constant_node.attributes["value"].t.raw_data, dtype=np.int64
                )
            )
        return parser.proxy.reshape(
            [inputs[0]],
            layer_name=node.name,
            reshape_dims=shape,
            reshape_dim_size=len(shape),
            reshape_axis=0,
            reshape_num_axes=-1,
        )


class CaffeSoftmax(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        softmax = layer.softmax_param

        dim_size = len(node.owning_graph.get_tensor_shape(node.input[0]))
        axis = softmax.axis if softmax.axis >= 0 else softmax.axis + dim_size
        return parser.proxy.softmax(inputs, layer_name=node.name, softmax_axis=axis)


class Softmax(CaffeParser.LayerParser, OpDesc, parser=CaffeParser, version=13):
    @classmethod
    def parse(cls, parser, node, inputs):
        dim_size = len(node.owning_graph.get_tensor_shape(node.input[0]))
        axis = node.get_attribute_value("axis")
        axis = axis % dim_size
        return parser.proxy.softmax(inputs, layer_name=node.name, softmax_axis=axis)


class Transpose(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        perm = node.get_attribute_value("perm")
        return parser.proxy.transpose(
            inputs, [], node.name, transpose_dims=perm, transpose_exgaxis=False
        )


class Upsample(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    NEAREST = 0
    BILINEAR = 1

    @classmethod
    def parse(cls, parser, node, inputs):
        mode = node.get_attribute_value("mode")
        height, width = node.owning_graph.get_tensor_shape(node.output[0])[2:]
        p = {}
        if mode == "bilinear" or mode == "linear":
            p["interp_height"] = height
            p["interp_width"] = width
            p["interp_type"] = Upsample.BILINEAR
        elif mode == "nearest":
            p["interp_height"] = height
            p["interp_width"] = width
            p["interp_type"] = Upsample.NEAREST
        else:
            raise RuntimeError(f"can not support mode {mode} in Upsample")
        return parser.proxy.interp([inputs[0]], layer_name=node.name, **p)


class DynamicUpsample(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    NEAREST = 0
    BILINEAR = 1

    @classmethod
    def parse(cls, parser, node, inputs):
        mode = node.attributes["mode"].s.decode("utf-8")
        p = {}
        if mode == "bilinear" or mode == "linear":
            p["interp_type"] = Upsample.BILINEAR
        elif mode == "nearest":
            p["interp_type"] = Upsample.NEAREST
        else:
            raise RuntimeError(f"can not support mode {mode} in Upsample")
        return parser.proxy.interp(inputs, layer_name=node.name, **p)


class LpNormalization(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.lpnormalization(
            inputs,
            layer_name=node.name,
            lpnormalization_p=node.get_attribute_value("p", 2),
            lpnormalization_axis=node.get_attribute_value("axis", -1),
        )


class RoiPool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.roi_pooling_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.roipooling(
            inputs,
            params,
            node.name,
            roipooling_pooled_height=pool.pooled_h,
            roipooling_pooled_width=pool.pooled_w,
            roipooling_spatial_scale=pool.spatial_scale,
        )


class MaxRoiPool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.roi_pooling_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.roipooling(
            inputs,
            params,
            node.name,
            roipooling_pooled_height=pool.pooled_h,
            roipooling_pooled_width=pool.pooled_w,
            roipooling_spatial_scale=pool.spatial_scale,
        )


class PSRoiPool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.psroi_pooling_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.psroipooling(
            inputs,
            params,
            node.name,
            psroipooling_group_size=pool.group_size,
            psroipooling_output_dim=pool.output_dim,
            psroipooling_spatial_scale=pool.spatial_scale,
        )


class RoiAlign(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.roi_align_pooling_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.podroialignpooling(
            inputs,
            params,
            node.name,
            podroialignpooling_pooled_height=pool.pooled_h,
            podroialignpooling_pooled_width=pool.pooled_w,
            podroialignpooling_sample_num=pool.sample_num,
            podroialignpooling_spatial_scale=pool.spatial_scale,
        )


class PODRoiAlign(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        pool = layer.podroi_align_pooling_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.podroialignpooling(
            inputs,
            params,
            node.name,
            podroialignpooling_pooled_height=pool.pooled_h,
            podroialignpooling_pooled_width=pool.pooled_w,
            podroialignpooling_sample_num=pool.sample_num,
            podroialignpooling_spatial_scale=pool.spatial_scale,
        )


class PSRoiMaskPool(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        group_size = node.get_attribute_value("group_size")
        output_dim = node.get_attribute_value("output_dim")
        spatial_scale = node.get_attribute_value("spatial_scale")
        roi_scale = node.get_attribute_value("roi_scale")
        bin_scale = node.get_attribute_value("bin_scale")
        inputs, params = split_input_param(inputs)
        return parser.proxy.psroimaskpooling(
            inputs,
            params,
            node.name,
            psroimaskpooling_group_size=group_size,
            psroimaskpooling_output_dim=output_dim,
            psroimaskpooling_spatial_scale=spatial_scale,
            psroimaskpooling_roi_scale=roi_scale,
            psroimaskpooling_bin_scale=bin_scale,
        )


class HeatMap2Coord(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        heatmap = layer.heatmap_param
        coord_h = heatmap.coord_h
        coord_w = heatmap.coord_w
        reposition = heatmap.coord_reposition
        inputs, params = split_input_param(inputs)
        return parser.proxy.heatmap2coord(
            inputs,
            params,
            node.name,
            heatmap2coord_coord_h=coord_h,
            heatmap2coord_coord_w=coord_w,
            heatmap2coord_reposition=reposition,
        )


class ShuffleChannel(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        sc = layer.shuffle_channel_param
        inputs, params = split_input_param(inputs)
        return parser.proxy.shufflechannel(
            inputs, params, node.name, shufflechannel_group=sc.group
        )


class Slice_1(
    CaffeParser.LayerParser,
    OpDesc,
    layertype="Slice",
    op_type="Slice",
    version=1,
    parser=CaffeParser,
):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        slice = layer.slice_param
        dim_size = len(node.owning_graph.get_tensor_shape(node.input[0]))
        axis = slice.axis if slice.axis >= 0 else slice.axis + dim_size
        slice_point = list(slice.slice_point)
        if len(slice_point) == 0:
            logger.error("The slice point of `{0}` is empty".format(node.name))
        return parser.proxy.slice(
            inputs, [], node.name, slice_axis=axis, slice_point=slice_point
        ), list(layer.top)


class Slice_10(
    CaffeParser.LayerParser,
    OpDesc,
    layertype="Slice",
    op_type="Slice",
    version=10,
    parser=CaffeParser,
):
    @classmethod
    def parse(cls, parser, node, inputs):
        shape = node.owning_graph.get_tensor_shape(node.input[0])
        dim_size = len(shape)
        graph = node.owning_graph
        starts = graph.get_const_tensor_as_array(node.input[1], False)
        ends = graph.get_const_tensor_as_array(node.input[2], False)
        axes = graph.get_const_tensor_as_array(node.input[3], False)
        if graph.has_input(4):
            # has steps input.
            steps = graph.get_const_tensor_as_array(node.input[4], False)
            assert all(x == 1 for x in steps), "strided slice is not supported yet"
        assert len(axes) == 1, "only support slice on one axis at present."
        axis, start, end = axes[0], starts[0], ends[0]
        axis = axis % dim_size
        slice_point = []
        outputs = []
        if start > 0:
            slice_point.append(start)
            outputs.append(f"{node.name}_ignore0")
        outputs.append(node.output[0])
        if end < shape[axis]:
            slice_point.append(end)
            outputs.append(f"{node.name}_ignore1")
        return (
            parser.proxy.slice(
                [inputs[0]], [], node.name, slice_axis=axis, slice_point=slice_point
            ),
            outputs,
        )


# class Slice_10(CaffeParser.LayerParser, OpDesc, layertype="Slice", op_type="Slice", version=10, parser=CaffeParser):
#     @classmethod
#     def parse(cls, parser, node, inputs):
#         # TODO: slice-10 not supported at present.
#         pass


class Split(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        slice = layer.slice_param
        dim_size = len(node.owning_graph.get_tensor_shape(node.input[0]))
        axis = (slice.axis if slice.axis >= 0 else slice.axis + dim_size,)
        slice_point = list(slice.slice_point)
        if len(slice_point) == 0:
            logger.error("The slice point of `{0}` is empty".format(node.name))
        return parser.proxy.slice(
            inputs, [], node.name, slice_axis=axis, slice_point=slice_point
        ), list(layer.top)


class SubPixel(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.subpixel(
            inputs,
            [],
            node.name,
            subpixel_method=node.get_attribute_value("method"),
            subpixel_sample=node.get_attribute_value("sample"),
        ), list(node.output)


class Pad(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        # in some situations. pads and constant value is not inputs but attributes
        # but now i have no such model, so you can add it when needed
        pad_mode = node.get_attribute_value("mode", "constant")
        inputs, params = split_input_param(inputs)
        pads = node.owning_graph.get_const_tensor_as_array(node.input[1])
        value = node.owning_graph.get_const_tensor_as_array(node.input[2])
        # if len(pads)!=8:
        #     assert any(pads)==0
        #     pads=[0,0,0,0,0,0,0,0]
        mode = 1
        if pad_mode == "constant":
            mode = 1
        elif pad_mode == "reflect":
            mode = 2
        elif pad_mode == "edge":
            mode = 3
        return parser.proxy.pad(
            inputs, params, node.name, pad_mode=mode, pad_value=value[0], pad_pads=pads
        )


class LSTM(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    """The lstm following onnx-7 standard
    BUT! without initial_h initial_c peepholes
    """

    # These three dict defined in op.py

    # Define the enum of activation functions
    # activation_mode = {'Relu': 1,
    #                    ...
    #                   }

    # Define the default alpha and beta perameter of activation fuctions
    # default_activation_alpha_beta = {'Relu': (0, 0),
    #                                  ...
    #                                 }

    # Define the enum of direction
    # direction_mode = {'forward': 1,
    #                   'reverse': 2,
    #                   'sigmoid': 3,
    #                  }

    @classmethod
    def parse(cls, parser, node, inputs):
        # inputs, params = split_input_param(inputs)
        params = None

        hidden_size = node.get_attribute_value("hidden_size")

        direction_str = node.get_attribute_value("direction", "forward")
        assert direction_str in node.direction_mode.keys()
        direction = node.direction_mode[direction_str]

        input_forget = node.get_attribute_value("input_forget", 0)

        clip = node.get_attribute_value("clip", None)
        if clip is None:
            clip = 0
            clip_exist = 0
        else:
            clip_exist = 1

        activations = node.get_attribute_value("activations")
        for activation in activations:
            assert activation in node.activation_mode.keys()

        (activation_f, activation_g, activation_h) = activations

        activation_alpha = node.get_attribute_value("activation_alpha")
        if activation_alpha is None:
            activation_alpha_f = node.default_activation_alpha_beta[activation_f][0]
            activation_alpha_g = node.default_activation_alpha_beta[activation_g][0]
            activation_alpha_h = node.default_activation_alpha_beta[activation_h][0]

        activation_beta = node.get_attribute_value("activation_beta")
        if activation_beta is None:
            activation_beta_f = node.default_activation_alpha_beta[activation_f][1]
            activation_beta_g = node.default_activation_alpha_beta[activation_g][1]
            activation_beta_h = node.default_activation_alpha_beta[activation_h][1]

        # transpose string activation into int mode
        activation_mode_f = node.activation_mode[activation_f]
        activation_mode_g = node.activation_mode[activation_g]
        activation_mode_h = node.activation_mode[activation_h]

        output_size = len(node.output)

        return parser.proxy.lstm(
            inputs,
            params,
            node.name,
            lstm_hidden_size=hidden_size,
            lstm_direction=direction,
            lstm_input_forget=input_forget,
            lstm_clip=clip,
            lstm_clip_exist=clip_exist,
            lstm_activation_f=activation_mode_f,
            lstm_activation_g=activation_mode_g,
            lstm_activation_h=activation_mode_h,
            lstm_activation_alpha_f=activation_alpha_f,
            lstm_activation_alpha_g=activation_alpha_g,
            lstm_activation_alpha_h=activation_alpha_h,
            lstm_activation_beta_f=activation_beta_f,
            lstm_activation_beta_g=activation_beta_g,
            lstm_activation_beta_h=activation_beta_h,
            lstm_output_size=output_size,
        )


class Hsigmoid(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.hardsigmoid(
            inputs,
            params,
            layer_name=node.name,
            hardsigmoid_alpha=node.get_attribute_value("alpha", 0.2),
            hardsigmoid_beta=node.get_attribute_value("beta", 0.5),
        )


class Erf(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.erf(
            inputs,
            params,
            layer_name=node.name,
        )


class Cast(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        onnx_dtype = node.get_attribute_value("to", 2)
        art_dtype = map_onnx_to_art_dtype[onnx_dtype]
        return parser.proxy.cast(
            inputs, params, layer_name=node.name, cast_dtype=art_dtype
        )


class Hswish(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.hswish(inputs, params, layer_name=node.name)


class Reciprocal(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        one_tensor = FakeTensor(
            dtype=inputs[0].dtype,
            name="ones_like_" + inputs[0].name,
            shape=([1], 1, 0),
            data=1,
        )
        return parser.proxy.div([one_tensor] + inputs, layer_name=node.name)


class Sign(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        return parser.proxy.sign(inputs, layer_name=node.name)


class RoundTo0(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        inputs, params = split_input_param(inputs)
        return parser.proxy.roundto0(inputs, params, layer_name=node.name)


class Elu(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        import numpy as np

        alpha = node.get_attribute_value("alpha")
        return parser.proxy.elu(inputs, [], node.name, elu_alpha=alpha)


class AddDivClipCast(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        if len(inputs) != 5:
            raise NotImplementedError(
                "Input should be of length 5 for fused op AddDivClipCast"
            )
        inputs, params = split_input_param(inputs)
        onnx_dtype = node.get_attribute_value("to", 2)
        art_dtype = map_onnx_to_art_dtype[onnx_dtype]
        return parser.proxy.add_div_clip_cast(
            inputs, params, layer_name=node.name, cast_dtype=art_dtype
        )


class ClipCast(CaffeParser.LayerParser, OpDesc, parser=CaffeParser):
    @classmethod
    def parse(cls, parser, node, inputs):
        if len(inputs) != 3:
            raise NotImplementedError(
                "Input should be of length 3 for fused op ClipCast"
            )
        inputs, params = split_input_param(inputs)
        onnx_dtype = node.get_attribute_value("to", 2)
        art_dtype = map_onnx_to_art_dtype[onnx_dtype]
        return parser.proxy.clip_cast(
            inputs, params, layer_name=node.name, cast_dtype=art_dtype
        )
