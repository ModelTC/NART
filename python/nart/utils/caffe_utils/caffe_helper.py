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
from ...core.art import Proxy
from ...core.art import FakeTensor
from ...core.art import FakeParade
from ...core.art import Dtype
import inspect
import numpy
from functools import reduce


class CaffeHelper:
    """parse caffe net to get imformation."""

    dispatch_dct = {}

    @classmethod
    def get_support_layer_list(cls):
        return list(map(lambda x: x, CaffeHelper.dispatch_dct))

    class LayerParser:
        @classmethod
        def __init_subclass__(cls, layertype=None, overwrite=False):
            if layertype is None:
                layertype = cls.__name__
            if cls.parse is CaffeHelper.LayerParser.parse:
                raise
            if overwrite is False and layertype in CaffeHelper.dispatch_dct:
                raise
            CaffeHelper.dispatch_dct[layertype] = cls

        @classmethod
        def parse(cls, parser, layerdef, inputs):
            pass

    def __init__(self, netdef, netweight=None, proxy=Proxy.ModuleProxy("default")):
        self.proxy = proxy
        self.max_batch_size = -1
        if len(CaffeHelper.dispatch_dct) is 0:
            raise RuntimeError(
                "No LayerParser registered, if you want to try out default LayerpParsers, call CaffeHelper.register_defaults()"
            )
        self.netdef = netdef.__class__()
        self.netdef.CopyFrom(netdef)
        if netweight is not None:
            for layer in self.netdef.layer:
                for lw in netweight.layer:
                    if lw.name == layer.name:
                        layer.ClearField("blobs")
                        layer.blobs.extend(lw.blobs)
                        break

    def get_tensor(name):
        return self.dct_name_tensor[name]

    def set_max_batch(self, batch_size):
        self.max_batch_size = batch_size
        return self

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        netdef = self.before_parse(self.netdef)
        for layer in netdef.layer:
            if layer.type not in CaffeHelper.dispatch_dct:
                raise RuntimeError(
                    f"LayerParser for type '{layer.type}' not registered"
                )
        into_parade_outputs = {}
        for t in into_parade.tensors:
            if t.dtype == "float32":
                into_parade_outputs[t.name] = t
        dct_name_tensor = self.parse_input(netdef)
        dct_name_tensor.update(into_parade_outputs)
        self.parse_layers(netdef, dct_name_tensor, into_parade)
        return into_parade

    def op_post(self, netdef, op):
        return op

    def parse_layers(self, netdef, dct_name_tensor, parade):
        for layer in netdef.layer:
            op = self.op_post(
                layer,
                CaffeHelper.dispatch_dct[layer.type]().parse(
                    self, layer, list(map(lambda x: dct_name_tensor[x], layer.bottom))
                ),
            )
            res_list = parade.append(op)
            assert len(res_list) == len(layer.top)
            for name, tensor in zip(layer.top, res_list):
                tensor.name = name
            dct_name_tensor.update({k: v for k, v in zip(layer.top, res_list)})
        return parade

    def parse_input(self, netdef):
        def apply_batch_size(shape):
            if self.max_batch_size > 0:
                shape[0] = self.max_batch_size
            return shape

        res = {}
        for ipt, sp in zip(netdef.input, netdef.input_shape):
            tensor = FakeTensor(
                dtype=Dtype.Float32, shape=apply_batch_size(list(sp.dim)), name=ipt
            )
            res[ipt] = tensor
        return res

    def before_parse(self, netdef):
        if len(netdef.input_dim) != 0:
            if len(netdef.input_shape) != 0:
                raise
            assert len(netdef.input_dim) == len(netdef.input) * 4
            for i in range(len(netdef.input)):
                netdef.input_shape.add().dim.extend(netdef.input_dim[i * 4 : i * 4 + 4])
        netdef.ClearField("input_dim")
        netdef = merge_split(netdef)
        return netdef

    def param_to_tensor(self, layerdef):
        params = []
        for p in layerdef.blobs:
            sp = (list(p.shape.dim), 1 if len(p.shape.dim) > 1 else -1, 0)
            ddd = numpy.array(p.data, dtype="float32")
            r = FakeTensor(data=ddd, shape=sp, dtype="float32")
            params.append(r)
        return params

    @classmethod
    def register_defaults(cls):
        class ConvolutionReLU(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                conv = layerdef.convolution_param
                if conv.HasField("kernel_h") and conv.HasField("kernel_w"):
                    kernel_h, kernel_w = conv.kernel_h, conv.kernel_w
                else:
                    kernel_h, kernel_w = conv.kernel_size, conv.kernel_size
                if conv.HasField("stride_h") and conv.HasField("stride_w"):
                    stride_h, stride_w = conv.stride_h, conv.stride_w
                else:
                    stride_h, stride_w = conv.stride, conv.stride
                if conv.HasField("pad_h") and conv.HasField("pad_w"):
                    pad_h, pad_w = conv.pad_h, conv.pad_w
                else:
                    pad_h, pad_w = conv.pad, conv.pad
                group = conv.group
                bias = conv.bias_term
                num_output = conv.num_output
                assert conv.hole_h == 1
                assert conv.hole_w == 1
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.conv_2d(
                    inputs,
                    params,
                    layerdef.name,
                    conv_2d_num_output=num_output,
                    conv_2d_kernel_h=kernel_h,
                    conv_2d_kernel_w=kernel_w,
                    conv_2d_pad_h=pad_h,
                    conv_2d_pad_w=pad_w,
                    conv_2d_stride_h=stride_h,
                    conv_2d_stride_w=stride_w,
                    conv_2d_group=group,
                    conv_2d_relu_flag=True,
                    conv_2d_bias_flag=bias,
                )

        class Convolution(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                conv = layerdef.convolution_param
                if conv.HasField("kernel_h") and conv.HasField("kernel_w"):
                    kernel_h, kernel_w = conv.kernel_h, conv.kernel_w
                else:
                    kernel_h, kernel_w = conv.kernel_size, conv.kernel_size
                if conv.HasField("stride_h") and conv.HasField("stride_w"):
                    stride_h, stride_w = conv.stride_h, conv.stride_w
                else:
                    stride_h, stride_w = conv.stride, conv.stride
                if conv.HasField("pad_h") and conv.HasField("pad_w"):
                    pad_h, pad_w = conv.pad_h, conv.pad_w
                else:
                    pad_h, pad_w = conv.pad, conv.pad
                group = conv.group
                bias = conv.bias_term
                num_output = conv.num_output
                assert conv.hole_h == 1
                assert conv.hole_w == 1
                params = parser.param_to_tensor(layerdef)

                # if (kernel_h == 3 and kernel_w == 3 and stride_h == 1 and stride_w == 1):
                #     nart_op = parser.proxy.conv_2d_wino(inputs, params,layerdef.name,
                #             conv_2d_num_output=num_output,
                #             conv_2d_kernel_h=3,
                #             conv_2d_kernel_w=3,
                #             conv_2d_pad_h=pad_h,
                #             conv_2d_pad_w=pad_w,
                #             conv_2d_stride_h=1,
                #             conv_2d_stride_w=1,
                #             conv_2d_group=group,
                #             conv_2d_bias_flag=bias
                #             )
                # else:
                #     nart_op = parser.proxy.conv_2d(inputs, params,layerdef.name,
                #             conv_2d_num_output=num_output,
                #             conv_2d_kernel_h=kernel_h,
                #             conv_2d_kernel_w=kernel_w,
                #             conv_2d_pad_h=pad_h,
                #             conv_2d_pad_w=pad_w,
                #             conv_2d_stride_h=stride_h,
                #             conv_2d_stride_w=stride_w,
                #             conv_2d_group=group,
                #             conv_2d_bias_flag=bias
                #             )
                # return nart_op
                return parser.proxy.conv_2d(
                    inputs,
                    params,
                    layerdef.name,
                    conv_2d_num_output=num_output,
                    conv_2d_kernel_h=kernel_h,
                    conv_2d_kernel_w=kernel_w,
                    conv_2d_pad_h=pad_h,
                    conv_2d_pad_w=pad_w,
                    conv_2d_stride_h=stride_h,
                    conv_2d_stride_w=stride_w,
                    conv_2d_group=group,
                    conv_2d_bias_flag=bias,
                )

        class BN(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                bn = layerdef.bn_param
                bn_eps = bn.var_eps
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.bn(inputs, params, layerdef.name, bn_eps=bn_eps)

        class BatchNorm(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                scale_factor = layerdef.blobs.pop().data[0]
                scale_factor = 0 if abs(scale_factor) < 1e-6 else 1 / scale_factor
                layerdef.blobs[0].data[:] = (
                    numpy.array(layerdef.blobs[0].data[:], dtype="float32")
                    * scale_factor
                )
                layerdef.blobs[1].data[:] = (
                    numpy.array(layerdef.blobs[1].data[:], dtype="float32")
                    * scale_factor
                )

                batchnorm = layerdef.batch_norm_param
                batchnorm_eps = batchnorm.eps
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.batchnorm(
                    inputs, params, layerdef.name, batchnorm_eps=batchnorm_eps
                )

        class Deconvolution(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                conv = layerdef.convolution_param
                if conv.HasField("kernel_h") and conv.HasField("kernel_w"):
                    kernel_h, kernel_w = conv.kernel_h, conv.kernel_w
                else:
                    kernel_h, kernel_w = conv.kernel_size, conv.kernel_size
                if conv.HasField("stride_h") and conv.HasField("stride_w"):
                    stride_h, stride_w = conv.stride_h, conv.stride_w
                else:
                    stride_h, stride_w = conv.stride, conv.stride
                if conv.HasField("pad_h") and conv.HasField("pad_w"):
                    pad_h, pad_w = conv.pad_h, conv.pad_w
                else:
                    pad_h, pad_w = conv.pad, conv.pad
                group = conv.group
                bias = conv.bias_term
                num_output = conv.num_output
                assert conv.hole_h == 1
                assert conv.hole_w == 1
                params = parser.param_to_tensor(layerdef)

                return parser.proxy.deconv_2d(
                    inputs,
                    params,
                    layerdef.name,
                    conv_2d_num_output=num_output,
                    conv_2d_kernel_h=kernel_h,
                    conv_2d_kernel_w=kernel_w,
                    conv_2d_pad_h=pad_h,
                    conv_2d_pad_w=pad_w,
                    conv_2d_stride_h=stride_h,
                    conv_2d_stride_w=stride_w,
                    conv_2d_group=group,
                    conv_2d_bias_flag=bias,
                )

        class Correlation(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                return parser.proxy.correlation(
                    [inputs[1], inputs[0]], layer_name=layerdef.name
                )

        class Eltwise(CaffeHelper.LayerParser):
            PROD = 0
            SUM = 1
            MAX = 2

            @classmethod
            def parse(cls, parser, layerdef, inputs):
                eltwise = layerdef.eltwise_param
                mp_op = {
                    eltwise.MAX: cls.MAX,
                    eltwise.PROD: cls.PROD,
                    eltwise.SUM: cls.SUM,
                }
                pm = {
                    "eltwise_operation": mp_op[layerdef.eltwise_param.operation],
                }
                if len(eltwise.coeff) > 0:
                    pm["eltwise_coeff"] = eltwise.coeff
                if len(layerdef.bottom) > len(eltwise.coeff):
                    pm["eltwise_coeff"] = list(eltwise.coeff) + [1.0] * (
                        len(layerdef.bottom) - len(eltwise.coeff)
                    )
                return parser.proxy.eltwise(inputs, layer_name=layerdef.name, **pm)

        class Pooling(CaffeHelper.LayerParser):
            MAX = 0
            AVE = 1

            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pool = layerdef.pooling_param
                if pool.HasField("kernel_h") and pool.HasField("kernel_w"):
                    kernel_h, kernel_w = pool.kernel_h, pool.kernel_w
                else:
                    kernel_h, kernel_w = pool.kernel_size, pool.kernel_size
                if pool.HasField("stride_h") and pool.HasField("stride_w"):
                    stride_h, stride_w = pool.stride_h, pool.stride_w
                else:
                    stride_h, stride_w = pool.stride, pool.stride
                if pool.HasField("pad_h") and pool.HasField("pad_w"):
                    pad_h, pad_w = pool.pad_h, pool.pad_w
                else:
                    pad_h, pad_w = pool.pad, pool.pad
                if pool.global_pooling:
                    [kernel_h, kernel_w] = (
                        inputs[0].shape.dims[2:4]
                        if inputs[0].shape.channel_axis == 1
                        else inputs[0].shape.dims[1:3]
                    )
                return parser.proxy.pool(
                    inputs,
                    layer_name=layerdef.name,
                    pool_method=cls.MAX if pool.pool == pool.MAX else cls.AVE,
                    pool_pad_h=pad_h,
                    pool_pad_w=pad_w,
                    pool_kernel_h=kernel_h,
                    pool_kernel_w=kernel_w,
                    pool_stride_h=stride_h,
                    pool_stride_w=stride_w,
                    pool_ceil_mode=pool.ceil_mode,
                )

        class InnerProductReLU(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                ip = layerdef.inner_product_param
                assert len(inputs[0].shape.dims) == 4 or len(inputs[0].shape.dims) == 2
                if len(inputs[0].shape.dims) == 4:
                    layerdef.blobs[0].shape.ClearField("dim")
                    assert (
                        inputs[0].shape.channel_axis == 1
                        or inputs[0].shape.channel_axis == 3
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
                        inputs[0].shape.channel_axis == 1
                        or inputs[0].shape.channel_axis == 0
                    )
                    if inputs[0].shape.channel_axis == 1:
                        layerdef.blobs[0].shape.dim.append(ip.num_output)
                        layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
                    else:
                        layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
                        layerdef.blobs[0].shape.dim.append(ip.num_output)

                params = parser.param_to_tensor(layerdef)
                return parser.proxy.ip(
                    inputs,
                    params,
                    layerdef.name,
                    ip_relu_flag=True,
                    ip_num_output=ip.num_output,
                )

        class InnerProduct(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                ip = layerdef.inner_product_param
                axis = ip.axis
                assert len(inputs[0].shape.dims) == 4 or len(inputs[0].shape.dims) == 2
                if len(inputs[0].shape.dims) == 4:
                    layerdef.blobs[0].shape.ClearField("dim")
                    assert (
                        inputs[0].shape.channel_axis == 1
                        or inputs[0].shape.channel_axis == 3
                    )
                    if inputs[0].shape.channel_axis == 1:
                        layerdef.blobs[0].shape.dim.append(ip.num_output)
                        succeeding_axes = reduce(
                            lambda x, y: x * y, inputs[0].shape.dims[axis:]
                        )
                        layerdef.blobs[0].shape.dim.append(succeeding_axes)
                    else:
                        dims = map(lambda x: inputs[0].shape.dims[x], [0, 3, 1, 2])
                        layerdef.blobs[0].shape.dim.append(ip.num_output)
                        succeeding_axes = reduce(lambda x, y: x * y, dims[axis:])
                        layerdef.blobs[0].shape.dim.append(succeeding_axes)
                elif len(inputs[0].shape.dims) == 2:
                    layerdef.blobs[0].shape.ClearField("dim")
                    assert (
                        inputs[0].shape.channel_axis == 1
                        or inputs[0].shape.channel_axis == 0
                    )
                    if inputs[0].shape.channel_axis == 1:
                        layerdef.blobs[0].shape.dim.append(ip.num_output)
                        layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
                    else:
                        layerdef.blobs[0].shape.dim.extend(inputs[0].shape.dims[1:])
                        layerdef.blobs[0].shape.dim.append(ip.num_output)

                params = parser.param_to_tensor(layerdef)
                return parser.proxy.ip(
                    inputs,
                    params,
                    layerdef.name,
                    ip_num_output=ip.num_output,
                    ip_axis=ip.axis,
                )

        class ReLU(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                return parser.proxy.relu(inputs, layer_name=layerdef.name)

        class ReLU6(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                return parser.proxy.relu6(inputs, layer_name=layerdef.name)

        class PReLU(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                prelu = layerdef.prelu_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.prelu(
                    inputs, params, layerdef.name, prelu_share=prelu.channel_shared
                )

        class TanH(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                return parser.proxy.tanh(inputs, layer_name=layerdef.name)

        class Sigmoid(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                return parser.proxy.sigmoid(inputs, layer_name=layerdef.name)

        CaffeHelper.register_defaults = classmethod(lambda x: None)

        class Concat(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                concat = layerdef.concat_param
                return parser.proxy.concat(
                    inputs, layer_name=layerdef.name, concat_axis=concat.axis
                )

        class Reshape(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                reshape = layerdef.reshape_param
                return parser.proxy.reshape(
                    inputs,
                    layer_name=layerdef.name,
                    reshape_dims=reshape.shape.dim,
                    reshape_dim_size=len(reshape.shape.dim),
                    reshape_axis=reshape.axis,
                    reshape_num_axes=reshape.num_axes,
                )

        class Softmax(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                softmax = layerdef.softmax_param
                return parser.proxy.softmax(
                    inputs, layer_name=layerdef.name, softmax_axis=softmax.axis
                )

        class Transpose(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                transpose = layerdef.transpose_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.transpose(
                    inputs,
                    params,
                    layerdef.name,
                    transpose_dims=transpose.dim,
                    transpose_exgaxis=False,
                )

        class Interp(CaffeHelper.LayerParser):
            INTERP_NEAREST = 0
            INTERP_BILINEAR = 1

            @classmethod
            def parse(cls, parser, layerdef, inputs):
                interp = layerdef.interp_param
                p = {}
                if interp.HasField("height"):
                    p["interp_height"] = interp.height
                if interp.HasField("width"):
                    p["interp_width"] = interp.width
                if interp.HasField("zoom_factor"):
                    p["interp_zoom_factor"] = interp.zoom_factor
                if interp.HasField("shrink_factor"):
                    p["interp_shrink_factor"] = interp.shrink_factor
                if interp.HasField("scale_factor"):
                    raise "scale_factor for Interp not supported."
                if interp.HasField("pad_beg"):
                    p["interp_pad_beg"] = interp.pad_beg
                if interp.HasField("pad_end"):
                    p["interp_pad_end"] = interp.pad_end
                p["interp_type"] = Interp.INTERP_BILINEAR
                return parser.proxy.interp(inputs, layer_name=layerdef.name, **p)

        class NNInterp(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                interp = layerdef.nninterp_param
                p = {}
                if interp.HasField("height"):
                    p["interp_height"] = interp.height
                if interp.HasField("width"):
                    p["interp_width"] = interp.width
                if interp.HasField("zoom_factor"):
                    p["interp_zoom_factor"] = interp.zoom_factor
                if interp.HasField("shrink_factor"):
                    p["interp_shrink_factor"] = interp.shrink_factor
                if interp.HasField("scale_factor"):
                    raise "scale_factor for NNInterp not supported."
                if interp.HasField("pad_beg"):
                    p["interp_pad_beg"] = interp.pad_beg
                if interp.HasField("pad_end"):
                    p["interp_pad_end"] = interp.pad_end
                p["interp_type"] = Interp.INTERP_NEAREST
                return parser.proxy.interp(inputs, layer_name=layerdef.name, **p)

        class Slice(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                sli = layerdef.slice_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.slice(
                    inputs, params, slice_axis=sli.axis, slice_point=sli.slice_point
                )

        class ROIPooling(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pool = layerdef.roi_pooling_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.roipooling(
                    inputs,
                    params,
                    layerdef.name,
                    roipooling_pooled_height=pool.pooled_h,
                    roipooling_pooled_width=pool.pooled_w,
                    roipooling_spatial_scale=pool.spatial_scale,
                )

        class PSROIPooling(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pool = layerdef.psroi_pooling_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.psroipooling(
                    inputs,
                    params,
                    layerdef.name,
                    psroipooling_group_size=pool.group_size,
                    psroipooling_output_dim=pool.output_dim,
                    psroipooling_spatial_scale=pool.spatial_scale,
                )

        class ROIAlignPooling(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pool = layerdef.roi_align_pooling_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.roialignpooling(
                    inputs,
                    params,
                    layerdef.name,
                    roialignpooling_pooled_height=pool.pooled_h,
                    roialignpooling_pooled_width=pool.pooled_w,
                    roialignpooling_sample_num=pool.sample_num,
                    roialignpooling_spatial_scale=pool.spatial_scale,
                )

        class PODROIAlignPooling(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pool = layerdef.podroi_align_pooling_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.podroialignpooling(
                    inputs,
                    params,
                    layerdef.name,
                    podroialignpooling_pooled_height=pool.pooled_h,
                    podroialignpooling_pooled_width=pool.pooled_w,
                    podroialignpooling_sample_num=pool.sample_num,
                    podroialignpooling_spatial_scale=pool.spatial_scale,
                )

        class PSROIMaskPooling(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pool = layerdef.psroi_mask_pooling_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.psroimaskpooling(
                    inputs,
                    params,
                    layerdef.name,
                    psroimaskpooling_group_size=pool.group_size,
                    psroimaskpooling_output_dim=pool.output_dim,
                    psroimaskpooling_spatial_scale=pool.spatial_scale,
                    psroimaskpooling_roi_scale=pool.roi_scale,
                    psroimaskpooling_bin_scale=pool.bin_scale,
                )

        class Scale(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.scale(
                    inputs, params, scale_bias_term=True if len(params) > 1 else False
                )

        class HeatMap2Coord(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                heatmap = layerdef.heatmap_param
                coord_h = heatmap.coord_h
                coord_w = heatmap.coord_w
                reposition = heatmap.coord_reposition
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.heatmap2coord(
                    inputs,
                    params,
                    layerdef.name,
                    heatmap2coord_coord_h=coord_h,
                    heatmap2coord_coord_w=coord_w,
                    heatmap2coord_reposition=reposition,
                )

        class Exchange(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.exchange(inputs, params, layerdef.name)

        class ShuffleChannel(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                sc = layerdef.shuffle_channel_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.shufflechannel(
                    inputs, params, layerdef.name, shufflechannel_group=sc.group
                )

        class Pad(CaffeHelper.LayerParser):
            @classmethod
            def parse(cls, parser, layerdef, inputs):
                pad = layerdef.pad_param
                params = parser.param_to_tensor(layerdef)
                return parser.proxy.pad(
                    inputs, params, mode=pad.mode, value=pad.value, pads=pad.pads
                )


def merge_split(netdef):
    del_list = []
    for i in range(len(netdef.layer)):
        layer = netdef.layer[i]
        if layer.type == "Split":
            del_list.append(i)
            del_names = set(layer.top)
            for j in range(i + 1, len(netdef.layer)):
                for sp in del_names:
                    for k in range(len(netdef.layer[j].bottom)):
                        if netdef.layer[j].bottom[k] == sp:
                            netdef.layer[j].bottom[k] = layer.bottom[0]
                    if sp in netdef.layer[j].top:
                        del_names.remove(sp)
    del_list.reverse()
    for i in del_list:
        del netdef.layer[i]
    return netdef
