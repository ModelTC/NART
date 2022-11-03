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

from ... import _nart
from . import FakeOp
from . import FakeTensor
import numpy


class Conv2D(FakeOp):
    def __init__(
        self,
        bias,
        relu,
        kernel,
        stride,
        pad,
        channel_in,
        channel_out,
        group,
        inputs,
        params,
    ):
        super(Conv2D, self).__init__("conv_2d", inputs, params)
        trans = lambda ii: (ii, ii) if not isinstance(ii, (tuple, list)) else tuple(ii)
        assert isinstance(bias, bool)
        assert isinstance(relu, bool)
        assert isinstance(channel_in, int)
        assert isinstance(channel_out, int)
        assert isinstance(group, int)
        self.kernel_h, self.kernel_w = trans(kernel)
        self.pad_h, self.pad_w = trans(pad)
        self.stride_h, self.stride_w = trans(stride)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.group = group
        self.relu = relu
        self.bias = bias


class ReLU(FakeOp.FakeOpInterface):
    def op_tp_code(self):
        return 0x0101

    def infer_shape(self, inputs):
        return [(inputs[0].dtype, inputs[0].shape)]


class PSROIPooling(FakeOp.FakeOpInterface):
    SPATIAL_SCALE = 1
    OUTPUT_DIM = 2
    GROUP_SIZE = 3
    SAMPLE_NUM = 4

    def __init__(self, inputs, spatial_scale, output_dim, group_size):
        super(PSROIPooling, self).__init__(inputs)
        self.add_setting(PSROIPooling.SPATIAL_SCALE, "float32", spatial_scale)
        self.add_setting(PSROIPooling.OUTPUT_DIM, "uint32", output_dim)
        self.add_setting(PSROIPooling.GROUP_SIZE, "uint32", group_size)

    def op_tp_code(self):
        return 0x0111

    def infer_shape(self, inputs):
        assert len(inputs[0].shape.dims) == 4
        assert inputs[0].shape.batch_axis == 0
        assert inputs[0].shape.channel_axis == 1 or inputs[0].shape.channel_axis == 3
        dtype = inputs[0].dtype
        b = inputs[1].shape.dims[inputs[0].shape.batch_axis]
        c = inputs[0].shape.dims[inputs[0].shape.channel_axis]
        if inputs[0].shape.channel_axis == 1:
            height, width = inputs[0].shape.dims[2], inputs[0].shape.dims[3]
            shape = FakeTensor.Shape(
                [b, c, height, width],
                inputs[0].shape.channel_axis,
                inputs[0].shape.batch_axis,
            )
        else:
            height, width = inputs[0].shape.dims[1], inputs[0].shape.dims[2]
            shape = FakeTensor.Shape(
                [b, height, width, c],
                inputs[0].shape.channel_axis,
                inputs[0].shape.batch_axis,
            )
        return [(dtype, shape)]


class ROIAlignPooling(FakeOp.FakeOpInterface):
    SPATIAL_SCALE = 1
    POOLED_H = 2
    POOLED_W = 3
    SAMPLE_NUM = 4

    def __init__(self, inputs, spatial_scale, pooled_h, pooled_w, sample_num):
        super(ROIAlignPooling, self).__init__(inputs)
        self.add_setting(ROIAlignPooling.SPATIAL_SCALE, "float32", spatial_scale)
        self.add_setting(ROIAlignPooling.POOLED_H, "uint32", pooled_h)
        self.add_setting(ROIAlignPooling.POOLED_W, "uint32", pooled_w)
        self.add_setting(ROIAlignPooling.SAMPLE_NUM, "uint32", sample_num)
        self.pooled_h, self.pooled_w = pooled_h, pooled_w

    def op_tp_code(self):
        return 0x0112

    def infer_shape(self, inputs):
        assert len(inputs[0].shape.dims) == 4
        assert inputs[0].shape.batch_axis == 0
        assert inputs[0].shape.channel_axis == 1 or inputs[0].shape.channel_axis == 3
        dtype = inputs[0].dtype
        b = inputs[1].shape.dims[inputs[0].shape.batch_axis]
        c = inputs[0].shape.dims[inputs[0].shape.channel_axis]
        height, width = self.pooled_h, self.pooled_w
        if inputs[0].shape.channel_axis == 1:
            # height, width = inputs[0].shape.dims[2], inputs[0].shape.dims[3]
            shape = FakeTensor.Shape(
                [b, c, height, width],
                inputs[0].shape.channel_axis,
                inputs[0].shape.batch_axis,
            )
        else:
            # height, width = inputs[0].shape.dims[1], inputs[0].shape.dims[2]
            shape = FakeTensor.Shape(
                [b, height, width, c],
                inputs[0].shape.channel_axis,
                inputs[0].shape.batch_axis,
            )
        return [(dtype, shape)]


class Quantize(FakeOp.FakeOpInterface):
    IALPHA = 0x00060101
    IZERO_POINT = 0x00060102
    IBITS = 0x00060103
    WALPHA = 0x00060104
    WZERO_POINT = 0x00060105
    WBITS = 0x00060106
    OALPHA = 0x00060107
    OZERO_POINT = 0x00060108
    OBITS = 0x00060109
    QTYPE = 0x0006010A

    QTYPE_DEFAULT = 0
    QTYPE_BIASED = 1
    QTYPE_SYMMETRIC = 2
    QTYPE_MAP = {"default": 0, "biased": 1, "symmetric": 2}

    def __init__(self, inputs, oalpha, ozero_point, bits=[8], qtype="default"):
        super(Quantize, self).__init__(inputs)
        if qtype is str:
            qtype = QTYPE_MAP[qtype]
        self.add_setting(Quantize.OALPHA, "float32", oalpha)
        self.add_setting(Quantize.OZERO_POINT, "uint8", ozero_point)
        self.add_setting(Quantize.OBITS, "uint8", bits)
        self.add_setting(Quantize.QTYPE, "int32", qtype)

        self.bits = bits
        self.qtype = qtype

    def op_tp_code(self):
        return 0x000680000000

    def infer_shape(self, inputs):
        assert len(inputs) == 1

        if self.bits[0] <= 8:
            if self.qtype == 0 or self.qtype == 1:
                return [("uint8", inputs[0].shape)]
            else:
                return [("int8", inputs[0].shape)]
        elif self.bits[0] <= 16:
            if self.qtype == 1:
                return [("uint16", inputs[0].shape)]
            else:
                return [("int16", inputs[0].shape)]
        else:
            if self.qtype == 1:
                return [("uint32", inputs[0].shape)]
            else:
                return [("int32", inputs[0].shape)]


class Dequantize(FakeOp.FakeOpInterface):
    QTYPE_DEFAULT = 0
    QTYPE_BIASED = 1
    QTYPE_SYMMETRIC = 2
    QTYPE_MAP = {"default": 0, "biased": 1, "symmetric": 2}

    def __init__(self, inputs, ialpha, izero_point, bits=[8], qtype="default"):
        super(Dequantize, self).__init__(inputs)
        if qtype is str:
            qtype = QTYPE_MAP[qtype]
        self.add_setting(Quantize.IALPHA, "float32", ialpha)
        self.add_setting(Quantize.IZERO_POINT, "uint8", izero_point)
        self.add_setting(Quantize.IBITS, "uint8", bits)
        self.add_setting(Quantize.QTYPE, "int32", qtype)

        self.bits = bits
        self.qtype = qtype

    def op_tp_code(self):
        return 0x000680000001

    def infer_shape(self, inputs):
        assert len(inputs) == 1
        return [("float32", inputs[0].shape)]


class CuQuantize(FakeOp.FakeOpInterface):
    IALPHA = 0x000F0101
    IZERO_POINT = 0x000F0102
    IBITS = 0x000F0103
    WALPHA = 0x000F0104
    WZERO_POINT = 0x000F0105
    WBITS = 0x000F0106
    OALPHA = 0x000F0107
    OZERO_POINT = 0x000F0108
    OBITS = 0x000F0109
    QTYPE = 0x000F010A

    QTYPE_DEFAULT = 0
    QTYPE_BIASED = 1
    QTYPE_SYMMETRIC = 2
    QTYPE_MAP = {"default": 0, "biased": 1, "symmetric": 2}

    def __init__(self, inputs, oalpha, ozero_point, bits=[8], qtype="default"):
        super(CuQuantize, self).__init__(inputs)
        if qtype is str:
            qtype = QTYPE_MAP[qtype]
        self.add_setting(CuQuantize.OALPHA, "float32", oalpha)
        self.add_setting(CuQuantize.OZERO_POINT, "uint8", ozero_point)
        self.add_setting(CuQuantize.OBITS, "uint8", bits)
        self.add_setting(CuQuantize.QTYPE, "int32", qtype)

        self.bits = bits
        self.qtype = qtype

    def op_tp_code(self):
        return 0x000F80000000

    def infer_shape(self, inputs):
        assert len(inputs) == 1

        if self.bits[0] <= 8:
            if self.qtype == 0 or self.qtype == 1:
                return [("uint8", inputs[0].shape)]
            else:
                return [("int8", inputs[0].shape)]
        elif self.bits[0] <= 16:
            if self.qtype == 1:
                return [("uint16", inputs[0].shape)]
            else:
                return [("int16", inputs[0].shape)]
        else:
            if self.qtype == 1:
                return [("uint32", inputs[0].shape)]
            else:
                return [("int32", inputs[0].shape)]


class CuConvolutionRelu(FakeOp.FakeOpInterface):
    NUM_OUTPUT = 1
    PAD_H = 2
    PAD_W = 3
    KERNEL_H = 4
    KERNEL_W = 5
    STRIDE_H = 6
    STRIDE_W = 7
    GROUP = 8
    RELU_FLAG = 9

    def __init__(
        self, bias, relu_flag, output_num, kernel, stride, pad, group, inputs, params
    ):
        super(CuConvolutionRelu, self).__init__(inputs, params)
        trans = lambda ii: (ii, ii) if not isinstance(ii, (tuple, list)) else tuple(ii)
        assert isinstance(bias, bool)
        assert isinstance(relu_flag, bool)
        assert isinstance(group, int)
        self.kernel_h, self.kernel_w = trans(kernel)
        self.pad_h, self.pad_w = trans(pad)
        self.stride_h, self.stride_w = trans(stride)
        self.group = group
        self.relu = relu_flag
        self.bias = bias
        self.output_num = output_num

        self.add_setting(CuConvolutionRelu.NUM_OUTPUT, "uint32", self.output_num)
        self.add_setting(CuConvolutionRelu.PAD_H, "uint32", self.pad_h)
        self.add_setting(CuConvolutionRelu.PAD_W, "uint32", self.pad_w)
        self.add_setting(CuConvolutionRelu.KERNEL_H, "uint32", self.kernel_h)
        self.add_setting(CuConvolutionRelu.KERNEL_W, "uint32", self.kernel_w)
        self.add_setting(CuConvolutionRelu.STRIDE_H, "uint32", self.stride_h)
        self.add_setting(CuConvolutionRelu.STRIDE_W, "uint32", self.stride_w)
        self.add_setting(CuConvolutionRelu.GROUP, "uint32", self.group)
        self.add_setting(CuConvolutionRelu.RELU_FLAG, "bool", self.relu)

    def op_tp_code(self):
        return 0x0100

    def infer_shape(self, inputs):
        assert len(inputs[0].shape.dims) == 4
        assert inputs[0].shape.batch_axis == 0
        assert inputs[0].shape.channel_axis == 1 or inputs[0].shape.channel_axis == 3
        dtype = inputs[0].dtype

        b = inputs[0].shape.dims[inputs[0].shape.batch_axis]
        c = inputs[0].shape.dims[inputs[0].shape.channel_axis]
        o_c = self.output_num
        if inputs[0].shape.channel_axis == 1:
            height, width = inputs[0].shape.dims[2], inputs[0].shape.dims[3]
            o_h = (height + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
            o_w = (width + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
            shape = FakeTensor.Shape(
                [b, o_c, o_h, o_w],
                inputs[0].shape.channel_axis,
                inputs[0].shape.batch_axis,
            )
        else:
            height, width = inputs[0].shape.dims[1], inputs[0].shape.dims[2]
            o_h = (height + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
            o_w = (width + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
            shape = FakeTensor.Shape(
                [b, o_h, o_w, o_c],
                inputs[0].shape.channel_axis,
                inputs[0].shape.batch_axis,
            )
        return [(dtype, shape)]


class CuEltwiseRelu(FakeOp.FakeOpInterface):
    OPERATION = 1
    COEFF = 2
    RELU_FLAG = 3

    def __init__(self, eltwise_param, relu_flag, inputs):
        super(CuEltwiseRelu, self).__init__(inputs)
        mp_op = {eltwise_param.MAX: 2, eltwise_param.PROD: 0, eltwise_param.SUM: 1}

        assert len(eltwise_param.coeff) > 0
        self.operation = mp_op[eltwise_param.operation]
        self.coeff = list(eltwise_param.coeff)
        self.relu = relu_flag

        self.add_setting(CuEltwiseRelu.OPERATION, "uint32", self.operation)
        self.add_setting(CuEltwiseRelu.COEFF, "float32", self.coeff)
        self.add_setting(CuEltwiseRelu.RELU_FLAG, "uint32", self.relu)

    def op_tp_code(self):
        return 0x010C

    def infer_shape(self, inputs):
        assert isinstance(inputs, list)
        assert len(inputs) >= 2
        for i in range(len(inputs) - 1):
            # print(inputs[i].shape, inputs[i+1].shape, inputs[i].shape.dims == inputs[i+1].shape.dims)
            assert inputs[i].shape.dims == inputs[i + 1].shape.dims
            assert inputs[i].dtype == inputs[i + 1].dtype
        dtype = inputs[0].dtype
        return [(dtype, inputs[0].shape)]


class CuDequantize(FakeOp.FakeOpInterface):
    QTYPE_DEFAULT = 0
    QTYPE_BIASED = 1
    QTYPE_SYMMETRIC = 2
    QTYPE_MAP = {"default": 0, "biased": 1, "symmetric": 2}

    def __init__(self, inputs, ialpha, izero_point, bits=[8], qtype="default"):
        super(CuDequantize, self).__init__(inputs)
        if qtype is str:
            qtype = QTYPE_MAP[qtype]
        self.add_setting(CuQuantize.IALPHA, "float32", ialpha)
        self.add_setting(CuQuantize.IZERO_POINT, "uint8", izero_point)
        self.add_setting(CuQuantize.IBITS, "uint8", bits)
        self.add_setting(CuQuantize.QTYPE, "int32", qtype)

        self.bits = bits
        self.qtype = qtype

    def op_tp_code(self):
        return 0x000F80000001

    def infer_shape(self, inputs):
        assert len(inputs) == 1
        return [("float32", inputs[0].shape)]


class TENSORRTNet(FakeOp.FakeOpInterface):
    TENSORRT_NET = 0x0100
    TENSORRT_OUTPUTS = 0x0101
    TENSORRT_INPUTS = 0x0102

    def __init__(self, inputs, netdef, outputs):
        # inputs is tensors
        # outputs is tensors
        # super(TENSORRTNet, self).__init__(list(map(lambda x:inputs[x], inputs)))
        super(TENSORRTNet, self).__init__(inputs)
        self.outputs = outputs
        assert len(inputs) != 0, "tensorrt net no inputs"
        assert len(outputs) != 0, "tensorrt net no outputs"
        self.add_setting(TENSORRTNet.TENSORRT_NET, "uint8", netdef)
        self.add_setting(
            TENSORRTNet.TENSORRT_OUTPUTS, "string", list(map(lambda x: x.name, outputs))
        )
        self.add_setting(
            TENSORRTNet.TENSORRT_INPUTS, "string", list(map(lambda x: x.name, inputs))
        )

    def op_tp_code(self):
        return 0x000D80000000

    def infer_shape(self, inputs):
        return list(map(lambda x: (x.dtype, x.shape), self.outputs))
