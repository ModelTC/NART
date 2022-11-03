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
from ..core.art import Proxy
from ..core.art import FakeParade
from ..core.art import FakeTensor
from ..core.art import Fakes
from ..core.art import Dtype
from ..ops.op import Op, OpDesc
from onnx import numpy_helper
import numpy
from . import CaffeParser as CaffeParser


class CaffeCUDAQuantParser(CaffeParser, module_name="cuda_quant"):
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

    dispatch_dct = {}

    @classmethod
    def get_support_layer_list(cls):
        return list(map(lambda x: x, CaffeCUDAQuantParser.dispatch_dct))

    class LayerParser:
        @classmethod
        def __init_subclass__(cls, layertype=None, overwrite=False, **kwargs):
            if layertype is None:
                layertype = cls.__name__
            if cls.parse is CaffeCUDAQuantParser.LayerParser.parse:
                raise
            if overwrite is False and layertype in CaffeCUDAQuantParser.dispatch_dct:
                raise
            CaffeCUDAQuantParser.dispatch_dct[layertype] = cls
            super().__init_subclass__(**kwargs)

        @classmethod
        def parse(cls, parser, layerdef, inputs):
            pass

    class QuantParam:
        DEFAULT = 0
        BIASED = 1
        SYMMETRIC = 2

        def __init__(self, alpha=1.0, zero_point=0, bits=8, qtype=0):
            self.alpha = alpha
            self.zero_point = zero_point
            self.bits = bits
            self.qtype = qtype

        def to_biased(self):
            if self.qtype == QuantParam.BIASED or self.zero_point != 0:
                return self
            return QuantParam(
                self.alpha, 2 ** (self.bits - 1), self.bits, QuantParam.SYMMETRIC
            )

    def __init__(self, net, config=None, data_generator=None, net_id=0):
        proxy = Proxy.ModuleProxy("default")
        super(CaffeCUDAQuantParser, self).__init__(net, proxy=proxy)
        self.quant_param = self.convert_quant_param(config["quant_param"])

    def convert_quant_param(self, quant_param=None):
        if quant_param is None:
            return None
        res = {}
        for k, v in quant_param.items():
            # {
            #   "max": float,           ;optional
            #   "min": float,           ;optional
            #
            #   "alpha": float,         ;optional
            #   "zero_point": int,      ;optional
            #
            #   "bit": int,             ;required
            #   "type": string,         ;required
            # }
            if "type" not in v:
                raise RuntimeError(f"`type` is required for quantization [{k}]")
            if v["type"] not in ["symmetric", "biased"]:
                raise RuntimeError(
                    f"`type` can be `symmetric` or `biased`, but got {v['type']} for [{k}]"
                )
            qtp = (
                self.QuantParam.SYMMETRIC
                if v["type"] == "symmetric"
                else self.QuantParam.BIASED
            )

            if "bit" not in v:
                raise RuntimeError(f"`bit` is required for quantization [{k}]")
            bit = v["bit"]

            if "alpha" in v:
                alpha = v["alpha"]
                if qtp == self.QuantParam.SYMMETRIC:
                    zp = 0
                zp = v["zero_point"]
            elif "max" in v and "min" in v:
                dmax = v["max"]
                dmin = v["min"]
                dmax = max(dmax, 0.0)
                dmin = min(dmin, 0.0)
                if qtp == self.QuantParam.SYMMETRIC:
                    dmax = max(dmax, -dmin)
                    alpha = dmax / (2 ** (bit - 1) - 1)
                    zp = 0
                else:
                    alpha = (dmax - dmin) / (2**bit - 1)
                    zp = round(-alpha * dmin)

            else:
                raise RuntimeError(
                    f"`max, min` or `alpha, zero_point` is required for quantization [{k}]"
                )

            res[k] = self.QuantParam(alpha, zp, bit, qtp)
        return res

    def _get_quant_param(self, name):
        if name in self.quant_param:
            return self.quant_param[name]
        else:
            print(
                f"WARN: {name} not founded in given QuantParam, use default value instead."
            )
            return self.QuantParam()

    # todo: only support conv now
    def param_to_tensor(self, node, params):
        assert node.op_type == "Conv"
        qparams = []

        for i in range(2):
            p = params[i]
            p_shape = tuple(p.shape.dims)
            qparam_name = node.name + "_param_" + str(i + 1)
            qparam = self._get_quant_param(qparam_name)
            sp = (list(p_shape), 1 if len(p_shape) > 1 else -1, 0)

            dtype = ""
            if i == 0:  # weight
                if qparam.qtype == 1:
                    ddd = numpy.round(
                        numpy.array(p.data, dtype="float32") / qparam.alpha
                        + qparam.zero_point
                    ).clip(0, 2**qparam.bits - 1)
                    dtype = "u"
                else:
                    # 8-bit qunatize to [-128, 127]; 4-bit quantize to [-8, 7]
                    ddd = numpy.round(
                        numpy.array(p.data, dtype="float32") / qparam.alpha
                    ).clip(-(2 ** (qparam.bits - 1)), 2 ** (qparam.bits - 1) - 1)
                if qparam.bits <= 8:
                    ddd = ddd.astype(dtype + "int8")
                elif qparam.bits <= 16:
                    ddd = ddd.astype(dtype + "int16")
                else:
                    ddd = ddd.astype(dtype + "int32")

                # the data of ddd is transposed from NCHW to NHWC
                # however, the shape of ddd is still NCHW
                if len(p_shape) > 1:
                    ddd = ddd.reshape(p_shape)
                    ddd = ddd.transpose([0, 2, 3, 1])

                    # int8 to int4
                    if qparam.bits == 4:

                        def int8toint4(data_int8):
                            shape = data_int8.shape
                            half_size = data_int8.size // 2
                            data_int8 = data_int8.flatten()
                            data_int8 = data_int8.reshape(half_size, 2).transpose()
                            data_int4 = numpy.zeros(data_int8.size, dtype="int8")
                            data_int4[:half_size] = (data_int8[0] & 0xF) | (
                                data_int8[1] << 4
                            )
                            return data_int4.reshape(shape)

                        ddd = int8toint4(ddd)

            else:  # bias: use float bias, ddd = bias / (input_scale * weight_scale)
                ddd = numpy.array(p.data, dtype="float32") / qparam.alpha

            r = FakeTensor(data=ddd, shape=sp, dtype=str(ddd.dtype))
            qparams.append(r)

        return qparams

    def parse_network_input_dict(self):
        """parse network input dict

        Returns:
            a dict, name to FakeTensor.

        """
        res = {}
        for t in self.net.network_inputs:
            sp = self.net.get_tensor_shape(t)
            if t not in self.net.initializer:
                tensor = FakeTensor(
                    dtype=Dtype.Float32, shape=((list(sp)), 1, 0), name=t
                )
            else:
                tensor = FakeTensor(
                    data=numpy_helper.to_array(self.net.initializer[t]),
                    dtype=Dtype.Float32,
                    shape=(sp),
                )
            res[t] = tensor
        return res

    def parse_input_dict(self):
        """parse input dict

        Returns:
            a dict, name to FakeTensor.

        """
        res = {}
        for node in self.net.nodes:
            for t in node.input:
                sp = self.net.get_tensor_shape(t)
                if t not in self.net.initializer:
                    tensor = FakeTensor(
                        dtype=Dtype.Float32, shape=((list(sp)), 1, 0), name=t
                    )
                else:
                    tensor = FakeTensor(
                        data=numpy_helper.to_array(self.net.initializer[t]),
                        dtype=Dtype.Float32,
                        shape=(sp),
                    )
                res[t] = tensor
        return res

    def op_post(self, node, op):
        ialpha = list(map(lambda x: self._get_quant_param(x).alpha, node.input))
        izp = list(map(lambda x: self._get_quant_param(x).zero_point, node.input))
        ibits = list(map(lambda x: self._get_quant_param(x).bits, node.input))

        # todo: verify support for weight perchannel quantize
        if node.op_type == "Conv" or node.op_type == "ConvolutionReLU":
            [
                layer,
            ] = self.trans_onnx_to_caffe(node)
            num_output = layer.convolution_param.num_output
            weight_param = self._get_quant_param(node.name + "_param_1")
            walpha = [weight_param.alpha] * num_output
            wzp = [weight_param.zero_point] * num_output
            wbits = [weight_param.bits]
        else:
            walpha = []
            wzp = []
            wbits = []

        oalpha = list(map(lambda x: self._get_quant_param(x).alpha, node.output))
        ozp = list(map(lambda x: self._get_quant_param(x).zero_point, node.output))
        obits = list(map(lambda x: self._get_quant_param(x).bits, node.output))
        qtype = self._get_quant_param(node.input[0]).qtype

        if len(ialpha) > 0:
            op.add_setting(CaffeCUDAQuantParser.IALPHA, "float32", ialpha)
            op.add_setting(CaffeCUDAQuantParser.IZERO_POINT, "uint8", izp)
            op.add_setting(CaffeCUDAQuantParser.IBITS, "uint8", ibits)
        if len(walpha) > 0:
            op.add_setting(CaffeCUDAQuantParser.WALPHA, "float32", walpha)
            op.add_setting(CaffeCUDAQuantParser.WZERO_POINT, "uint8", wzp)
            op.add_setting(CaffeCUDAQuantParser.WBITS, "uint8", wbits)
        if len(oalpha) > 0:
            op.add_setting(CaffeCUDAQuantParser.OALPHA, "float32", oalpha)
            op.add_setting(CaffeCUDAQuantParser.OZERO_POINT, "uint8", ozp)
            op.add_setting(CaffeCUDAQuantParser.OBITS, "uint8", obits)
        op.add_setting(CaffeCUDAQuantParser.QTYPE, "int32", qtype)
        op.set_group_code(0x000F)
        return op

    def parse_layers(self, net, dct_name_tensor, parade):
        relu_fused = []
        for node in net.nodes:
            outputs = node.output.copy()
            if node.op_type == "Conv" or node.op_type == "Eltwise":
                relu_flag = False
                for next_node in net.nodes:
                    if (
                        node.output[0] == next_node.input[0]
                        and next_node.op_type == "Relu"
                    ):
                        relu_flag = True
                        relu_fused.append(next_node.name)
                        outputs[0] = next_node.output[0]
                        break

                node_type = (
                    "ConvolutionRelu" if node.op_type == "Conv" else "EltwiseRelu"
                )
                res = CaffeCUDAQuantParser.dispatch_dct[node_type]().parse(
                    self,
                    node,
                    list(map(lambda x: dct_name_tensor[x], node.input)),
                    relu_flag,
                )
            elif node.op_type == "Relu":
                if node.name in relu_fused:
                    continue
                raise RuntimeError(
                    f"Single Relu layer is not supported now in cuda_quant"
                )
            else:
                res = CaffeParser.dispatch_dct[node.op_type]().parse(
                    self, node, list(map(lambda x: dct_name_tensor[x], node.input))
                )

            if isinstance(res, (tuple, list)):
                op, outputs = res
            else:
                op = res
                # outputs = node.output.copy()
            op = self.op_post(node, op)
            res_list = parade.append(op)
            assert len(res_list) == len(outputs)
            for name, tensor in zip(outputs, res_list):
                tensor.name = name
            dct_name_tensor.update({k: v for k, v in zip(outputs, res_list)})

        return parade

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        self.before_parse()

        # tensors for network input
        into_parade_outputs = {}
        for t in into_parade.outputs:
            into_parade_outputs[t.name] = t
        input_dct_name_tensor = self.parse_network_input_dict()
        input_dct_name_tensor.update(into_parade_outputs)

        # all tensor, including bottom, weight and bias
        dct_name_tensor = self.parse_input_dict()

        # quantize input
        for b in self.net.input:
            if b in self.net.initializer:
                continue
            if input_dct_name_tensor[b].dtype == "float32":
                qparam = self._get_quant_param(b)
                [quant,] = into_parade.append(
                    Fakes.CuQuantize(
                        [input_dct_name_tensor[b]],
                        [qparam.alpha],
                        [qparam.zero_point],
                        [qparam.bits],
                        qparam.qtype,
                    )
                )
                dct_name_tensor[b] = quant
                quant.name = b

        self.parse_layers(self.net, dct_name_tensor, into_parade)

        # dequantize output
        for t_name in self.net.output:
            output_tensor = dct_name_tensor[t_name]
            if output_tensor.dtype != "float32":
                qparam = self._get_quant_param(output_tensor.name)
                [dequant,] = into_parade.append(
                    Fakes.CuDequantize(
                        [dct_name_tensor[output_tensor.name]],
                        [qparam.alpha],
                        [qparam.zero_point],
                        [qparam.bits],
                        qparam.qtype,
                    )
                )
                dct_name_tensor[output_tensor.name] = dequant
                dequant.name = output_tensor.name

        return into_parade

    @classmethod
    def register_defaults(cls):
        pass

    @classmethod
    def gen_input_workspace(cls):
        return "cuda_quant"


def split_input_param(inputs):
    inp = []
    param = []
    for i in inputs:
        if i.data is not None:
            param.append(i)
        else:
            inp.append(i)
    return inp, param


class Conv(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class ConvolutionRelu(
    CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser
):
    @classmethod
    def parse(cls, parser, node, inputs, relu_flag):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        conv_descriptor = layer.convolution_param
        inputs, params = split_input_param(inputs)
        params = parser.param_to_tensor(node, params)
        return Fakes.CuConvolutionRelu(
            conv_descriptor.bias_term,
            relu_flag,
            conv_descriptor.num_output,
            (conv_descriptor.kernel_h, conv_descriptor.kernel_w),
            (conv_descriptor.stride_h, conv_descriptor.stride_w),
            (conv_descriptor.pad_h, conv_descriptor.pad_w),
            conv_descriptor.group,
            inputs,
            params,
        )


class Eltwise(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class EltwiseRelu(
    CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser
):
    @classmethod
    def parse(cls, parser, node, inputs, relu_flag):
        [
            layer,
        ] = parser.trans_onnx_to_caffe(node)
        return Fakes.CuEltwiseRelu(layer.eltwise_param, relu_flag, inputs)


class ConvTranspose(
    CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser
):
    pass


class AveragePool(
    CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser
):
    pass


class AveragePool_10(
    CaffeCUDAQuantParser.LayerParser,
    OpDesc,
    parser=CaffeCUDAQuantParser,
    op_type="AveragePool",
    version=10,
):
    pass


class MaxPool(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class MaxPool_10(
    CaffeCUDAQuantParser.LayerParser,
    OpDesc,
    parser=CaffeCUDAQuantParser,
    op_type="MaxPool",
    version=10,
):
    pass


class BatchNormalization(
    CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser
):
    pass


class CaffeScale(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Mul(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Add(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Sub(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Max(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Relu(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class PRelu(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Sigmoid(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Softmax(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class CaffeSoftmax(
    CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser
):
    pass


class Flatten(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Concat(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Gemm(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass


class Upsample(CaffeCUDAQuantParser.LayerParser, OpDesc, parser=CaffeCUDAQuantParser):
    pass
