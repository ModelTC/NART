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
import numpy
from . import CaffeParser as CaffeParser


class CaffeQuantParser(CaffeParser):
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

    @classmethod
    def get_support_layer_list(cls):
        return list(map(lambda x: x, CaffeQuantParser.dispatch_list))

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

    def __init__(
        self,
        net,
        config={},
        data_generator=None,
        proxy=Proxy.ModuleProxy("default"),
        net_id=0,
    ):
        super(CaffeQuantParser, self).__init__(net, proxy=proxy)
        self.quant_param = self.convert_quant_param(config.get("quant_param"))

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

    def param_to_tensor(self, node):
        params = []
        for in_name in node.input:
            if in_name not in node.input_shape_dict:
                # weight
                p = self.net._weight[in_name]
                qparam_name = node.name + "_weight_" + str(in_name)
                qparam = self._get_quant_param(qparam_name)
                sp = (list(p.shape), 1 if len(p.shape) > 1 else -1, 0)

                dtype = ""
                if qparam.qtype == 1:
                    ddd = numpy.round(
                        numpy.array(p.data, dtype="float32") / qparam.alpha
                        + qparam.zero_point
                    ).clip(0, 2**qparam.bits - 1)
                    dtype = "u"
                else:
                    ddd = numpy.round(
                        numpy.array(p.data, dtype="float32") / qparam.alpha
                    ).clip(-(2 ** (qparam.bits - 1) - 1), 2 ** (qparam.bits - 1) - 1)
                if qparam.bits <= 8:
                    ddd = ddd.astype(dtype + "int8")
                elif qparam.bits <= 16:
                    ddd = ddd.astype(dtype + "int16")
                else:
                    ddd = ddd.astype(dtype + "int32")

                r = FakeTensor(data=ddd, shape=sp, dtype=str(ddd.dtype))
                params.append(r)
        return params

    def op_post(self, node, op):
        weights = []
        for i in node.input:
            prods = node.owning_graph.get_tensor_producer(i)
            if prods[0] == "input" or prods[0].op_type == "Constant":
                weights.append(i)
        ialpha = list(map(lambda x: self._get_quant_param(x).alpha, node.input))
        izp = list(map(lambda x: self._get_quant_param(x).zero_point, node.input))
        ibits = list(map(lambda x: self._get_quant_param(x).bits, node.input))
        walpha = list(
            map(
                lambda x: self._get_quant_param(node.name + "_weight_" + str(x)).alpha,
                range(len(weights)),
            )
        )
        wzp = list(
            map(
                lambda x: self._get_quant_param(
                    node.name + "_weight_" + str(x)
                ).zero_point,
                range(len(weights)),
            )
        )
        wbits = list(
            map(
                lambda x: self._get_quant_param(node.name + "_weight_" + str(x)).bits,
                range(len(weights)),
            )
        )
        oalpha = list(map(lambda x: self._get_quant_param(x).alpha, node.output))
        ozp = list(map(lambda x: self._get_quant_param(x).zero_point, node.output))
        obits = list(map(lambda x: self._get_quant_param(x).bits, node.output))
        qtype = self._get_quant_param(node.input[0]).qtype

        if len(ialpha) > 0:
            op.add_setting(CaffeQuantParser.IALPHA, "float32", ialpha)
            op.add_setting(CaffeQuantParser.IZERO_POINT, "uint8", izp)
            op.add_setting(CaffeQuantParser.IBITS, "uint8", ibits)
        if len(walpha) > 0:
            op.add_setting(CaffeQuantParser.WALPHA, "float32", walpha)
            op.add_setting(CaffeQuantParser.WZERO_POINT, "uint8", wzp)
            op.add_setting(CaffeQuantParser.WBITS, "uint8", wbits)
        if len(oalpha) > 0:
            op.add_setting(CaffeQuantParser.OALPHA, "float32", oalpha)
            op.add_setting(CaffeQuantParser.OZERO_POINT, "uint8", ozp)
            op.add_setting(CaffeQuantParser.OBITS, "uint8", obits)
        op.add_setting(CaffeQuantParser.QTYPE, "int32", qtype)
        op.set_group_code(0x0006)
        return op

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        self.before_parse()

        into_parade_outputs = {}
        for t in into_parade.tensors:
            into_parade_outputs[t.name] = t
        dct_name_tensor = self.parse_input_dict()
        dct_name_tensor.update(into_parade_outputs)

        bottoms = set()
        for l in self.net.nodes:
            bottoms.update(l.input)

        # gen input
        for b in bottoms:
            if b in dct_name_tensor and dct_name_tensor[b].dtype == "float32":
                qparam = self._get_quant_param(b)
                [quant,] = into_parade.append(
                    Fakes.Quantize(
                        [dct_name_tensor[b]],
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
        for t in into_parade.outputs:
            if t.dtype != "float32":
                qparam = self._get_quant_param(t.name)
                [dequant,] = into_parade.append(
                    Fakes.Dequantize(
                        [dct_name_tensor[t.name]],
                        [qparam.alpha],
                        [qparam.zero_point],
                        [qparam.bits],
                        qparam.qtype,
                    )
                )
                dct_name_tensor[t.name] = dequant
                dequant.name = t.name
        return into_parade

    @classmethod
    def register_defaults(cls):
        pass


class Conv(OpDesc, parser=CaffeQuantParser):
    pass


class ConvTranspose(OpDesc, parser=CaffeQuantParser):
    pass


class AveragePool(OpDesc, parser=CaffeQuantParser):
    pass


class MaxPool(OpDesc, parser=CaffeQuantParser):
    pass


class BatchNormalization(OpDesc, parser=CaffeQuantParser):
    pass


class CaffeScale(OpDesc, parser=CaffeQuantParser):
    pass


class Mul(OpDesc, parser=CaffeQuantParser):
    pass


class Add(OpDesc, parser=CaffeQuantParser):
    pass


class Sub(OpDesc, parser=CaffeQuantParser):
    pass


class Max(OpDesc, parser=CaffeQuantParser):
    pass


class Relu(OpDesc, parser=CaffeQuantParser):
    pass


class PRelu(OpDesc, parser=CaffeQuantParser):
    pass


class Sigmoid(OpDesc, parser=CaffeQuantParser):
    pass


class Softmax(OpDesc, parser=CaffeQuantParser):
    pass


class CaffeSoftmax(OpDesc, parser=CaffeQuantParser):
    pass


class Flatten(OpDesc, parser=CaffeQuantParser):
    pass


class Concat(OpDesc, parser=CaffeQuantParser):
    pass


class Gemm(OpDesc, parser=CaffeQuantParser):
    pass


class Upsample(OpDesc, parser=CaffeQuantParser):
    pass
