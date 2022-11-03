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
from .default import CaffeParser
from ..ops.op import Op, OpDesc


class CaffeCUDAParser(CaffeParser, module_name="cuda"):
    dispatch_list = []

    @classmethod
    def get_support_layer_list(clf):
        return list(map(lambda x: x, CaffeCUDAParser.dispatch_list))

    def __init__(self, net, config={}, proxy=Proxy.ModuleProxy("default"), **kwargs):
        super(CaffeCUDAParser, self).__init__(net, config=config, proxy=proxy)

    def op_post(self, net, op):
        op.set_group_code(0x0002)
        return op

    @classmethod
    def register_defaults(cls):
        pass

    @classmethod
    def register_layer(cls, layer_list):
        for layer in layer_list:
            if layer not in cls.dispatch_list:
                cls.dispatch_list.append(layer)

    @classmethod
    def gen_workspace(cls):
        return {}

    @classmethod
    def gen_input_workspace(cls):
        return "cuda"

    @classmethod
    def get_passes(cls, backend_specialized=False):
        ret = super().get_passes()

        if backend_specialized:
            from .. import passes

            ret.extend([passes.FuseAddDivClipCast(), passes.FuseClipCast()])
        return ret

    def parse(self, into_parade=None):
        # call backend specialized passes
        for item in self.get_passes(backend_specialized=True):
            item.run(self.net)

        return super().parse(into_parade)


class Add(OpDesc, parser=CaffeCUDAParser):
    pass


class Abs(OpDesc, parser=CaffeCUDAParser):
    pass


class Sub(OpDesc, parser=CaffeCUDAParser):
    pass


class Mul(OpDesc, parser=CaffeCUDAParser):
    pass


class Div(OpDesc, parser=CaffeCUDAParser):
    pass


class Pow(OpDesc, parser=CaffeCUDAParser):
    pass


class Sqrt(OpDesc, parser=CaffeCUDAParser):
    pass


class Exp(OpDesc, parser=CaffeCUDAParser):
    pass


class Log(OpDesc, parser=CaffeCUDAParser):
    pass


class Tanh(OpDesc, parser=CaffeCUDAParser):
    pass


class ReduceMax(OpDesc, parser=CaffeCUDAParser):
    pass


class ReduceMin(OpDesc, parser=CaffeCUDAParser):
    pass


class ReduceMean(OpDesc, parser=CaffeCUDAParser):
    pass


class ReduceProd(OpDesc, parser=CaffeCUDAParser):
    pass


class ReduceSum(OpDesc, parser=CaffeCUDAParser):
    pass


class ReduceL2(OpDesc, parser=CaffeCUDAParser):
    pass


class MatMul(OpDesc, parser=CaffeCUDAParser):
    pass


class Max(OpDesc, parser=CaffeCUDAParser):
    pass


class MaxPool(OpDesc, parser=CaffeCUDAParser):
    pass


class MaxPool_10(OpDesc, parser=CaffeCUDAParser, op_type="MaxPool", version=10):
    pass


class AveragePool(OpDesc, parser=CaffeCUDAParser):
    pass


class AveragePool_10(OpDesc, parser=CaffeCUDAParser, op_type="AveragePool", version=10):
    pass


class Conv(OpDesc, parser=CaffeCUDAParser):
    pass


class ConvTranspose(OpDesc, parser=CaffeCUDAParser):
    pass


class Transpose(OpDesc, parser=CaffeCUDAParser):
    pass


class BatchNorm(OpDesc, parser=CaffeCUDAParser):
    pass


class BatchNormalization(OpDesc, parser=CaffeCUDAParser):
    pass


class LpNormalization(OpDesc, parser=CaffeCUDAParser):
    pass


class InstanceNormalization(OpDesc, parser=CaffeCUDAParser):
    pass


class DeformableConvolution(OpDesc, parser=CaffeCUDAParser):
    pass


class Relu(OpDesc, parser=CaffeCUDAParser):
    pass


class PRelu(OpDesc, parser=CaffeCUDAParser):
    pass


class Gemm(OpDesc, parser=CaffeCUDAParser):
    pass


class Softmax(OpDesc, parser=CaffeCUDAParser, version=13):
    pass


class CaffeSoftmax(OpDesc, parser=CaffeCUDAParser):
    pass


class Concat(OpDesc, parser=CaffeCUDAParser):
    pass


class Upsample(OpDesc, parser=CaffeCUDAParser):
    pass


class DynamicUpsample(OpDesc, parser=CaffeCUDAParser):
    pass


class RoiAlign(OpDesc, parser=CaffeCUDAParser):
    pass


class PODRoiAlign(OpDesc, parser=CaffeCUDAParser):
    pass


class RoiPool(OpDesc, parser=CaffeCUDAParser):
    pass


class PSRoiPool(OpDesc, parser=CaffeCUDAParser):
    pass


class PSRoiMaskPool(OpDesc, parser=CaffeCUDAParser):
    pass


class Sigmoid(OpDesc, parser=CaffeCUDAParser):
    pass


class Slice_1(OpDesc, op_type="Slice", version=1, parser=CaffeCUDAParser):
    pass


# NOTE: Slice-10 is not supported in default.py
class Slice_10(OpDesc, op_type="Slice", version=10, parser=CaffeCUDAParser):
    pass


class Split(OpDesc, parser=CaffeCUDAParser):
    pass


class SubPixel(OpDesc, parser=CaffeCUDAParser):
    pass


class Eltwise(OpDesc, parser=CaffeCUDAParser):
    pass


class Corr(OpDesc, parser=CaffeCUDAParser):
    pass


class Reshape(OpDesc, parser=CaffeCUDAParser):
    pass


class GridSample(OpDesc, parser=CaffeCUDAParser):
    pass


class Unfold(OpDesc, parser=CaffeCUDAParser):
    pass


class LSTM(OpDesc, parser=CaffeCUDAParser):
    pass


class LeakyRelu(OpDesc, parser=CaffeCUDAParser):
    pass


class Cast(OpDesc, parser=CaffeCUDAParser):
    pass


class Clip(OpDesc, parser=CaffeCUDAParser):
    pass


class HeatMap2Coord(OpDesc, parser=CaffeCUDAParser):
    pass


class Erf(OpDesc, parser=CaffeCUDAParser):
    pass


class PSRoiMaskPool(OpDesc, parser=CaffeCUDAParser):
    pass


class Hswish(OpDesc, parser=CaffeCUDAParser):
    pass


class Pad(OpDesc, parser=CaffeCUDAParser):
    pass


class Elu(OpDesc, parser=CaffeCUDAParser):
    pass


class ScatterND(OpDesc, parser=CaffeCUDAParser):
    pass
