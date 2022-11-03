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

from .layer import Layer
from ...proto import caffe_pb2
import numpy as np


class Eltwise(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        if self.node.op_type == "Mul":
            self.params.eltwise_param.operation = caffe_pb2.EltwiseParameter.PROD
        if self.node.op_type == "Add":
            self.params.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
            attributes = dict(
                zip([attr.name for attr in self.node.attribute], self.node.attribute)
            )
            if "coeff" in attributes.keys():
                coeff = attributes["coeff"].floats
            else:
                coeff = [1.0] * len(self.params.bottom)
            self.params.eltwise_param.coeff.extend(coeff)
        if self.node.op_type == "Sub":
            self.params.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
            attributes = dict(
                zip([attr.name for attr in self.node.attribute], self.node.attribute)
            )
            self.params.eltwise_param.coeff.extend(
                [1, *[-1] * (len(self.node.input) - 1)]
            )
        if self.node.op_type == "Max":
            self.params.eltwise_param.operation = caffe_pb2.EltwiseParameter.MAX

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]
