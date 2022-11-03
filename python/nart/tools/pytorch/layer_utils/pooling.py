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
import math


class Pooling(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        if self.node.op_type == "MaxPool" or self.node.op_type == "GlobalMaxPool":
            self.params.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
        elif (
            self.node.op_type == "AveragePool"
            or self.node.op_type == "GlobalAveragePool"
        ):
            self.params.pooling_param.pool = caffe_pb2.PoolingParameter.AVE

        # global pool
        if (
            self.node.op_type == "GlobalAveragePool"
            or self.node.op_type == "GlobalMaxPool"
        ):
            self.params.pooling_param.global_pooling = True
            return

        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        if "ceil_mode" not in attributes or attributes["ceil_mode"].i != 1:
            self.params.pooling_param.ceil_mode = False

        [
            self.params.pooling_param.kernel_h,
            self.params.pooling_param.kernel_w,
        ] = Layer.get_list(2, attributes["kernel_shape"].ints)

        [
            self.params.pooling_param.pad_h,
            self.params.pooling_param.pad_w,
        ] = Layer.get_list(2, attributes["pads"].ints)

        [
            self.params.pooling_param.stride_h,
            self.params.pooling_param.stride_w,
        ] = Layer.get_list(2, attributes["strides"].ints)

    def set_blobshape(self):
        n, c, h, w = self.network.blobshape[self.params.bottom[0]]

        # global pool
        if (
            self.node.op_type == "GlobalAveragePool"
            or self.node.op_type == "GlobalMaxPool"
        ):
            h, w = (1, 1)
            self.network.blobshape[self.params.top[0]] = [n, c, h, w]
            return

        if self.params.pooling_param.ceil_mode:
            h = math.ceil(
                (
                    h
                    + 2 * self.params.pooling_param.pad_h
                    - self.params.pooling_param.kernel_h
                )
                / self.params.pooling_param.stride_h
                + 1
            )
            w = math.ceil(
                (
                    w
                    + 2 * self.params.pooling_param.pad_w
                    - self.params.pooling_param.kernel_w
                )
                / self.params.pooling_param.stride_w
                + 1
            )
        else:
            h = math.floor(
                (
                    h
                    + 2 * self.params.pooling_param.pad_h
                    - self.params.pooling_param.kernel_h
                )
                / self.params.pooling_param.stride_h
                + 1
            )
            w = math.floor(
                (
                    w
                    + 2 * self.params.pooling_param.pad_w
                    - self.params.pooling_param.kernel_w
                )
                / self.params.pooling_param.stride_w
                + 1
            )
        self.network.blobshape[self.params.top[0]] = [n, c, h, w]
