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
import math


class HoleConvolution(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        assert (
            attributes["dilations"].ints[0] != 1 or attributes["dilations"].ints[1] != 1
        )
        self.params.convolution_param.hole_h = attributes["dilations"].ints[0]
        self.params.convolution_param.hole_w = attributes["dilations"].ints[1]

        self.params.convolution_param.group = attributes["group"].i

        self.params.convolution_param.kernel_h = attributes["kernel_shape"].ints[0]
        self.params.convolution_param.kernel_w = attributes["kernel_shape"].ints[1]

        self.params.convolution_param.pad_h = attributes["pads"].ints[0]
        self.params.convolution_param.pad_w = attributes["pads"].ints[1]

        self.params.convolution_param.stride_h = attributes["strides"].ints[0]
        self.params.convolution_param.stride_w = attributes["strides"].ints[1]

        weight_blob = self.network.blobs[self.node.input[1]]
        self.params.blobs.extend([Layer.array_to_blobproto(weight_blob)])
        self.params.convolution_param.num_output = weight_blob.shape[0]

        if len(self.node.input) == 3:
            bias_blob = self.network.blobs[self.node.input[2]]
            self.params.blobs.extend([Layer.array_to_blobproto(bias_blob)])
            self.params.convolution_param.bias_term = True
        else:
            self.params.convolution_param.bias_term = False

    def set_blobshape(self):
        n, c, h, w = self.network.blobshape[self.params.bottom[0]]
        h = math.floor(
            (
                h
                + self.params.convolution_param.pad_h * 2
                - self.params.convolution_param.hole_h
                * (self.params.convolution_param.kernel_h - 1)
                - 1
            )
            / self.params.convolution_param.stride_h
            + 1
        )
        w = math.floor(
            (
                w
                + self.params.convolution_param.pad_w * 2
                - self.params.convolution_param.hole_w
                * (self.params.convolution_param.kernel_w - 1)
                - 1
            )
            / self.params.convolution_param.stride_w
            + 1
        )
        c = self.params.convolution_param.num_output
        self.network.blobshape[self.params.top[0]] = [n, c, h, w]
