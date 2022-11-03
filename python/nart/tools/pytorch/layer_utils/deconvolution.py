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


class Deconvolution(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        if attributes["dilations"].ints[0] != 1 or attributes["dilations"].ints[1] != 1:
            raise NotImplementedError(
                "Deconvolution with dilation not supported in caffe."
            )

        self.params.convolution_param.group = attributes["group"].i

        [
            self.params.convolution_param.kernel_h,
            self.params.convolution_param.kernel_w,
        ] = Layer.get_list(2, attributes["kernel_shape"].ints)

        [
            self.params.convolution_param.pad_h,
            self.params.convolution_param.pad_w,
        ] = Layer.get_list(2, attributes["pads"].ints)

        [
            self.params.convolution_param.stride_h,
            self.params.convolution_param.stride_w,
        ] = Layer.get_list(2, attributes["strides"].ints)

        weight_blob = self.network.blobs[self.node.input[1]]
        self.params.blobs.extend([Layer.array_to_blobproto(weight_blob)])
        self.params.convolution_param.num_output = (
            weight_blob.shape[1] * attributes["group"].i
        )

        if len(self.node.input) == 3 and self.node.input[2] in self.network.blobs:
            bias_blob = self.network.blobs[self.node.input[2]]
            self.params.blobs.extend([Layer.array_to_blobproto(bias_blob)])
            self.params.convolution_param.bias_term = True
        else:
            self.params.convolution_param.bias_term = False

    def set_blobshape(self):
        n, c, h, w = self.network.blobshape[self.params.bottom[0]]
        h = (
            self.params.convolution_param.stride_h * (h - 1)
            + self.params.convolution_param.kernel_h
            - 2 * self.params.convolution_param.pad_h
        )
        w = (
            self.params.convolution_param.stride_w * (w - 1)
            + self.params.convolution_param.kernel_w
            - 2 * self.params.convolution_param.pad_w
        )
        c = self.params.convolution_param.num_output
        self.network.blobshape[self.params.top[0]] = [n, c, h, w]
