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


class DeformableConvolution(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:2])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )

        self.params.deformable_convolution_param.hole_h = attributes["dilations"].ints[
            0
        ]
        self.params.deformable_convolution_param.hole_w = attributes["dilations"].ints[
            1
        ]

        self.params.deformable_convolution_param.group = attributes["groups"].i

        self.params.deformable_convolution_param.deformable_group = attributes[
            "deformable_groups"
        ].i

        self.params.deformable_convolution_param.kernel_h = attributes[
            "kernel_size"
        ].ints[0]
        self.params.deformable_convolution_param.kernel_w = attributes[
            "kernel_size"
        ].ints[1]

        self.params.deformable_convolution_param.pad_h = attributes["pad"].ints[0]
        self.params.deformable_convolution_param.pad_w = attributes["pad"].ints[1]

        self.params.deformable_convolution_param.stride_h = attributes["stride"].ints[0]
        self.params.deformable_convolution_param.stride_w = attributes["stride"].ints[1]

        self.params.deformable_convolution_param.num_output = attributes[
            "out_channels"
        ].i

        weight = self.network.blobs[self.node.input[2]]
        self.params.blobs.extend([Layer.array_to_blobproto(weight)])
        self.params.deformable_convolution_param.bias_term = False

    def set_blobshape(self):
        n, c, h, w = self.network.blobshape[self.params.bottom[0]]
        h = math.floor(
            (
                h
                + self.params.deformable_convolution_param.pad_h * 2
                - self.params.deformable_convolution_param.hole_h
                * (self.params.deformable_convolution_param.kernel_h - 1)
                - 1
            )
            / self.params.deformable_convolution_param.stride_h
            + 1
        )
        w = math.floor(
            (
                w
                + self.params.deformable_convolution_param.pad_w * 2
                - self.params.deformable_convolution_param.hole_w
                * (self.params.deformable_convolution_param.kernel_w - 1)
                - 1
            )
            / self.params.deformable_convolution_param.stride_w
            + 1
        )
        c = self.params.deformable_convolution_param.num_output
        self.network.blobshape[self.params.top[0]] = [n, c, h, w]
