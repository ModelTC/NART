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


class NNInterp(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        if len(self.node.input) == 2:
            scales = self.network.blobs[self.node.input[1]]
            scales = list(scales)
            input_shape = self.network.blobshape[self.node.input[0]]
            import math

            height = math.floor(input_shape[2] * scales[2])
            width = math.floor(input_shape[3] * scales[3])
        elif "height" in attributes:
            height = attributes["height"].i
            width = attributes["width"].i
        else:
            print(f"Unsupported `Upsample` {self.node}")
        self.params.nninterp_param.height = height
        self.params.nninterp_param.width = width

    def set_blobshape(self):
        n, c = self.network.blobshape[self.params.bottom[0]][:2]
        h = self.params.nninterp_param.height
        w = self.params.nninterp_param.width
        self.network.blobshape[self.params.top[0]] = [n, c, h, w]
