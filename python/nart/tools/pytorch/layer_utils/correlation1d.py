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


class Correlation1D(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        # in caffe, bottom[0] is kernel, bottom[1] is input
        self.params.bottom.extend(self.node.input[::-1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.correlation_param.max_displacement = attributes[
            "max_displacement"
        ].i
        self.params.correlation_param.pad = attributes["pad"].i
        self.params.correlation_param.single_direction = attributes[
            "single_direction"
        ].i
        self.params.correlation_param.kernel_size = attributes["kernel_size"].i

    def set_blobshape(self):
        n, c, h, w = self.network.blobshape[self.params.bottom[0]]
        c = self.params.correlation_param.max_displacement + 1
        self.network.blobshape[self.params.top[0]] = [n, c, h, w]
