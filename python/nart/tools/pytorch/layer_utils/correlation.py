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


class Correlation(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        # in caffe, bottom[0] is kernel, bottom[1] is input
        self.params.bottom.extend(self.node.input[::-1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.correlation_param.groups = attributes["groups"].i

    def set_blobshape(self):
        self.kernel_size = self.network.blobshape[self.params.bottom[0]][2]
        groups = self.params.correlation_param.groups
        kernel_c = self.network.blobshape[self.params.bottom[0]][1]
        n, c, h, w = self.network.blobshape[self.params.bottom[0]]
        h = h - self.kernel_size + 1
        w = h - self.kernel_size + 1
        if groups == 1:
            self.network.blobshape[self.params.top[0]] = [n, kernel_c // c, h, w]
        else:
            self.network.blobshape[self.params.top[0]] = [n, c, h, w]
