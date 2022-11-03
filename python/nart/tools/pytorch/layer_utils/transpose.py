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


class Transpose(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.transpose_param.dim.extend(attributes["perm"].ints)

    def set_blobshape(self):
        bottom_shape = self.network.blobshape[self.params.bottom[0]]
        top_shape = bottom_shape[:]
        for i, j in enumerate(self.params.transpose_param.dim):
            top_shape[i] = bottom_shape[j]
        self.network.blobshape[self.params.top[0]] = top_shape
