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


class Scale(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        if self.node.input[1] in self.network.blobs:
            self.params.bottom.extend(self.node.input[0:1])
        else:
            self.params.bottom.extend(self.node.input[0:2])

    def set_param(self):
        if self.node.input[1] in self.network.blobs:
            self.params.scale_param.bias_term = True
            scale_blob = self.network.blobs[self.node.input[1]]
            self.params.blobs.extend([Layer.array_to_blobproto(scale_blob)])
            bias_blob = self.network.blobs[self.node.input[2]]
            self.params.blobs.extend([Layer.array_to_blobproto(bias_blob)])
        else:
            self.params.scale_param.bias_term = False
            self.params.scale_param.axis = 0

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]
