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
import numpy as np


class BN(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        # self.params.batch_norm_param.use_global_stats = attributes['is_test'].i
        self.params.bn_param.moving_average = True
        self.params.bn_param.var_eps = attributes["epsilon"].f

        scale_blob = self.network.blobs[self.node.input[1]]
        self.params.blobs.extend([Layer.array_to_blobproto(scale_blob)])
        bias_blob = self.network.blobs[self.node.input[2]]
        self.params.blobs.extend([Layer.array_to_blobproto(bias_blob)])

        running_mean = self.network.blobs[self.node.input[3]]
        self.params.blobs.extend([Layer.array_to_blobproto(running_mean)])

        running_var = self.network.blobs[self.node.input[4]]
        self.params.blobs.extend([Layer.array_to_blobproto(running_var)])
        channel = len(self.params.blobs[3].data)
        for i in range(4):
            self.params.blobs[i].shape.ClearField("dim")
            self.params.blobs[i].shape.dim.extend([1, channel, 1, 1])

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]
