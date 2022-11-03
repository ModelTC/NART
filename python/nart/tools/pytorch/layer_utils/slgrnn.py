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


class SLGRNN(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output[0:1])

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:1])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.recurrent_param.num_output = attributes["num_output"].i
        self.params.recurrent_param.hidden_bias = True

        weight_ih = self.network.blobs[self.node.input[-4]]
        weight_hh = self.network.blobs[self.node.input[-3]]
        bias_ih = self.network.blobs[self.node.input[-2]]
        bias_hh = self.network.blobs[self.node.input[-1]]

        # caffe gate order: Z R G
        # ONNX gate order: R Z G
        shape = weight_ih.shape[0] // 3
        weight_ih = np.concatenate(
            (
                weight_ih[shape : 2 * shape, :],
                weight_ih[:shape, :],
                weight_ih[2 * shape : 3 * shape, :],
            ),
            axis=0,
        )
        weight_hh = np.concatenate(
            (
                weight_hh[shape : 2 * shape, :],
                weight_hh[:shape, :],
                weight_hh[2 * shape : 3 * shape, :],
            ),
            axis=0,
        )
        bias_ih = np.concatenate(
            (
                bias_ih[shape : 2 * shape],
                bias_ih[:shape],
                bias_ih[2 * shape : 3 * shape],
            ),
            axis=0,
        )
        bias_hh = np.concatenate(
            (
                bias_hh[shape : 2 * shape],
                bias_hh[:shape],
                bias_hh[2 * shape : 3 * shape],
            ),
            axis=0,
        )

        bias = np.concatenate((bias_ih, bias_hh), axis=0)
        self.params.blobs.extend([Layer.array_to_blobproto(weight_hh)])
        self.params.blobs.extend([Layer.array_to_blobproto(bias)])
        self.params.blobs.extend([Layer.array_to_blobproto(weight_ih)])

    def set_blobshape(self):
        seq_len, batch = self.network.blobshape[self.node.input[0]][:-1]
        self.network.blobshape[self.params.top[0]] = [
            seq_len,
            batch,
            self.params.recurrent_param.num_output,
        ]
