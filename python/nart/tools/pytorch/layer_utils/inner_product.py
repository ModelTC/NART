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


class InnerProduct(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input[0:1])

    def set_param(self):
        weight_blob = self.network.blobs[self.node.input[1]]
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        assert "transA" not in attributes or attributes["transA"].i == 0
        if "transB" not in attributes or attributes["transB"].i == 0:
            # if transB==0, B will not be transposed before matrix multiply, this requires B to be transposed in caffe InnerProduct layer.
            import numpy as np

            weight_blob = np.transpose(weight_blob, [1, 0])
        self.params.blobs.extend([Layer.array_to_blobproto(weight_blob)])
        self.params.inner_product_param.num_output = weight_blob.shape[0]

        if (
            len(self.node.input) == 3
            and self.node.input[2] in self.network.blobs.keys()
        ):
            bias_blob = self.network.blobs[self.node.input[2]]
            self.params.blobs.extend([Layer.array_to_blobproto(bias_blob)])
            self.params.inner_product_param.bias_term = True
        else:
            self.params.inner_product_param.bias_term = False

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = [
            self.network.blobshape[self.params.bottom[0]][0],
            self.params.inner_product_param.num_output,
        ]
