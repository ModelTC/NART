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

import torch.nn as nn
from .layer import LayerTestUnit


class BatchNormTestUnit(LayerTestUnit):
    def test_batchnorm_affine(self):
        model = nn.BatchNorm2d(num_features=16, eps=1e-6, momentum=1e-1).eval()
        model.weight.data.normal_(0, 1)
        model.bias.data.normal_(0, 1)
        name = "batchnorm_affine"
        input_shapes = [(16, 224, 224)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_batchnorm_not_affine(self):
        model = nn.BatchNorm2d(
            num_features=16, eps=1e-6, momentum=1e-1, affine=False
        ).eval()
        name = "batchnorm_not_affine"
        input_shapes = [(16, 224, 224)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
