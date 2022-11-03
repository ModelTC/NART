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

import torch
import torch.nn as nn
from .layer import LayerTestUnit


class GroupNormTestUnit(LayerTestUnit):
    def test_groupnorm_affine(self):
        if torch.__version__[:3] == "0.3":
            print("torch 0.3.x does not suppoert GroupNorm - Skipping this test.")
            return
        model = nn.GroupNorm(num_groups=2, num_channels=6).eval()
        model.weight.data.normal_(0, 1)
        model.bias.data.normal_(0, 1)
        name = "groupnorm_affine"
        input_shapes = [(6, 224, 224)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_groupnorm_not_affine(self):
        if torch.__version__[:3] == "0.3":
            print("torch 0.3.x does not suppoert GroupNorm - Skipping this test.")
            return
        model = nn.GroupNorm(num_groups=2, num_channels=6, affine=False).eval()
        name = "groupnorm_not_affine"
        input_shapes = [(6, 224, 224)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
