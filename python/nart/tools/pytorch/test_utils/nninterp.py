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


class NNInterp(nn.Module):
    def __init__(self):
        super(NNInterp, self).__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(1, 3, 3, 1, 1)

    def forward(self, x, y):
        return torch.cat((self.upsample(x), self.conv(y)), dim=1)


class NNInterpTestUnit(LayerTestUnit):
    def test_nninterp(self):
        model = NNInterp()
        name = "nninterp"
        input_shapes = [(1, 14, 14), (1, 28, 28)]
        input_names = ["data", "other"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
