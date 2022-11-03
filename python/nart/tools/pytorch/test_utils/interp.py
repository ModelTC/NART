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
import torch.nn.functional as F
from .layer import LayerTestUnit


class Interp(nn.Module):
    def __init__(self):
        super(Interp, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, y):
        return self.upsample(x) + F.relu(y)


class InterpTestUnit(LayerTestUnit):
    def test_interp(self):
        model = Interp()
        name = "interp"
        input_shapes = [(1, 14, 14), (1, 28, 28)]
        input_names = ["data", "bias"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
