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


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, x):
        return x[:, :50], x[:, 50:]


class Split(nn.Module):
    def __init__(self, split_size, axis):
        super(Split, self).__init__()
        self.split_size = split_size
        self.axis = axis

    def forward(self, x):
        return torch.split(x, self.split_size, self.axis)


class SliceTestUnit(LayerTestUnit):
    def test_slice(self):
        model = Slice().eval()
        name = "slice"
        input_shapes = [(150,)]
        input_names = ["data"]
        output_names = ["output0", "output1"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_split(self):
        model = Split(50, 1).eval()
        name = "split"
        input_shapes = [(150,)]
        input_names = ["data"]
        output_names = ["output0", "output1", "output2"]
        self.compare(model, input_shapes, name, input_names, output_names)
