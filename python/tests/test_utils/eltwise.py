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


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


class Mul(nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y


class Max(nn.Module):
    def __init__(self):
        super(Max, self).__init__()

    def forward(self, x, y):
        return torch.max(x, y)


class MulAdd(nn.Module):
    def __init__(self):
        super(MulAdd, self).__init__()

    def forward(self, x, y, z):
        return 3 * x + 2.5 * y + z


class EltwiseTestUnit(LayerTestUnit):
    def test_add(self):
        model = Add().eval()
        name = "add"
        input_shapes = [(3, 112, 112), (3, 112, 112)]
        input_names = ["data0", "data1"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_mul(self):
        model = Mul().eval()
        name = "mul"
        input_shapes = [(3, 112, 112), (3, 112, 112)]
        input_names = ["data0", "data1"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_max(self):
        model = Max().eval()
        name = "max"
        input_shapes = [(3, 112, 112), (3, 112, 112)]
        input_names = ["data0", "data1"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_muladd(self):
        model = MulAdd().eval()
        name = "muladd"
        input_shapes = [(3, 112, 112), (3, 112, 112), (3, 112, 112)]
        input_names = ["data0", "data1", "data2"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
