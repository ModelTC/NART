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


class Reshape(nn.Module):
    def __init__(self, width, height):
        super(Reshape, self).__init__()
        self.width = width
        self.height = height

    def forward(self, x):
        return x.view(-1, self.width, self.height)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(0)


class Unsqueeze(nn.Module):
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(0)


class ReshapeTestUnit(LayerTestUnit):
    def test_reshape(self):
        model = Reshape(112, 112).eval()
        name = "reshape"
        input_shapes = [(224, 56)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_flatten(self):
        model = Flatten().eval()
        name = "flatten"
        input_shapes = [(128, 7, 7)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_squeeze(self):
        model = Squeeze().eval()
        name = "squeeze"
        input_shapes = [(1, 7, 7)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_unsqueeze(self):
        model = Unsqueeze().eval()
        name = "unsqueeze"
        input_shapes = [(28, 7, 7)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
