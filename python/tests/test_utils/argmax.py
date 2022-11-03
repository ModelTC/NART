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


class ArgMax(nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.argmax(self.dim, self.keepdim)


class ArgMaxTestUnit(LayerTestUnit):
    def test_argmax(self):
        model = ArgMax()
        name = "argmax"
        input_shapes = [(1, 14, 14)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_argmax_with_axis(self):
        model = ArgMax(dim=2)
        name = "argmax"
        input_shapes = [(1, 14, 14)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_argmax_with_axis_keepdim(self):
        model = ArgMax(dim=2, keepdim=True)
        name = "argmax"
        input_shapes = [(1, 14, 14)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)

    def test_argmax_with_neg_axis(self):
        model = ArgMax(dim=-2)
        name = "argmax"
        input_shapes = [(1, 14, 14)]
        input_names = ["data"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
