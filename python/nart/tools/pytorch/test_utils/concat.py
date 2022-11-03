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


class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, x, y):
        return torch.cat([x, y], 1)


class ConcatTestUnit(LayerTestUnit):
    def test_interp(self):
        model = Concat(axis=1)
        name = "concat"
        input_shapes = [(1, 14), (1, 14)]
        input_names = ["data0", "data1"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names)
