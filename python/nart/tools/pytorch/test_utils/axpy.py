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


class Axpy(nn.Module):
    def __init__(self):
        super(Axpy, self).__init__()

    def forward(self, a, x, y):
        return a * x + y


class AxpyTestUnit(LayerTestUnit):
    def test_axpy(self):
        model = Axpy()
        name = "Axpy"
        input_shapes = [(1, 3, 1, 1), (1, 3, 16, 16), (1, 3, 16, 16)]
        input_names = ["a", "x", "y"]
        output_names = ["output"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)
