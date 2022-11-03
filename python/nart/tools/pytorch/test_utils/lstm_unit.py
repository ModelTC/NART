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


class LSTMUnitLinear(nn.Module):
    def __init__(self):
        super(LSTMUnitLinear, self).__init__()
        self.lstmcell = nn.LSTMCell(2, 3)
        self.linear = nn.Linear(3, 10)

    def forward(self, x, hx):
        h, c = self.lstmcell(x, hx)
        h = self.linear(h)
        return h, c


class LSTMUnitTestUnit(LayerTestUnit):
    def test_lstm_unit(self):
        name = "lstm_unit"
        model = nn.LSTMCell(2, 3)
        input_shapes = [(4, 2), ((4, 3), (4, 3))]
        input_names = ["x", "hx", "cx"]
        output_names = ["ht", "ct"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)

    def test_lstm_unit_linear(self):
        name = "lstm_unit_linear"
        model = LSTMUnitLinear()
        input_shapes = [(4, 2), ((4, 3), (4, 3))]
        input_names = ["x", "hx", "cx"]
        output_names = ["ht", "ct"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)
