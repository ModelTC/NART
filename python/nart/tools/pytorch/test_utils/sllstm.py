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


class LSTMLinear(nn.Module):
    def __init__(self, batch_first, bidirectional):
        super(LSTMLinear, self).__init__()
        self.lstm = nn.LSTM(
            3, 4, 2, batch_first=batch_first, bidirectional=bidirectional
        )
        self.linear = nn.Linear(8 if bidirectional else 4, 10)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = self.linear(x.view(-1, x.size(2)))
        return x


class SLLSTMTestUnit(LayerTestUnit):
    def test_sllstm(self):
        name = "sllstm"
        model = nn.LSTM(3, 4, 2)
        input_shapes = [(5, 1, 3)]
        input_names = ["x"]
        output_names = ["o", "h", "c"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)

    def test_lstm_linear(self):
        name = "lstm_linear"
        model = LSTMLinear(True, True)
        input_shapes = [(1, 5, 3)]
        input_names = ["x"]
        output_names = ["o"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)

    def test_lstm_linear_batch_first(self):
        name = "lstm_linear"
        model = LSTMLinear(True, False)
        input_shapes = [(1, 5, 3)]
        input_names = ["x"]
        output_names = ["o"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)

    def test_lstm_linear_bidirectional(self):
        name = "lstm_linear"
        model = LSTMLinear(False, True)
        input_shapes = [(5, 1, 3)]
        input_names = ["x"]
        output_names = ["o"]
        self.compare(model, input_shapes, name, input_names, output_names, cloze=False)
