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
import math
from torch.autograd import Function
from torch.nn import Parameter


class LSTMCellForward(Function):
    @staticmethod
    def forward(ctx, input, h, c, weight_ih, weight_hh, bias_ih, bias_hh):
        return torch.nn.backends.thnn.backend._LSTMCell(
            input, (h, c), weight_ih, weight_hh, bias_ih, bias_hh
        )

    @staticmethod
    def symbolic(g, input, h, c, weight_ih, weight_hh, bias_ih, bias_hh):
        return g.op(
            "LSTMCell",
            input,
            h,
            c,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            outputs=2,
            num_output_i=h.type().sizes()[1],
        )


def lstm_cell(input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
    return LSTMCellForward.apply(
        input, hx[0], hx[1], weight_ih, weight_hh, bias_ih, bias_hh
    )
