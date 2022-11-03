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
import warnings
from torch.autograd import Function
from torch.nn import Parameter


class LSTMForward(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        func,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        h,
        c,
        batch_first,
        bidirectional,
        hidden_size,
    ):
        if h is None or c is None:
            max_batch_size = input.size(0) if batch_first else input.size(1)
            num_directions = 2 if bidirectional else 1
            h = input.new_zeros(
                num_directions, max_batch_size, hidden_size, requires_grad=False
            )
            c = h
        all_weights = []
        all_weights.append([weight_ih, weight_hh, bias_ih, bias_hh])
        o, (h, c) = func(input, all_weights, (h, c), None)
        return o, h, c

    @staticmethod
    def symbolic(
        g,
        input,
        func,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        h,
        c,
        batch_first,
        bidirectional,
        hidden_size,
    ):
        if h is not None and c is not None:
            return g.op(
                "LSTM",
                input,
                h,
                c,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                outputs=3,
                num_output_i=hidden_size,
            )
        else:
            return g.op(
                "LSTM",
                input,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                outputs=3,
                num_output_i=hidden_size,
            )


def forward(self, input, hx=None):
    if hx is not None:
        warnings.warn(
            "[FATAL WARNING] LSTM should take no hidden state or cell state. "
            + "Please modify the input to be the data sequence only."
        )
        h_prev, c_prev = hx
    else:
        h_prev, c_prev = None, None

    batch_first = self.batch_first
    if self.batch_first:
        input = input.permute((1, 0, 2))
        self.batch_first = False

    bidirectional = self.bidirectional
    if self.bidirectional:
        input_reverse = torch.flip(input, dims=[0])
        h_reverse_prev = None
        c_reverse_prev = None
        self.bidirectional = False

    has_flat_weights = (
        list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
    )
    if has_flat_weights:
        first_data = next(self.parameters()).data
        assert first_data.storage().size() == self._param_buf_size
        flat_weight = first_data.new().set_(
            first_data.storage(), 0, torch.Size([self._param_buf_size])
        )
    else:
        flat_weight = None

    input_size = self.input_size
    hidden_size = self.hidden_size
    all_hidden = []
    all_cell = []
    for i in range(self.num_layers):
        func = self._backend.RNN(
            self.mode,
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            dropout_state=self.dropout_state,
            variable_length=False,
            flat_weight=flat_weight,
        )
        weight_ih = getattr(self, "weight_ih_l{}".format(i))
        weight_hh = getattr(self, "weight_hh_l{}".format(i))
        bias_ih = getattr(self, "bias_ih_l{}".format(i))
        bias_hh = getattr(self, "bias_hh_l{}".format(i))
        o, h, c = LSTMForward.apply(
            input,
            func,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h_prev,
            c_prev,
            self.batch_first,
            self.bidirectional,
            self.hidden_size,
        )
        all_hidden.append(h)
        all_cell.append(c)
        input = o
        input_size = hidden_size

        if bidirectional:
            weight_ih_reverse = getattr(self, "weight_ih_l{}_reverse".format(i))
            weight_hh_reverse = getattr(self, "weight_hh_l{}_reverse".format(i))
            bias_ih_reverse = getattr(self, "bias_ih_l{}_reverse".format(i))
            bias_hh_reverse = getattr(self, "bias_hh_l{}_reverse".format(i))
            o_reverse, h_reverse, c_reverse = LSTMForward.apply(
                input_reverse,
                func,
                weight_ih_reverse,
                weight_hh_reverse,
                bias_ih_reverse,
                bias_hh_reverse,
                h_reverse_prev,
                c_reverse_prev,
                self.batch_first,
                self.bidirectional,
                self.hidden_size,
            )
            o_reverse = o_reverse.flip([0])
            all_hidden.append(h_reverse)
            all_cell.append(c_reverse)
            o = torch.cat((o, o_reverse), dim=2)
            input = o
            input_reverse = input.flip([0])

    if batch_first:
        o = o.permute((1, 0, 2))
    if bidirectional:
        h = torch.cat(all_hidden, dim=0)
        h = torch.cat(all_cell, dim=0)
    return o, (h, c)
