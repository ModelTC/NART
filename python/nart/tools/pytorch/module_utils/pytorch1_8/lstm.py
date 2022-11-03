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


class LSTMForwardo(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        h,
        c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bias,
        hidden_size,
        dropout,
        training,
        bidirectional,
        batch_first,
    ):
        max_batch_size = input.size(0) if batch_first else input.size(1)
        num_directions = 2 if bidirectional else 1
        num_layers = 1
        if h is None or c is None:
            h = input.new_zeros(
                num_directions, max_batch_size, hidden_size, requires_grad=False
            )
            c = h
        return torch._C._VariableFunctions.lstm(
            input,
            (h, c),
            (weight_ih, weight_hh, bias_ih, bias_hh),
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
            batch_first,
        )[0]

    @staticmethod
    def symbolic(
        g,
        input,
        h,
        c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bias,
        hidden_size,
        dropout,
        training,
        bidirectional,
        batch_first,
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
                outputs=1,
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
                outputs=1,
                num_output_i=hidden_size,
            )


class LSTMForwardh(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        h,
        c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bias,
        hidden_size,
        dropout,
        training,
        bidirectional,
        batch_first,
    ):
        max_batch_size = input.size(0) if batch_first else input.size(1)
        num_directions = 2 if bidirectional else 1
        num_layers = 1
        if h is None or c is None:
            h = input.new_zeros(
                num_directions, max_batch_size, hidden_size, requires_grad=False
            )
            c = h
        return torch._C._VariableFunctions.lstm(
            input,
            (h, c),
            (weight_ih, weight_hh, bias_ih, bias_hh),
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
            batch_first,
        )[1]

    @staticmethod
    def symbolic(
        g,
        input,
        h,
        c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bias,
        hidden_size,
        dropout,
        training,
        bidirectional,
        batch_first,
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
                outputs=1,
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
                outputs=1,
                num_output_i=hidden_size,
            )


class LSTMForwardc(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        h,
        c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bias,
        hidden_size,
        dropout,
        training,
        bidirectional,
        batch_first,
    ):
        max_batch_size = input.size(0) if batch_first else input.size(1)
        num_directions = 2 if bidirectional else 1
        num_layers = 1
        if h is None or c is None:
            h = input.new_zeros(
                num_directions, max_batch_size, hidden_size, requires_grad=False
            )
            c = h
        return torch._C._VariableFunctions.lstm(
            input,
            (h, c),
            (weight_ih, weight_hh, bias_ih, bias_hh),
            bias,
            num_layers,
            dropout,
            training,
            bidirectional,
            batch_first,
        )[2]

    @staticmethod
    def symbolic(
        g,
        input,
        h,
        c,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        bias,
        hidden_size,
        dropout,
        training,
        bidirectional,
        batch_first,
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
                outputs=1,
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
                outputs=1,
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

    all_hidden = []
    all_cell = []
    for i in range(self.num_layers):
        weight_ih = getattr(self, "weight_ih_l{}".format(i))
        weight_hh = getattr(self, "weight_hh_l{}".format(i))
        bias_ih = getattr(self, "bias_ih_l{}".format(i))
        bias_hh = getattr(self, "bias_hh_l{}".format(i))
        o = LSTMForwardo.apply(
            input,
            h_prev,
            c_prev,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            self.bias,
            self.hidden_size,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        h = LSTMForwardh.apply(
            input,
            h_prev,
            c_prev,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            self.bias,
            self.hidden_size,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        c = LSTMForwardc.apply(
            input,
            h_prev,
            c_prev,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            self.bias,
            self.hidden_size,
            self.dropout,
            self.training,
            self.bidirectional,
            self.batch_first,
        )
        all_hidden.append(h)
        all_cell.append(c)
        input = o

        if bidirectional:
            weight_ih_reverse = getattr(self, "weight_ih_l{}_reverse".format(i))
            weight_hh_reverse = getattr(self, "weight_hh_l{}_reverse".format(i))
            bias_ih_reverse = getattr(self, "bias_ih_l{}_reverse".format(i))
            bias_hh_reverse = getattr(self, "bias_hh_l{}_reverse".format(i))
            o_reverse = LSTMForwardo.apply(
                input_reverse,
                h_reverse_prev,
                c_reverse_prev,
                weight_ih_reverse,
                weight_hh_reverse,
                bias_ih_reverse,
                bias_hh_reverse,
                self.bias,
                self.hidden_size,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            ).flip([0])
            h_reverse = LSTMForwardh.apply(
                input_reverse,
                h_reverse_prev,
                c_reverse_prev,
                weight_ih_reverse,
                weight_hh_reverse,
                bias_ih_reverse,
                bias_hh_reverse,
                self.bias,
                self.hidden_size,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            ).flip([0])
            c_reverse = LSTMForwardc.apply(
                input_reverse,
                h_reverse_prev,
                c_reverse_prev,
                weight_ih_reverse,
                weight_hh_reverse,
                bias_ih_reverse,
                bias_hh_reverse,
                self.bias,
                self.hidden_size,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            ).flip([0])
            all_hidden.append(h_reverse)
            all_cell.append(c_reverse)
            o = torch.cat((o, o_reverse), dim=2)
            input = o
            input_reverse = input.flip([0])

    if batch_first:
        o = o.permute((1, 0, 2))
    if bidirectional:
        h = torch.cat(all_hidden, dim=0)
        c = torch.cat(all_cell, dim=0)
    return o, (h, c)
