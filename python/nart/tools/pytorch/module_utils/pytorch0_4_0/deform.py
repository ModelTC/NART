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

import torch.nn._functions as _functions
from torch.autograd import Function
from torch._thnn import type2backend
from torch.nn.modules.utils import _single, _pair, _triple
from pod import extensions
import torch


class DeformableConvFunction(Function):
    @staticmethod
    def forward(
        self,
        input,
        offset,
        weight,
        out_channels,
        kernel_size,
        stride,
        pad,
        dilations,
        groups,
        deformable_groups,
    ):
        if input.dim() == 4:
            return torch.zeros(
                [input.size(0), input.size(1), input.size(2), input.size(3)]
            )
        else:
            raise NotImplementedError(
                "Input Error: Only 4D input Tensors supported"
                " (got {})".format(input.dim())
            )

    @staticmethod
    def symbolic(
        g,
        input,
        offset,
        weight,
        out_channels,
        kernel_size,
        stride,
        pad,
        dilations,
        groups,
        deformable_groups,
    ):
        if True:  # input.dim() == 4:
            height = input.type().sizes()[2]
            width = input.type().sizes()[3]
        else:
            raise NotImplementedError("Input Error: Only 4D input Tensors Supported")
        return g.op(
            "DeformConv",
            input,
            offset,
            weight,
            out_channels_i=out_channels,
            kernel_size_i=kernel_size,
            stride_i=stride,
            pad_i=pad,
            dilations_i=dilations,
            groups_i=groups,
            deformable_groups_i=deformable_groups,
        )


def forward(self, input, offset):
    return DeformableConvFunction.apply(
        input,
        offset,
        self.weight,
        self.out_channels,
        self.kernel_size,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
        self.num_deformable_groups,
    )
