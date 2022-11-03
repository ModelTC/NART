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
from torch.autograd import Function


class FlipForward(Function):
    @staticmethod
    def forward(ctx, input, dims):
        assert len(dims) == 1
        dim = dims[0]
        size = input.size()
        dim = input.dim() + dim if dim < 0 else dim
        input = input.view(-1, *size[dim:])
        input = input.view(input.size(0), input.size(1), -1)[
            :, torch.arange(input.size(1) - 1, -1, -1).long(), :
        ]
        return input.view(size)

    @staticmethod
    def symbolic(g, input, dims):
        return g.op("Flip", input, dims_i=dims)


def forward(input, dims):
    return FlipForward.apply(input, dims)
