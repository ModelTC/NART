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


class SplitForward(Function):
    @staticmethod
    def forward(ctx, input, start, end, dim):
        idx = [slice(None)] * len(input.shape)
        idx[dim] = slice(start, end)
        return input[idx]

    @staticmethod
    def symbolic(g, input, start, end, dim):
        return g.op("Slice", input, starts_i=[start], ends_i=[end], axes_i=[dim])


def forward(input, split_size, dim=0):
    chunks = []
    if isinstance(split_size, int):
        start = 0
        end = split_size
        while end <= input.shape[dim]:
            chunks.append(SplitForward.apply(input, start, end, dim))
            start += split_size
            end += split_size
        return chunks
    elif isinstance(split_size, (tuple, list)) and all(
        map(lambda x: isinstance(x, (int, torch.Tensor)), split_size)
    ):
        start = 0
        end = 0
        for size in split_size:
            if isinstance(size, torch.Tensor):
                size = size.item()
            end += size
            chunks.append(SplitForward.apply(input, start, end, dim))
            start += size
        return chunks
    else:
        raise NotImplementedError(
            "Split: split_size {}, dim {}".format(split_size, dim)
        )
