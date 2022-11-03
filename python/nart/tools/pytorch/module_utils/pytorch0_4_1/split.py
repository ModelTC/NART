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
    def split(input, start, end, dim):
        idx = [slice(None)] * len(input.shape)
        idx[dim] = slice(start, end)
        return input[idx]

    @staticmethod
    def forward(ctx, input, split_size, dim):
        chunks = []
        if isinstance(split_size, int):
            start = 0
            end = split_size
            while end <= input.shape[dim]:
                chunks.append(SplitForward.split(input, start, end, dim))
                start += split_size
                end += split_size
            return tuple(chunks)
        elif isinstance(split_size, (tuple, list)) and all(
            map(lambda x: isinstance(x, int), split_size)
        ):
            start = 0
            end = 0
            for size in split_size:
                end += size
                chunks.append(SplitForward.split(input, start, end, dim))
                start += size
            return tuple(chunks)
        else:
            raise NotImplementedError(
                "Split: split_size {}, dim {}".format(split_size, dim)
            )

    @staticmethod
    def symbolic(g, input, split_size, dim):
        starts = []
        ends = []
        if isinstance(split_size, int):
            start = 0
            end = split_size
            while end <= input.type().sizes()[dim]:
                starts.append(start)
                ends.append(end)
                start += split_size
                end += split_size
        elif isinstance(split_size, (tuple, list)) and all(
            map(lambda x: isinstance(x, int), split_size)
        ):
            start = 0
            end = 0
            for size in split_size:
                end += size
                starts.append(start)
                ends.append(end)
                start += size
        else:
            raise NotImplementedError(
                "Split: split_size {}, dim {}".format(split_size, dim)
            )
        return g.op(
            "Slice",
            input,
            starts_i=starts,
            ends_i=ends,
            axes_i=[dim],
            outputs=len(starts),
        )


def forward(input, split_size_or_sections, dim=0):
    return SplitForward.apply(input, split_size_or_sections, dim)
