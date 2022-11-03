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


class GroupNormForward(Function):
    @staticmethod
    def forward(ctx, input, num_groups, weight, bias, eps):
        return torch.group_norm(input, num_groups, weight, bias, eps)

    @staticmethod
    def symbolic(g, input, num_groups, weight, bias, eps):
        if weight is not None and bias is not None:
            return g.op(
                "GroupNorm", input, weight, bias, num_groups_i=num_groups, eps_f=eps
            )
        else:
            return g.op("GroupNorm", input, num_groups_i=num_groups, eps_f=eps)


def forward(input, num_groups, weight=None, bias=None, eps=1e-5):
    return GroupNormForward.apply(input, num_groups, weight, bias, eps)
