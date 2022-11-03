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
from torch.autograd import Variable
from functools import reduce


def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    if training:
        size = list(input.size())
        if reduce(mul, size[2:], size[0]) == 1:
            raise ValueError(
                "Expected more than 1 value per channel when training, got input size {}".format(
                    size
                )
            )
    if weight is None and bias is None:
        num_features = len(running_mean)
        weight = Variable(torch.ones(num_features), requires_grad=False)
        bias = Variable(torch.zeros(num_features), requires_grad=False)
    f = torch._C._functions.BatchNorm(
        running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled
    )
    return f(input, weight, bias)
