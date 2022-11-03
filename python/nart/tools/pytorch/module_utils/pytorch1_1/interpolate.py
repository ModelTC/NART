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

import torch.nn.functional as F
from torch.autograd import Function
import warnings


class InterpolateForward(Function):
    @staticmethod
    def forward(ctx, input, size, scale_factor, mode, align_corners):
        return F.ori_interpolate(input, size, scale_factor, mode, align_corners)

    @staticmethod
    def symbolic(g, input, size, scale_factor, mode, align_corners):
        if scale_factor is not None:
            height = int(input.type().sizes()[2] * scale_factor)
            width = int(input.type().sizes()[3] * scale_factor)
        elif len(size) == 2:
            height = size[0]
            width = size[1]
        elif len(size == 1):
            height = size[0]
            width = size[0]
        return g.op("Upsample", input, mode_s=mode, height_i=height, width_i=width)


def forward(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    if mode == "bilinear" and (align_corners is None or align_corners == False):
        message = "[PRECISION WARNING] CaffeInfer supports bilinear upsampling with align_corners == True.\n"
        message += " Setting it False/None will cause performance loss."
        warnings.warn(message)
    return InterpolateForward.apply(input, size, scale_factor, mode, align_corners)
