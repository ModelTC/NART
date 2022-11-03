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
from numbers import Integral


class UpsampleForward(Function):
    @staticmethod
    def forward(ctx, input, size, scale_factor, mode):
        if input.dim() == 4 and mode == "nearest":
            return upsample_nearest_2d(ctx, input, _pair(size), scale_factor)
        elif input.dim() == 4 and mode == "bilinear":
            return upsample_bilinear_2d(ctx, input, _pair(size), scale_factor)
        else:
            raise NotImplementedError(
                "Input Error: Only 4D input Tensors supported"
                " (got {}D) for the modes: nearest | linear | bilinear | trilinear"
                " (got {})".format(input.dim(), mode)
            )

    @staticmethod
    def symbolic(g, input, size, scale_factor, mode):
        if scale_factor is not None:
            height = input.type().sizes()[2] * scale_factor
            width = input.type().sizes()[3] * scale_factor
        elif len(size) == 2:
            height = size[0]
            width = size[1]
        elif len(size == 1):
            height = size[0]
            width = size[0]
        return g.op("Upsample", input, mode_s=mode, height_i=height, width_i=width)


def forward(input, size=None, scale_factor=None, mode="bilinear"):
    return UpsampleForward.apply(input, size, scale_factor, mode)


def _check_size_scale_factor(size, scale_factor):
    if size is None and scale_factor is None:
        raise ValueError("either size or scale_factor should be defined")
    if scale_factor is not None and not isinstance(scale_factor, (Integral, tuple)):
        raise ValueError(
            "scale_factor must be of integer type or a tuple of integer types"
        )


def upsample_bilinear_2d(ctx, input, size, scale_factor):
    assert input.dim() == 4

    ctx.size = size
    ctx.scale_factor = scale_factor

    if ctx.scale_factor is not None:
        ctx.scale_factor = _check_linear_scale_factor(ctx.scale_factor, dim=2)

    if ctx.scale_factor is not None:
        ctx.output_size = (
            input.size(2) * ctx.scale_factor[0],
            input.size(3) * ctx.scale_factor[1],
        )
    else:
        ctx.output_size = ctx.size

    ctx.input_size = input.size()
    output = input.new()
    backend = type2backend[type(input)]
    backend.SpatialUpSamplingBilinear_updateOutput(
        backend.library_state,
        input,
        output,
        ctx.output_size[0],
        ctx.output_size[1],
    )
    return output


def _check_linear_scale_factor(scale_factor, dim=2):
    if dim == 1:
        scale_factor = _single(scale_factor)
    elif dim == 2:
        scale_factor = _pair(scale_factor)
    elif dim == 3:
        scale_factor = _triple(scale_factor)
    else:
        raise ValueError("dim has to be 1, 2 or 3")

    try:
        assert (
            len(scale_factor) == 1 or len(scale_factor) == 2 or len(scale_factor) == 3
        )
        assert all(isinstance(s, Integral) and s >= 1 for s in scale_factor)
    except AssertionError as e:
        raise ValueError(
            "scale_factor must be a non-negative integer, "
            "or a tuple of non-negative integers for linear, bilinear and trilinear upsampling, but got: "
            "{}".format(scale_factor)
        )
    return scale_factor


def upsample_nearest_2d(ctx, input, size, scale_factor):
    assert input.dim() == 4

    _check_size_scale_factor(size, scale_factor)

    ctx.size = size
    ctx.scale_factor = scale_factor

    if ctx.scale_factor is not None and not isinstance(ctx.scale_factor, Integral):
        raise ValueError(
            "scale_factor must be a single Integer value for nearest neighbor sampling"
        )

    if ctx.scale_factor is None:
        if ctx.size[0] % input.size(2) != 0 or ctx.size[1] % input.size(3) != 0:
            raise RuntimeError(
                "output size specified in UpsamplingNearest "
                "({}) has to be divisible by the input size, but got: "
                "{}".format(
                    "x".join(map(str, ctx.size)), "x".join(map(str, input.size()))
                )
            )
        ctx.scale_factor = ctx.size[0] // input.size(2)
        if ctx.scale_factor != ctx.size[1] // input.size(3):
            raise RuntimeError("input aspect ratio doesn't match the " "output ratio")

    output = input.new()
    backend = type2backend[type(input)]
    ctx.save_for_backward(input)
    backend.SpatialUpSamplingNearest_updateOutput(
        backend.library_state, input, output, ctx.scale_factor
    )
    return output
