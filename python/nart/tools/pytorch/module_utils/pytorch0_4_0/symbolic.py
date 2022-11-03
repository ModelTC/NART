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

from torch.nn.modules.utils import _single, _pair, _triple
import warnings


def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def view(g, self, size):
    return g.op("Reshape", self, shape_i=size)


def softmax(g, input, dim=None):
    if dim < 0:
        dim = len(input.type().sizes()) + dim
    return g.op("Softmax", input, axis_i=dim)


def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        ceil_number = 1
    else:
        ceil_number = 0
    if set(_pair(dilation)) != {1}:
        return _unimplemented("max_pool2d", "dilation")
    if not stride:
        stride = kernel_size
    r = g.op(
        "MaxPool",
        input,
        kernel_shape_i=_pair(kernel_size),
        pads_i=_pair(padding) * 2,
        strides_i=_pair(stride),
        ceil_mode_i=ceil_number,
    )
    return r, None


def avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad):
    if ceil_mode:
        ceil_number = 1
    else:
        ceil_number = 0
    if not stride:
        stride = kernel_size
    return g.op(
        "AveragePool",
        input,
        kernel_shape_i=_pair(kernel_size),
        pads_i=_pair(padding),
        strides_i=_pair(stride),
        ceil_mode_i=ceil_number,
    )


def upsample_nearest2d(g, input, scale_factor):
    return g.op(
        "Upsample",
        input,
        width_scale_f=scale_factor,
        height_scale_f=scale_factor,
        mode_s="nearest",
    )


def upsample_bilinear2d(g, input, output_size, align_corners=False):
    if align_corners:
        _unimplemented("upsample_bilinear2d", "align_corners == True")
    w_scale = float(output_size[-1]) / input.type().sizes()[-1]
    h_scale = float(output_size[-2]) / input.type().sizes()[-2]
    return g.op(
        "Upsample",
        input,
        width_scale_f=w_scale,
        height_scale_f=h_scale,
        mode_s="bilinear",
    )


def hardtanh(g, input, min_val, max_val, inplace=False):
    return g.op(
        "Hardtanh", input, min_val_f=_scalar(min_val), max_val_f=_scalar(max_val)
    )
