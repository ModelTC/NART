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


def softmax(g, input, dim=None):
    if dim < 0:
        dim = len(input.type().sizes()) + dim
    return g.op("Softmax", input, axis_i=dim)


def max_pool2d_with_indices(
    g, input, kernel_size, stride, padding, dilation, ceil_mode
):
    if ceil_mode:
        ceil_number = 1
    else:
        ceil_number = 0
    if set(_pair(dilation)) != {1}:
        return _unimplemented("max_pool2d_with_indices", "dilation")
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


def _avg_pool(name, tuple_fn):
    def symbolic_fn(
        g, input, kernel_size, stride, padding, ceil_mode, count_include_pad
    ):
        if ceil_mode:
            ceil_number = 1
        else:
            ceil_number = 0
        if not stride:
            stride = kernel_size

        padding = tuple(tuple_fn(padding))

        return g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            pads_i=padding,
            strides_i=tuple_fn(stride),
            ceil_mode_i=ceil_number,
        )

    return symbolic_fn


avg_pool1d = _avg_pool("avg_pool1d", _single)
avg_pool2d = _avg_pool("avg_pool2d", _pair)
avg_pool3d = _avg_pool("avg_pool3d", _triple)


def upsample_bilinear2d(g, input, output_size, align_corners):
    w_scale = float(output_size[-1]) / input.type().sizes()[-1]
    h_scale = float(output_size[-2]) / input.type().sizes()[-2]
    return g.op(
        "Upsample",
        input,
        width_scale_f=w_scale,
        height_scale_f=h_scale,
        mode_s="bilinear",
    )
