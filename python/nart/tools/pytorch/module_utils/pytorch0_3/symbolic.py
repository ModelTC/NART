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

from torch.nn.modules.utils import _pair
import numbers
import warnings


def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    if set(_pair(dilation)) != {1}:
        return _unimplemented("max_pool2d", "dilation")
    if not stride:
        stride = kernel_size
    if ceil_mode:
        ceil_number = 1
    else:
        ceil_number = 0
    r = g.op(
        "MaxPool",
        input,
        kernel_shape_i=_pair(kernel_size),
        pads_i=_pair(padding),
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


def narrow(g, self, dimension, start, length):
    return g.op(
        "Slice", self, axes_i=[dimension], starts_i=[start], ends_i=[start + length]
    )


def unsqueeze(g, self, dim):
    return g.op("Unsqueeze", self, axes_i=[dim])


def max(g, self, *args, **kwargs):
    dim = kwargs.get("dim", None)
    if dim is None and isinstance(args[0], numbers.Number):
        dim = args[0]
    if dim is not None:
        keepdim = kwargs.get("keepdim", False)
        # TODO: export it as ReduceMax
        return g.op(
            "ATen", self, operator_s="max", dim_i=dim, keepdim_i=keepdim, outputs=2
        )
    else:
        (other,) = args
        return g.op("Max", self, other)


def Constant(g, value):
    return g.op("Constant", value_t=value)


def clamp(g, self, min, max):
    return g.op("Hardtanh", self, min_val_f=_scalar(min), max_val_f=_scalar(max))


def hardtanh(g, input, min_val, max_val, inplace=False):
    return g.op(
        "Hardtanh", input, min_val_f=_scalar(min_val), max_val_f=_scalar(max_val)
    )


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]
