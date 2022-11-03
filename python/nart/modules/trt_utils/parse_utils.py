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

""" Some utils about network construction
"""
import numpy as np

from .environment import ParserContext

import logging

logger = logging.getLogger("nart.modules.tensorrt")


def to_const_tensor(array, name=None):
    """convert a numpy array to const itensor
    Args:
        array: a numpy array
    Returns:
        a created itensor
    """
    if array.dtype == np.int64:
        logger.warning("tensorrt doesnot support int64 array, converting to int32")
        array = array.astype(np.int32)
    elif array.dtype == np.float64:
        logger.warning("tensorrt doesnot support float64 array, converting to float32")
        array = array.astype(np.float32)
    ctx = ParserContext.get_current()
    from tensorrt import tensorrt as trt

    shape = trt.Dims(array.shape)
    weights = trt.Weights(array)
    constant = ctx.network.add_constant(shape, weights)
    if name is not None:
        constant.name = name
        constant.get_output(0).name = name
    return constant.get_output(0)


def get_nb_dims(tensor):
    """get the number of dimension of a tensorrt.itensor or weights."""
    return len(tensor.shape)


class ShapeTensor(object):
    """represents the shape of a tensor or weight."""

    def __init__(self, shape_vals, shape_tensor=None, shape_tensor_getter=None):
        shape_vals = [int(x) for x in shape_vals]
        self._shape_vals = shape_vals
        is_dynamic = any(x < 0 for x in shape_vals)
        self._is_dynamic = is_dynamic
        self._shape_tensor = shape_tensor
        self._shape_tensor_getter = shape_tensor_getter

    @property
    def is_dynamic(self):
        return self._is_dynamic

    def is_dim_dynamic(self, axis):
        return self._shape_vals[axis] < 0

    @property
    def shape_vals(self):
        return self._shape_vals.copy()

    @property
    def shape_tensor(self):
        if self._shape_tensor is not None:
            return self._shape_tensor
        elif self._shape_tensor_getter is not None:
            self._shape_tensor = self._shape_tensor_getter()
            return self._shape_tensor
        else:
            # set shape_tensor if possible
            assert (
                not self.is_dynamic
            ), "creating ShapeTensor of dynamic shape without shape_tensor provided"
            shape_vals = self.shape_vals
            array = np.array(shape_vals, dtype=np.int32)
            self._shape_tensor = to_const_tensor(array)
            return self._shape_tensor

    @property
    def nb_dim(self):
        return len(self.shape_vals)

    @property
    def is_empty(self):
        return self.nb_dim == 0

    def __getitem__(self, arg):
        ret = self.shape_vals[arg]
        if isinstance(ret, (list, tuple)):
            assert not any(
                x < 0 for x in ret
            ), "accessing part of shape which contains dynamic dimension"
        else:
            assert (
                not ret < 0
            ), "accessing part of shape which contains dynamic dimension"
        return ret

    def __len__(self):
        return len(self.shape_vals)

    def slice(self, arg):
        vals = self.shape_vals[arg]
        vals = vals if isinstance(vals, (list, tuple)) else [vals]
        if not any(x < 0 for x in vals):
            # if the slice is static
            return ShapeTensor(vals, None)
        else:
            # use gather layer to select the range
            # first create a indices tensor
            if isinstance(arg, slice):
                indices = range(arg.start or 0, arg.stop or len(self), arg.step or 1)
            else:
                indices = [arg]
            indices = np.array(indices, dtype=np.int32)
            indice_tensor = to_const_tensor(indices)
            ctx = ParserContext.get_current()
            # create gather layer. shape_tensor is a 1 dimension tensor, so gather on axis 0.
            gather = ctx.network.add_gather(self.shape_tensor, indice_tensor, axis=0)
            gather.num_elementwise_dims = 0
            return ShapeTensor(vals, gather.get_output(0))

    def gather(self, indices):
        vals = [self._shape_vals[index] for index in indices]
        if all(x >= 0 for x in vals):
            # all dimension static
            return ShapeTensor(vals)

        indices = np.array(indices, dtype=np.int32)
        indice_tensor = to_const_tensor(indices)
        ctx = ParserContext.get_current()
        # create gather layer. shape_tensor is a 1 dimension tensor, so gather on axis 0.
        gather = ctx.network.add_gather(self.shape_tensor, indice_tensor, axis=0)
        gather.num_elementwise_dims = 0
        return ShapeTensor(vals, gather.get_output(0))


def reduce_mul(shape_tensor):
    assert isinstance(shape_tensor, ShapeTensor)
    if not shape_tensor.is_dynamic:
        # for staic shape, simply reduce multiply
        from functools import reduce

        res = reduce(lambda x, y: x * y, shape_tensor.shape_vals, 1)
        return ShapeTensor([res], None)
    elif shape_tensor.nb_dim == 1:
        # dynamic, but has only one element
        return shape_tensor
    else:
        # for dynamic shape, use reduce layer to multiply it up
        ctx = ParserContext.get_current()
        from tensorrt import tensorrt as trt

        layer = ctx.network.add_reduce(
            shape_tensor.shape_tensor,
            op=trt.ReduceOperation.PROD,
            axes=(1 << 0),
            keep_dims=True,
        )
        return ShapeTensor([-1], layer.get_output(0))


def concat_shape(a, b):
    import tensorrt.tensorrt as trt

    if isinstance(a, trt.ITensor) and isinstance(b, trt.ITensor):
        ctx = ParserContext.get_current()
        concat = ctx.network.add_concatenation([a, b])
        concat.axis = 0
        return concat.get_output(0)
    if a.is_empty:
        return b
    if b.is_empty:
        return a
    if not a.is_dynamic and not b.is_dynamic:
        # both are static, so simply concat it up
        vals = tuple(a.shape_vals) + tuple(b.shape_vals)
        return ShapeTensor(vals, None)
    else:
        # use concat layer to concatenate two parts up
        ctx = ParserContext.get_current()
        concat = ctx.network.add_concatenation([a.shape_tensor, b.shape_tensor])
        concat.axis = 0  # the axis HAS TO BE ZERO.
        vals = tuple(a.shape_vals) + tuple(b.shape_vals)
        return ShapeTensor(vals, concat.get_output(0))


def concat_shapes(shapes):
    ret = shapes[0]
    for item in shapes[1:]:
        ret = concat_shape(ret, item)
    return ret


def get_shape(tensor):
    """get the shape of a tensorrt.itensor or weights.
    Returns:
        for weights or numpy.array: returns tensor.shape.
        for tensorrt.itensor: a ShapeTensor which wraps the tensor shape.
    """
    shape_vals = tensor.shape
    is_dynamic = any(x < 0 for x in shape_vals)
    if not is_dynamic:
        return ShapeTensor(shape_vals, None)
    else:

        def shape_getter():
            ctx = ParserContext.get_current()
            shape_layer = ctx.network.add_shape(tensor)
            return shape_layer.get_output(0)

        return ShapeTensor(shape_vals, shape_getter(), None)


def add_shuffle(tensor, shape):
    import tensorrt.tensorrt as trt

    ctx = ParserContext.get_current()
    reshape = ctx.network.add_shuffle(tensor)
    if isinstance(shape, trt.ITensor):
        reshape.set_input(1, shape)
        return reshape
    if not shape.is_dynamic:
        # for static shape, use reshape_dims parameter
        reshape.reshape_dims = shape.shape_vals
    else:
        # for dynamic shape, use shape tensor
        reshape.set_input(1, shape.shape_tensor)
    return reshape


""" some utility funtions about network creation
"""


def flatten_tensor(tensor, axis):
    shape = get_shape(tensor)

    d0 = reduce_mul(shape.slice(slice(axis)))
    d1 = reduce_mul(shape.slice(slice(axis, None)))
    reshape = add_shuffle(tensor, concat_shape(d0, d1))
    return reshape.get_output(0)


def cast_to(tensor, dtype):
    ctx = ParserContext.get_current()
    identity_layer = ctx.network.add_identity(tensor)
    # identity_layer.name = node.name
    identity_layer.set_output_type(0, dtype)
    return identity_layer.get_output(0)


def dtype_from_art_dtype(art_dtype):
    from ...core.art import Dtype
    import tensorrt.tensorrt as trt

    if art_dtype == Dtype.Float32:
        return trt.float32
    if art_dtype == Dtype.Int32:
        return trt.int32
    if art_dtype == Dtype.Bool:
        return trt.bool
    raise ValueError(f"unexpected dtype {art_dtype}")


def art_dtype_from_dtype(dtype):
    from ...core.art import Dtype
    import tensorrt.tensorrt as trt

    if dtype == trt.float32:
        return Dtype.Float32
    if dtype == trt.int32:
        return Dtype.Int32
    if dtype == trt.bool:
        return Dtype.Bool
    raise ValueError(f"unexpected dtype {dtype}")
