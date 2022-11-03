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

""" Contains parser(s) for a ONNX op, different class may be defined based on the tensorrt version.
    NOTE: This file will not be imported by default, call trt_utils.init_parsers() to import.
          This also means, when adding a new parser file, you should add `import xxx` to trt_utils.init_parsers
"""
from numpy.core.fromnumeric import prod
from ..tensorrt import ONNXNodeParser, TensorrtParser
from ...ops.op import OpDesc
from .parse_utils import (
    get_nb_dims,
    ShapeTensor,
    reduce_mul,
    concat_shape,
    concat_shapes,
    get_shape,
    add_shuffle,
    flatten_tensor,
    to_const_tensor,
)
from tensorrt import tensorrt as trt
import numpy as np
import logging

LOGGER = logging.getLogger("nart.modules.tensorrt")

real_parser = TensorrtParser.get_class()


class Expand(ONNXNodeParser, OpDesc, layertype="Expand", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = itensor_by_name[node.input[0]]
        ipt_shape = get_shape(ipt)
        ipt_dims = get_nb_dims(ipt)
        shape = cls.get_const_array(node.input[1])
        if shape is not None:
            shape = list(shape)
            shape = [int(_) for _ in shape]
        else:
            producers = node.owning_graph.get_tensor_producer(node.input[1])
            if len(producers) != 1:
                raise RuntimeError(
                    f"producer cannot be determined for tensor`{node.input[1]}`, producers={producers}"
                )
            if not hasattr(producers[-1], "to_ndarray"):
                shape = itensor_by_name[node.input[1]]
            else:
                shape = producers[-1].to_ndarray().tolist()
                shape = [int(_) for _ in shape]
        if isinstance(shape, trt.ITensor):
            raise NotImplemented("expand with dynamic shape not implemented")
        new_rank = max(len(shape), ipt_dims)
        new_dims = [1] * (new_rank - ipt_dims) + ipt_shape.shape_vals
        h, w = ipt_shape.shape_vals[2:]
        _is_dynamic = h == -1 and w == -1
        if _is_dynamic:
            assert ipt_shape.shape_vals[0] != -1
            assert ipt_dims == 4, "input dims != 4 for dynamic shape"
            h_t = ipt_shape.slice(2).shape_tensor
            w_t = ipt_shape.slice(3).shape_tensor
            new_dims_t = concat_shapes(
                [to_const_tensor(np.array(new_dims[:2], dtype=np.int32)), h_t, w_t]
            )
            shuffle_layer = network.add_shuffle(ipt)
            shuffle_layer.set_input(1, new_dims_t)
        else:
            shuffle_layer = network.add_shuffle(ipt)
            shuffle_layer.reshape_dims = new_dims
        shuffle_layer.name = f"{node.name}::shuffle1"
        new_ipt = shuffle_layer.get_output(0)
        new_shape = [1] * (new_rank - len(shape)) + shape
        starts = [0] * len(new_dims)
        sizes = list(map(max, zip(new_dims, new_shape)))
        strides = list(map(lambda x: int(x > 1), new_dims))

        slice_layer = network.add_slice(
            new_ipt, start=starts, shape=sizes, stride=strides
        )
        slice_layer.name = node.name
        if _is_dynamic:
            assert new_shape[2] == 1 and new_shape[3] == 1
            starts_t = to_const_tensor(np.array(starts, dtype=np.int32))
            sizes_t = concat_shapes(
                [to_const_tensor(np.array(sizes[:2], dtype=np.int32)), h_t, w_t]
            )
            strides[2] = 1
            strides[3] = 1
            slice_layer.set_input(1, starts_t)
            slice_layer.set_input(2, sizes_t)
            slice_layer.stride = strides
            # Avoid getting -1 in batch and channel dim
            shuffle_layer2 = network.add_shuffle(slice_layer.get_output(0))
            shuffle_layer2.reshape_dims = sizes[:2] + [0, 0]
            shuffle_layer2.name = f"{node.name}::shuffle2"
            outputs = {node.output[0]: shuffle_layer2.get_output(0)}
            return shuffle_layer2, outputs
        else:
            outputs = {node.output[0]: slice_layer.get_output(0)}
            return slice_layer, outputs


class Tile(ONNXNodeParser, OpDesc, layertype="Tile", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        ipt = itensor_by_name[node.input[0]]
        ipt_shape = get_shape(ipt)
        ipt_dims = get_nb_dims(ipt)

        repeats = cls.get_const_array(node.input[1])
        if repeats is not None:
            repeats = list(repeats)
            repeats = [int(_) for _ in repeats]
        else:
            producers = node.owning_graph.get_tensor_producer(node.input[1])
            if not hasattr(producers[-1], "to_ndarray"):
                repeats = itensor_by_name[node.input[1]]
            else:
                repeats = producers[-1].to_ndarray().tolist()
                repeats = [int(_) for _ in repeats]
        if isinstance(repeats, trt.ITensor):
            raise NotImplemented("tile with dynamic repeats not implemented")
        new_dims = list(map(lambda x: x[0] * x[1], zip(ipt_shape.shape_vals, repeats)))
        starts = [0] * ipt_dims
        strides = [1] * ipt_dims
        slice_layer = network.add_slice(
            ipt, start=starts, shape=new_dims, stride=strides
        )
        slice_layer.mode = trt.SliceMode.WRAP
        slice_layer.name = node.name
        _is_dynamic = ipt_shape.shape_vals[2] == -1 and ipt_shape.shape_vals[3] == -1
        if _is_dynamic:
            starts_t = to_const_tensor(np.array(starts, dtype=np.int32))
            for repeat in repeats:
                assert repeat == 1, "Only `repeat = 1` is supported."
            slice_layer.set_input(1, starts_t)
            slice_layer.set_input(2, ipt_shape.shape_tensor)
            # Avoid getting -1 in batch and channel dim
            shuffle_layer = network.add_shuffle(slice_layer.get_output(0))
            shuffle_layer.reshape_dims = ipt_shape.shape_vals[:-2] + [0, 0]
            shuffle_layer.name = f"{node.name}::shuffle"
            outputs = {node.output[0]: shuffle_layer.get_output(0)}
            return shuffle_layer, outputs
        else:
            outputs = {node.output[0]: slice_layer.get_output(0)}
            return slice_layer, outputs
