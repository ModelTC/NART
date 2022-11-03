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


class Slice(ONNXNodeParser):
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    @classmethod
    def get_slice_info(cls, node):
        raise NotImplementedError(
            "Derivated class `{0}` should override get_slice_info method".format(
                cls.__name__
            )
        )

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        def _clamp_index(v, upbound):
            if v < 0:
                return v + upbound
            if v > upbound:
                return upbound
            return v

        ipt = itensor_by_name[node.input[0]]
        input_shape = get_shape(ipt)
        h, w = input_shape.shape_vals[-2:]
        # Why use `and`?
        _is_dynamic = h == -1 and w == -1
        nb_dim = get_nb_dims(ipt)
        # call dericated class to get slice information.
        starts, ends, axes, steps = cls.get_slice_info(node)
        # handle neg-axis
        axes = [i + nb_dim if i < 0 else i for i in axes]
        # create output shape along all axes
        # to create output shape from input shape and slice axes:
        # iterate over all slice axes, if there are any 'not sliced' axis before it,
        # then add a slice from input shape to output shape.
        if _is_dynamic:
            assert nb_dim == 4
            h = input_shape.slice(2).shape_tensor
            w = input_shape.slice(3).shape_tensor
            h_st = to_const_tensor(np.array([0], dtype=np.int32))
            w_st = to_const_tensor(np.array([0], dtype=np.int32))
            for idx, (axis, st, sp) in enumerate(zip(axes, list(starts), list(steps))):
                if axis not in [2, 3]:
                    continue
                if not isinstance(ends, trt.ITensor):
                    ed = min(ends[idx], 2147483647)
                    assert st >= 0 and ed >= 0
                    ed_tensor = to_const_tensor(np.array([ed], dtype=np.int32))
                else:
                    ed_tensor = ends
                st_tensor = to_const_tensor(np.array([st], dtype=np.int32))
                sp_tensor = to_const_tensor(np.array([sp], dtype=np.int32))
                if axis == 2:
                    dim_tensor = h
                else:
                    dim_tensor = w
                clip_op1 = network.add_elementwise(
                    dim_tensor, ed_tensor, op=trt.ElementWiseOperation.MIN
                )
                clip_op2 = network.add_elementwise(
                    dim_tensor, st_tensor, op=trt.ElementWiseOperation.MIN
                )
                ed_tensor = clip_op1.get_output(0)
                st_tensor = clip_op2.get_output(0)
                sub_op = network.add_elementwise(
                    st_tensor, ed_tensor, op=trt.ElementWiseOperation.SUB
                )
                div_op = network.add_elementwise(
                    sub_op.get_output(0),
                    sp_tensor,
                    op=trt.ElementWiseOperation.FLOOR_DIV,
                )
                sub_op2 = network.add_elementwise(
                    to_const_tensor(np.array([0], dtype=np.int32)),
                    div_op.get_output(0),
                    op=trt.ElementWiseOperation.SUB,
                )
                if axis == 2:
                    h = sub_op2.get_output(0)
                    h_st = st_tensor
                else:
                    w = sub_op2.get_output(0)
                    w_st = st_tensor
        parts = []
        temp = []
        last = -1
        for idx, (axis, st, sp) in enumerate(zip(axes, list(starts), list(steps))):
            if last + 1 != axis:
                # there is a segment of axes that are not sliced before `axis`.
                if len(temp) != 0:
                    # first add collected slicing axis.
                    parts.append(ShapeTensor(temp, None))
                temp = []
                # add the axes segment from input_shape
                parts.append(input_shape.slice(slice(last + 1, axis)))
            upbound = input_shape.shape_vals[axis]
            if isinstance(ends, trt.ITensor):
                temp.append(1)
            else:
                ed = ends[idx]
                st = _clamp_index(st, upbound)
                ed = _clamp_index(ed, upbound)
                temp.append((ed - st + sp - 1) // sp)
            last = axis
        if len(temp) != 0:
            parts.append(ShapeTensor(temp, None))
        if last + 1 != nb_dim:
            parts.append(input_shape.slice(slice(last + 1, nb_dim)))

        out_shape = concat_shapes(parts)
        # create slice starts of all axes
        start = [0] * nb_dim

        for axis, st in zip(axes, list(starts)):
            upbound = input_shape.shape_vals[axis]
            start[axis] = _clamp_index(st, upbound)
        step = [1] * nb_dim
        for axis, sp in zip(axes, list(steps)):
            step[axis] = sp
        if _is_dynamic:
            out_shape_tensor = concat_shapes(
                [
                    to_const_tensor(
                        np.array(out_shape.shape_vals[:-2], dtype=np.int32)
                    ),
                    h,
                    w,
                ]
            )
            start_tensor = concat_shapes(
                [to_const_tensor(np.array(start[:-2], dtype=np.int32)), h_st, w_st]
            )
        else:
            start = ShapeTensor(start, None)
            step = ShapeTensor(step, None)

        slice_op = network.add_slice(
            input=ipt,
            start=[0] * nb_dim,
            shape=input_shape.shape_vals,
            stride=[1] * nb_dim,
        )
        slice_op.name = node.name
        if _is_dynamic:
            slice_op.set_input(1, start_tensor)
            slice_op.set_input(2, out_shape_tensor)
            slice_op.stride = step
            # Avoid getting -1 in batch and channel dim
            shuffle_layer = network.add_shuffle(slice_op.get_output(0))
            shuffle_layer.reshape_dims = out_shape.shape_vals[:-2] + [0, 0]
            shuffle_layer.name = f"{node.name}::shuffle"
            outputs = {node.output[0]: shuffle_layer.get_output(0)}
            return shuffle_layer, outputs
        else:
            if not start.is_dynamic and not out_shape.is_dynamic:
                slice_op.start = start.shape_vals
                slice_op.shape = out_shape.shape_vals
                slice_op.stride = step.shape_vals
            else:
                slice_op.set_input(1, start.shape_tensor)
                slice_op.set_input(2, out_shape.shape_tensor)
                slice_op.set_input(3, step.shape_tensor)
            outputs = {node.output[0]: slice_op.get_output(0)}
            return slice_op, outputs


class Slice_1(Slice, OpDesc, op_type="Slice", version=1, parser=real_parser):
    @classmethod
    def get_slice_info(cls, node):
        ipt = cls.get_itensor_by_name(node.input[0])
        nb_dim = get_nb_dims(ipt)
        # slice-1
        starts = node.get_attribute_value("starts")
        ends = node.get_attribute_value("ends")
        axes = node.get_attribute_value("axes", list(range(nb_dim)))
        steps = node.get_attribute_value("steps", [1] * nb_dim)
        return starts, ends, axes, steps


class Slice_10(Slice, OpDesc, op_type="Slice", version=10, parser=real_parser):
    # slice-10 +
    @classmethod
    def get_slice_info(cls, node):
        # ctx = Parser
        ipt = cls.get_itensor_by_name(node.input[0])
        nb_dim = get_nb_dims(ipt)
        # starts
        starts = cls.get_const_array(node.input[1])
        assert starts is not None, "dynamic slice not implemented"
        # ends
        ends = cls.get_const_array(node.input[2])
        if ends is None:
            ends = cls.get_itensor_by_name(node.input[2])
            assert len(ends.shape) == 1
        else:
            ends = [int(x) for x in ends.tolist()]

        if node.has_input(3):
            axes = cls.get_const_array(node.input[3])
            assert axes is not None, "slice axes must be constant"
            axes = list(axes)
        else:
            axes = range(nb_dim)
        if node.has_input(4):
            steps = cls.get_const_array(node.input[4])
            assert steps is not None, "slice steps must be constant"
            steps = list(steps)
            if any(step < 0 for step in steps):
                raise NotImplementedError("negatively strided slice not implemented")
        else:
            steps = [1] * nb_dim
        return starts, ends, axes, steps
