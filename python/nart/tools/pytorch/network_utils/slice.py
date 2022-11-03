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

from ..onnx_utils import onnx_pb2


def merge_slice_nodes(model):
    """
    A sequence of slice nodes with the same input blob
    corresponds to one holistic Slice layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Slice":
            model.graph.node.extend([node])
            idx += 1
            continue
        attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))
        # Slice layer in caffe only supports one-axis slicing
        # so the slice nodes must share the same axes
        # and they must slice on the same axis
        slicing_axis = None
        segs = []
        while idx < len(nodes):
            idx += 1
            if idx >= len(nodes):
                break
            slice_node = nodes[idx]
            if slice_node.op_type != "Slice" or node.input[0] != slice_node.input[0]:
                break
            node.output.extend(slice_node.output)
            slice_attributes = dict(
                zip([attr.name for attr in slice_node.attribute], slice_node.attribute)
            )
            assert slice_attributes["axes"].ints == attributes["axes"].ints
            for axis_idx, axis in enumerate(attributes["axes"].ints):
                start = attributes["starts"].ints[axis_idx]
                end = attributes["ends"].ints[axis_idx]

                slice_start = slice_attributes["starts"].ints[axis_idx]
                slice_end = slice_attributes["ends"].ints[axis_idx]
                # if not the slicing axis, ignore them
                if (start, end) == (slice_start, slice_end):
                    continue
                if slicing_axis is None:
                    segs.append((start, end))
                    slicing_axis = axis
                assert axis == slicing_axis
                segs.append((slice_start, slice_end))
        # if slice nodes are merged, refill the starts and ends
        if slicing_axis is not None:
            del attributes["axes"].ints[:]
            attributes["axes"].ints.extend([slicing_axis])
            del attributes["starts"].ints[:]
            attributes["starts"].ints.extend([seg[0] for seg in segs])
            del attributes["ends"].ints[:]
            attributes["ends"].ints.extend([seg[1] for seg in segs])
        model.graph.node.extend([node])


def complement_slice_nodes(model):
    """
    Version specific for 1.0.x
    A sequence of slice nodes with consistent blob flow
    corresponds to one holistic Slice layer in caffe.
    This method complements the possibly missing slice nodes among them.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Slice":
            model.graph.node.extend([node])
            idx += 1
            continue
        slice_node = node
        slice_nodes = [slice_node]
        slice_attributes = dict(
            zip([attr.name for attr in slice_node.attribute], slice_node.attribute)
        )
        starts = list(slice_attributes["starts"].ints)
        ends = list(slice_attributes["ends"].ints)
        idx += 1
        while idx < len(nodes):
            node = nodes[idx]
            attributes = dict(
                zip([attr.name for attr in node.attribute], node.attribute)
            )
            if (
                node.op_type != "Slice"
                or node.input != slice_node.input
                or attributes["axes"].ints != slice_attributes["axes"].ints
            ):
                break
            starts.extend(attributes["starts"].ints)
            ends.extend(attributes["ends"].ints)
            slice_nodes.append(node)
            idx += 1
        slice_points = sorted(list(set(starts).union(set(ends))))
        for i in range(len(slice_points) - 1):
            start, end = slice_points[i : (i + 2)]

            extra_slice_node = onnx_pb2.NodeProto()
            extra_slice_node.MergeFrom(slice_node)
            slice_attributes = dict(
                zip(
                    [attr.name for attr in extra_slice_node.attribute],
                    extra_slice_node.attribute,
                )
            )
            slice_attributes["starts"].ints[0] = start
            slice_attributes["ends"].ints[0] = end
            extra_slice_node.output[0] = "{}.{}".format(idx, i)

            if i >= len(slice_nodes):
                slice_nodes.append(extra_slice_node)
                continue
            node = slice_nodes[i]
            attributes = dict(
                zip([attr.name for attr in node.attribute], node.attribute)
            )
            if (
                start != attributes["starts"].ints[0]
                or end != attributes["ends"].ints[0]
            ):
                slice_nodes.insert(i, extra_slice_node)

        model.graph.node.extend(slice_nodes)


def remove_slice_nodes(model):
    """
    Version specific for 1.0.x
    A sequence of slice nodes with consistent blob flow
    corresponds to one holistic Slice layer in caffe.
    This method removes the redundant nonsense slice nodes among them.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Slice":
            model.graph.node.extend([node])
            idx += 1
            continue
        inp = node.input
        output = node.output
        while idx < len(nodes):
            idx += 1
            if idx >= len(nodes):
                model.graph.node.extend([node])
                break
            slice_node = nodes[idx]
            if slice_node.op_type != "Slice" or output != slice_node.input:
                model.graph.node.extend([node])
                break
            slice_node.input.pop()
            slice_node.input.extend(node.input)
            node = slice_node
            output = node.output
