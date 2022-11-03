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


def merge_shuffle_channel_nodes(model):
    """
    A sequence of (5-channel reshape -> transpose -> 4-channel reshape) nodes
    corresponds to one holistic ShuffleChannel layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]

        if idx + 2 >= len(nodes) or node.op_type != "Reshape":
            model.graph.node.extend([node])
            idx += 1
            continue

        if node.op_type == "Reshape":
            attributes = dict(
                zip([attr.name for attr in node.attribute], node.attribute)
            )
            shape_key = None
            if "shape" in attributes.keys():
                shape_key = "shape"
            elif "dims" in attributes.keys():
                shape_key = "dims"
            else:
                raise KeyError(attributes.keys)
            if len(attributes[shape_key].ints) != 5:
                model.graph.node.extend([node])
                idx += 1
                continue
        group = attributes[shape_key].ints[1]

        transpose_node = nodes[idx + 1]
        transpose_attributes = dict(
            zip(
                [attr.name for attr in transpose_node.attribute],
                transpose_node.attribute,
            )
        )
        if transpose_node.op_type != "Transpose" or list(
            transpose_attributes["perm"].ints
        ) != [0, 2, 1, 3, 4]:
            model.graph.node.extend([node])
            idx += 1
            continue

        reshape_node = nodes[idx + 2]
        reshape_attributes = dict(
            zip([attr.name for attr in reshape_node.attribute], reshape_node.attribute)
        )
        if reshape_node.op_type != "Reshape":
            model.graph.node.extend([node])
            idx += 1
            continue
        shape_key = None
        if "shape" in reshape_attributes.keys():
            shape_key = "shape"
        elif "dims" in reshape_attributes.keys():
            shape_key = "dims"
        else:
            raise KeyError(reshape_attributes.keys)

        if reshape_attributes[shape_key].ints[1] != -1:
            model.graph.node.extend([node])
            idx += 1
            continue

        node.op_type = "ShuffleChannel"
        del node.output[:]
        while len(node.attribute) > 0:
            node.attribute.pop()
        from onnx import helper

        group_attr = helper.make_attribute("group", group)
        node.attribute.extend([group_attr])
        node.output.extend(reshape_node.output)
        model.graph.node.extend([node])
        idx += 3
