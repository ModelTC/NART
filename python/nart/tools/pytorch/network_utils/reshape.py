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

import numpy as np
import warnings


def merge_reshape_nodes(model):
    """
    Version specific for 0.4.1 and 1.0.x
    A sequence of constant-reshape nodes with consistent blob flow
    corresponds to one holistic Reshape layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    import onnx

    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Constant":
            model.graph.node.extend([node])
            idx += 1
            continue

        constant_node = node
        while idx < len(nodes):
            idx += 1
            if idx >= len(nodes):
                break
            node = nodes[idx]
            if node.op_type != "Reshape":
                model.graph.node.extend([constant_node])
                break
            reshape_node = node
            assert constant_node.output[0] in reshape_node.input
            reshape_node.input.remove(constant_node.output[0])

            constant_attributes = dict(
                zip(
                    [attr.name for attr in constant_node.attribute],
                    constant_node.attribute,
                )
            )
            shape = np.frombuffer(
                constant_attributes["value"].t.raw_data, dtype=np.int64
            )
            reshape_node.attribute.add(
                ints=shape, name="shape", type=onnx.AttributeProto.INTS
            )
            model.graph.node.extend([reshape_node])
            idx += 1
            break


def delete_reshape_nodes(model):

    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        if nodes[idx].op_type != "Reshape":

            model.graph.node.extend([nodes[idx]])
            idx += 1
            continue
        else:
            temp_idx = idx + 1
            repeat_mark = False
            repeat_nodes = []
            use_mark = False
            if temp_idx >= len(nodes):
                model.graph.node.extend([nodes[idx]])
            else:
                while temp_idx < len(nodes):
                    if nodes[temp_idx].op_type == "Reshape":
                        if nodes[idx].output[0] in nodes[temp_idx].input:
                            repeat_nodes.append(nodes[temp_idx])
                            repeat_mark = True
                    else:
                        if nodes[idx].output[0] in nodes[temp_idx].input:
                            use_mark = True
                    temp_idx += 1

                if repeat_mark == True and use_mark == False:
                    for temp_node in repeat_nodes:
                        temp_node.input[0] = nodes[idx].input[0]
                else:
                    model.graph.node.extend([nodes[idx]])
            idx += 1
