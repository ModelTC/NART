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


def merge_axpy_nodes(model):
    """
    A pair of Mul-Add nodes with consistent blobflow
    corresponds to one holistic Axpy layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    blobnames = set([blob.name for blob in model.graph.initializer])

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Mul" or idx + 1 >= len(nodes):
            model.graph.node.extend([node])
            idx += 1
            continue

        mul_node = node
        node = nodes[idx + 1]
        if node.op_type != "Add" or mul_node.output[0] != node.input[0]:
            model.graph.node.extend([mul_node])
            idx += 1
            continue

        add_node = node
        if (
            mul_node.input[0] in blobnames
            or mul_node.input[1] in blobnames
            or add_node.input[0] in blobnames
            or add_node.input[1] in blobnames
        ):
            model.graph.node.extend([mul_node])
            idx += 1
            continue

        mul_node.input.extend([add_node.input[1]])
        mul_node.output[0] = add_node.output[0]
        mul_node.op_type = "Axpy"
        model.graph.node.extend([mul_node])
        idx += 2
