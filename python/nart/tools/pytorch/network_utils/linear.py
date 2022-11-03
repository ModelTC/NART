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


def remove_constant_nodes(model):
    """
    Version specific for 0.3.0 and 0.3.1
    A sequence of constant-gemm nodes with consistent blob flow
    corresponds to one holistic InnerProduct layer in caffe.
    This method removes the additional constant node.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Constant" or idx + 1 >= len(nodes):
            model.graph.node.extend([node])
            idx += 1
            continue

        constant_node = node
        node = nodes[idx + 1]
        if node.op_type != "Gemm":
            model.graph.node.extend([constant_node])
            idx += 1
            continue

        gemm_node = node
        if constant_node.output[0] not in gemm_node.input:
            model.graph.node.extend([constant_node])
            idx += 1
            continue

        model.graph.node.extend([gemm_node])
        idx += 2


def merge_gemm_nodes(model):
    """
    Version specific for 0.4.x and 1.0.x
    A sequence of transpose-matmul nodes with consistent blob flow
    corresponds to one holistic InnerProduct layer in caffe.
    This method merges the nodes into one gemm node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Transpose" or idx + 1 >= len(nodes):
            model.graph.node.extend([node])
            idx += 1
            continue

        transpose_node = node
        node = nodes[idx + 1]
        if node.op_type != "MatMul":
            model.graph.node.extend([transpose_node])
            idx += 1
            continue

        matmul_node = node
        if transpose_node.output[0] not in matmul_node.input:
            model.graph.node.extend([transpose_node])
            idx += 1
            continue

        matmul_node.op_type = "Gemm"
        matmul_node.input[1] = transpose_node.input[0]
        model.graph.node.extend([matmul_node])
        idx += 2
