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


def merge_biased_deconvolution_nodes(model):
    """
    Version specific for 0.3.0 and 0.3.1
    A sequence of constant-convtranspose-add nodes with consistent blob flow
    corresponds to one holistic Deconvolution layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Constant" or idx + 2 >= len(nodes):
            model.graph.node.extend([node])
            idx += 1
            continue

        constant_node = node
        node = nodes[idx + 1]
        if node.op_type != "ConvTranspose" or constant_node.output[0] != node.input[2]:
            model.graph.node.extend([constant_node])
            idx += 1
            continue

        convtranspose_node = node
        node = nodes[idx + 2]
        if node.op_type != "Add" or convtranspose_node.output[0] != node.input[0]:
            model.graph.node.extend([constant_node])
            idx += 1
            continue

        add_node = node
        add_node.input.remove(convtranspose_node.output[0])
        convtranspose_node.input.remove(constant_node.output[0])
        convtranspose_node.input.extend(add_node.input)
        del convtranspose_node.output[:]
        convtranspose_node.output.extend(add_node.output)
        model.graph.node.extend([convtranspose_node])
        idx += 3


def merge_nonbiased_deconvolution_nodes(model):
    """
    Version specific for 0.3.0 and 0.3.1
    A sequence of constant-convtranspose nodes with consistent blob flow
    corresponds to one holistic Deconvolution layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
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
        if node.op_type != "ConvTranspose" or constant_node.output[0] != node.input[2]:
            model.graph.node.extend([constant_node])
            idx += 1
            continue

        convtranspose_node = node
        convtranspose_node.input.remove(convtranspose_node.input[-1])
        model.graph.node.extend([convtranspose_node])
        idx += 2
