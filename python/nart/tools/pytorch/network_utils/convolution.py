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


def merge_convolution_nodes(model):
    """
    Version specific for 0.3.0 and 0.3.1
    A sequence of convolution-add nodes with consistent blob flow
    corresponds to one holistic Convolution layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    blobnames = set([blob.name for blob in model.graph.initializer])
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Conv" or idx + 1 >= len(nodes):
            model.graph.node.extend([node])
            idx += 1
            continue

        conv_node = node
        node = nodes[idx + 1]
        # a following add node that takes base blob from
        # convolution node's output and bias from intializing
        # blobs is servient to that convolution node
        if (
            node.op_type != "Add"
            or conv_node.output[0] != node.input[0]
            or node.input[1] not in blobnames
        ):
            model.graph.node.extend([conv_node])
            idx += 1
            continue
        add_node = node
        add_node.input.remove(conv_node.output[0])
        conv_node.input.extend(add_node.input)
        del conv_node.output[:]
        conv_node.output.extend(add_node.output)
        model.graph.node.extend([conv_node])
        idx += 2
