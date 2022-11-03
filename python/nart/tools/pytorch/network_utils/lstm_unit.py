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


def merge_lstmcell_nodes(model):
    """
    Version specific for 1.0.x
    A sequence of lstmcell nodes with the same input blob
    corresponds to one holistic LSTMUnit layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "LSTMCell":
            model.graph.node.extend([node])
            idx += 1
            continue
        while idx < len(nodes):
            idx += 1
            if idx >= len(nodes):
                break
            lstmcell_node = nodes[idx]
            if lstmcell_node.op_type != "LSTMCell" or node.input != lstmcell_node.input:
                break
            node.output.extend(lstmcell_node.output)
        model.graph.node.extend([node])
