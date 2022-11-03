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

import warnings


def delete_expand_nodes(model):
    replace_dict = {}
    delete_node_list = []
    nodes = model.graph.node
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        temp_idx = 0
        while temp_idx < len(node.input):
            if node.input[temp_idx] in replace_dict:
                # print(node.input[temp_idx], replace_dict[node.input[temp_idx]])
                node.input[temp_idx] = replace_dict[node.input[temp_idx]]
            temp_idx += 1

        if (
            node.op_type == "Constant"
            and idx + 1 < len(nodes)
            and nodes[idx + 1].op_type == "Expand"
        ):
            # print(nodes[idx+1].output[0], nodes[idx+1].input[0])
            replace_dict[nodes[idx + 1].output[0]] = nodes[idx + 1].input[0]
            delete_node_list.extend([node, nodes[idx + 1]])
            idx += 1
        else:
            for output in node.output:
                if output in replace_dict:
                    replace_dict.pop(output)

        idx += 1

    for node in delete_node_list:
        nodes.remove(node)
