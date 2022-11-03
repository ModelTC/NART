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


def delete_concat_nodes(model):
    replace_dict = {}
    delete_node_list = []
    nodes = model.graph.node
    for node in nodes:
        idx = 0
        while idx < len(node.input):
            if node.input[idx] in replace_dict:
                node.input[idx] = replace_dict[node.input[idx]]
            idx += 1

        if node.op_type == "Concat" and len(node.input) == 1:
            replace_dict[node.output[0]] = node.input[0]
            delete_node_list.append(node)
        else:
            for output in node.output:
                if output in replace_dict:
                    replace_dict.pop(output)

    for node in delete_node_list:
        nodes.remove(node)
