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


def remove_cast_nodes(model):
    """
    Cast node can be safely abandoned in caffe.
    This method removes the redundant cast slice nodes.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    as_input = dict()
    as_output = dict()
    for node in nodes:
        for inp in node.input:
            if inp not in as_input.keys():
                as_input[inp] = []
            as_input[inp].append(node)
        for output in node.output:
            if output not in as_output.keys():
                as_output[output] = []
            as_output[output].append(node)

    filtered_nodes = []
    for cast_node in nodes:
        if cast_node.op_type != "Cast":
            filtered_nodes.append(cast_node)
            continue
        inp = cast_node.input[0]
        output = cast_node.output[0]
        message = "[PRECISION WARNING] Cast node (input: {}, output: {}) automatically abandoned.\n".format(
            inp, output
        )
        message += " This may cause precision loss."
        warnings.warn(message)

        if output not in as_input.keys() or len(as_input[inp]) == 0:
            # not the input of any node, it must be
            # the original output of network
            for i in range(len(model.graph.output)):
                if model.graph.output[i].name == output:
                    model.graph.output[i].name = inp
                    break
            continue

        for node in as_input[output]:
            for i in range(len(node.input)):
                if node.input[i] == output:
                    node.input[i] = inp
                    break

    model.graph.node.extend(filtered_nodes)
