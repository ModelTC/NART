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


def merge_eltwise_nodes(model):
    """
    A subgraph of constant-mul-add nodes with consistent blob flow
    corresponds to one holistic Eltwise layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
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

    for add_node in nodes:
        if add_node.op_type != "Add":
            model.graph.node.extend([add_node])
            continue

        add_inps = []
        add_coeffs = []
        for inp in add_node.input:
            if inp not in as_output.keys() or len(as_output[inp]) == 0:
                # not the output of any node, it must be
                # the original input of network
                add_inps.append(inp)
                add_coeffs.append(1.0)
                continue
            for node in as_output[inp]:
                if node.op_type == "Mul":
                    mul_node = node
                    constant_node = None
                    if (
                        mul_node.input[0] in as_output.keys()
                        and as_output[mul_node.input[0]][0].op_type == "Constant"
                    ):
                        add_inps.append(mul_node.input[1])
                        constant_node = as_output[mul_node.input[0]][0]
                    elif (
                        mul_node.input[1] in as_output.keys()
                        and as_output[mul_node.input[1]][0].op_type == "Constant"
                    ):
                        add_inps.append(mul_node.input[0])
                        constant_node = as_output[mul_node.input[1]][0]
                    # mul with constant is merged
                    if constant_node is not None:
                        model.graph.node.remove(mul_node)
                        model.graph.node.remove(constant_node)
                        constant_attributes = dict(
                            zip(
                                [attr.name for attr in constant_node.attribute],
                                constant_node.attribute,
                            )
                        )
                        coeff = np.frombuffer(
                            constant_attributes["value"].t.raw_data, dtype=np.float32
                        )[0]
                        add_coeffs.append(coeff)
                    # no constant, treat as normal node
                    else:
                        add_inps.append(inp)
                        add_coeffs.append(1.0)
                # normal node
                else:
                    add_inps.append(inp)
                    add_coeffs.append(1.0)
        del add_node.input[:]
        add_node.input.extend(add_inps)
        attr_coeff = add_node.attribute.add()
        attr_coeff.name = "coeff"
        attr_coeff.floats.extend(add_coeffs)
        model.graph.node.extend([add_node])
