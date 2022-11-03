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
import math


def merge_upsample_nodes(model):
    """
    Version specific for 0.4.1 and 1.0.x
    A sequence of constant-upsample nodes with consistent blob flow
    corresponds to one holistic upsample(interp nninterp) layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
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
            if node.op_type != "Upsample":
                model.graph.node.extend([constant_node])
                break
            upsample_node = node
            assert constant_node.output[0] in upsample_node.input
            upsample_node.input.remove(constant_node.output[0])

            constant_attributes = dict(
                zip(
                    [attr.name for attr in constant_node.attribute],
                    constant_node.attribute,
                )
            )
            scale = np.frombuffer(
                constant_attributes["value"].t.raw_data, dtype=np.float64
            )
            input_size = list(upsample_node.input_shape_dict.values())[0]
            upsample_node.attribute.add(
                i=math.floor(scale[2] * input_size[2]), name="height"
            )
            upsample_node.attribute.add(
                i=math.floor(scale[3] * input_size[3]), name="width"
            )
            model.graph.node.extend([upsample_node])
            idx += 1
            break
