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

""" This file contains passes that standarize custom ops to onnx ops.
"""
from . import Pass
from ..core import Graph
from ..ops.op import Constant, Op
from ..utils.caffe_utils.caffe_to_onnx import adjusted_scale
import numpy as np


class StandardizeUpsample(Pass):
    """The Upsample op exported by nart.tools is not standard, it doesn't have the second input `scales`,
    instead it store output height&width in attribute. This pass will standardize those ops.
    """

    def run(self, graph: Graph):

        for node in list(graph.nodes):
            if node.op_type != "Upsample" or node.has_input(1):
                continue
            # found an Upsample op which is not standard.
            height = node.get_attribute_value("height")
            width = node.get_attribute_value("width")
            mode = node.get_attribute_value("mode")
            target_shape = [height, width]
            original_shape = graph.get_tensor_shape(node.input[0])[2:]
            scales = [
                adjusted_scale(t, o) for t, o in zip(target_shape, original_shape)
            ]
            scales = [1.0, 1.0] + scales
            scales_op = Constant.make_constant(
                f"{node.name}::scales_const", np.array(scales, np.float32)
            )
            upsample_op = Op.make_op(
                "Upsample",
                node.name,
                [node.input[0], scales_op.output[0]],
                node.output,
                {"mode": mode},
            )
            pos = graph.nodes.index(node)
            graph.del_node_purely(node)
            graph.insert_nodes_purely([scales_op, upsample_op], pos)
        graph.update_topology()
        graph.update_tensor_shape()
