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

from ...core import Node
from ...core import Graph
from ...ops import Op
from ...passes import DeadCodeElimination, ConstantToInitializer

import onnx


def canonicalize_name(name: str):
    import re

    p = re.compile("[^a-zA-Z0-9\._-]")
    name = p.sub("_", name)
    return name[:64]


class OnnxDecoder:
    """convert onnx to Net"""

    def __init__(self):
        pass

    def decode(self, onnx_model):
        """ """
        if isinstance(onnx_model, onnx.ModelProto):
            g = onnx_model.graph
            opsets = onnx_model.opset_import
            # print(opsets)
            opsets = [(x.domain, x.version) for x in opsets]
        else:
            g = onnx_model
            opsets = [("", 9)]
        ret_net = Graph.make_graph(
            canonicalize_name(g.name),
            {i.name: i for i in g.initializer},
            [Op.from_onnx_node(n, opsets=opsets) for n in g.node],
            {
                i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                for i in g.input
            },
            {
                i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                for i in g.output
            },
            {
                i.name: Graph.get_nart_dtype_from_onnx(i.type.tensor_type.elem_type)
                for i in g.input
            },
            {
                i.name: Graph.get_nart_dtype_from_onnx(i.type.tensor_type.elem_type)
                for i in g.output
            },
            g.doc_string,
        )

        ret_net.update_topology()
        ret_net.update_tensor_shape()
        DeadCodeElimination().run(ret_net)
        return ret_net
