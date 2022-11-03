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

from ...core import Graph
from ...core import Node
from ...core.match_utils import ConstantMatcher, OpMatcher, AnyAttrValue, make_pattern
from ...passes import Pass
from ...ops import Op
import math


class ExtractGemm(Pass):
    """
    input   weight
      |       |
      \   [Transpose(1, 0)]
       \     /
       [MatMul]
          |
        output

    or

    input   weight
      |       |
      \   [Transpose(1, 0)]
       \     /
      [MatMul]
          | bias
          |  /
         [Add]
          |
        output

    """

    def __init__(self):
        from onnx.helper import make_attribute

        trans = Node.make_node(
            "tr", "Transpose", ["0"], ["1"], [make_attribute("perm", [1, 0])]
        )
        matmul = Node.make_node("matmul", "MatMul", ["2", "1"], ["3"])
        self.pattern = Graph.make_graph(
            "fc module", {}, [trans, matmul], ["0", "2"], ["3"]
        )
        self.pattern.update_topology()

        biased_trans = Node.make_node(
            "tr", "Transpose", ["0"], ["1"], [make_attribute("perm", [1, 0])]
        )
        biased_matmul = Node.make_node("matmul", "MatMul", ["2", "1"], ["3"])
        bias = Node.make_node("add", "Add", ["3", "4"], ["5"])
        self.biased_pattern = Graph.make_graph(
            "fc module", {}, [trans, matmul, bias], ["0", "2", "4"], ["5"]
        )
        self.biased_pattern.update_topology()

        nodes = [
            ConstantMatcher("weight"),
            Op.make_op("MatMul", "matmul", ["0", "weight"], ["2"]),
            ConstantMatcher("bias"),
            Op.make_op("Add", "add", ["2", "bias"], ["4"]),
        ]
        self.biased_xxpattern = make_pattern(nodes, ["0"], ["4"])

        nodes = [
            ConstantMatcher("weight"),
            Op.make_op("MatMul", "matmul", ["0", "weight"], ["2"]),
        ]
        self.xxpattern = make_pattern(nodes, ["0"], ["2"])

    def run(self, graph):
        from onnx.helper import make_attribute

        def make_gemm(r):
            inp = r["matmul"].input[0]
            if "tr" in r:
                # has weight transpose, so w is the input of transpose,
                # since it will be used as input of Gemm.
                w = r["tr"].input[0]
                transpose_b = 1
            else:
                w = r["matmul"].input[1]
                transpose_b = 0
            if "add" in r:
                # has bias add
                out = r["add"].output[0]
                b = r["add"].input[1]
            else:
                out = r["matmul"].output[0]
                b = None
            lsp = graph.get_tensor_shape(inp)
            # rsp is the shape of second input into matmul
            rsp = graph.get_tensor_shape(r["matmul"].input[1])
            res = []

            # computation should be (.., k) * (k, n) -> (.., n)
            if len(rsp) != 2 or lsp[-1] != rsp[0]:
                return None

            if len(lsp) != 2:
                reshape_name = r["matmul"].name + "::reshape0"
                reshape_out = r["matmul"].name + "_reshape_out"
                reshape = Op.make_op(
                    "Reshape",
                    reshape_name,
                    [inp],
                    [reshape_out],
                    {"shape": make_attribute("shape", [-1, rsp[0]])},
                )
                inp = reshape_out
                res.append(reshape)

                gemm_out = out + "_reshape_out"

                inputs = [inp, w, b] if b else [inp, w]
                gemm = Op.make_op(
                    "Gemm",
                    r["matmul"].name,
                    inputs,
                    [gemm_out],
                    {"transB": transpose_b},
                )
                res.append(gemm)

                reshape_name = r["matmul"].name + "::reshape1"
                out_sp = graph.get_tensor_shape(out)
                reshape = Op.make_op(
                    "Reshape",
                    reshape_name,
                    [gemm_out],
                    [out],
                    {"shape": make_attribute("shape", [-1] + out_sp[1:])},
                )
                res.append(reshape)
            else:
                inputs = [inp, w, b] if b else [inp, w]
                gemm = Op.make_op(
                    "Gemm", r["matmul"].name, inputs, [out], {"transB": transpose_b}
                )
                res.append(gemm)
            return res

        for r in graph.match(self.biased_pattern):
            res = make_gemm(r)
            if not res:
                continue

            graph.insert_nodes(res, graph.nodes.index(r["tr"]))
            [graph.del_node_purely(r[n]) for n in r]
            graph.update_topology()
            graph.update_tensor_shape()

        for r in graph.match(self.pattern):
            res = make_gemm(r)
            if not res:
                continue

            graph.insert_nodes(res, graph.nodes.index(r["tr"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()

        for r in graph.match(self.biased_xxpattern):
            res = make_gemm(r)
            if not res:
                continue

            graph.insert_nodes(res, graph.nodes.index(r["matmul"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()

        for r in graph.match(self.xxpattern):
            res = make_gemm(r)
            if not res:
                continue

            graph.insert_nodes(res, graph.nodes.index(r["matmul"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()

        for n in list(graph.nodes):
            if n.op_type == "MatMul":
                lsp = graph.get_tensor_shape(n.input[0])
                rsp = graph.get_tensor_shape(n.input[1])
                if not (len(lsp) == 2 and len(rsp) == 2):
                    continue
                gemm = Op.make_op("Gemm", n.name, n.input, n.output)
                graph.insert_nodes([gemm], graph.nodes.index(n))
                graph.del_node_purely(n)
                graph.update_topology()
                graph.update_tensor_shape()


from ...core.match_utils import ConstantMatcher


class ExtractHswish(Pass):
    """This pass find and convert Hswish constructure, which looks like
    input [Constant(3)]
      |   /
     / \ /
     | [Add]
     |    |
     |  [Clip(min=0, max=6)]
     \   /
     [Mul] [Constant(6)]
       \   /
       [Div]
         |
       output
    """

    def __init__(self):
        import numpy as np
        from onnx import numpy_helper
        from onnx.helper import make_attribute

        cons3 = ConstantMatcher("cons3", value=np.array(3).astype("float32"))
        addn = Node.make_node("addn", "Add", ["0", "cons3"], ["2"])
        clipn = Node.make_node(
            "clipn",
            "Clip",
            ["2"],
            ["3"],
            [make_attribute("min", 0.0), make_attribute("max", 6.0)],
        )
        muln = Node.make_node("muln", "Mul", ["0", "3"], ["4"])
        cons6 = ConstantMatcher("cons6", value=np.array(6).astype("float32"))
        divn = Node.make_node("divn", "Div", ["4", "cons6"], ["6"])

        self.pattern = Graph.make_graph(
            "hswish module", {}, [cons3, addn, clipn, muln, cons6, divn], ["0"], ["6"]
        )
        self.pattern.update_topology()

    def run(self, graph):
        for r in graph.match(self.pattern):
            for n in r.values():
                if n.op_type == "Add":
                    op_name = n.name
                    inp = n.input[0]
                if n.op_type == "Div":
                    out = n.output[0]
            hswish = Op.from_onnx_node(
                Node.make_node(op_name, "Hswish", [inp], [out]).dump_to_onnx()
            )
            graph.insert_nodes([hswish], graph.nodes.index(r["cons3"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()


class ExtractHsigmoid(Pass):
    """This pass find and convert Hsigmoid constructure, which looks like
           input [Constant(3)]
             |   /
              \ /
             [Add]
               |
    [Clip(min=0, max=6)]
              |    [Constant(6)]
              \   /
              [Div]
                |
              output
    """

    def __init__(self):
        import numpy as np
        from onnx import numpy_helper
        from onnx.helper import make_attribute

        cons3 = ConstantMatcher("cons3", value=np.array(3).astype("float32"))
        addn = Node.make_node("addn", "Add", ["0", "cons3"], ["2"])
        clipn = Node.make_node(
            "clipn",
            "Clip",
            ["2"],
            ["3"],
            [make_attribute("min", 0.0), make_attribute("max", 6.0)],
        )
        cons6 = ConstantMatcher("cons6", value=np.array(6).astype("float32"))
        divn = Node.make_node("divn", "Div", ["3", "cons6"], ["6"])

        self.pattern = Graph.make_graph(
            "hsigmoid module", {}, [cons3, addn, clipn, cons6, divn], ["0"], ["6"]
        )
        self.pattern.update_topology()

    def run(self, graph):
        from onnx.helper import make_attribute

        for r in graph.match(self.pattern):
            for n in r.values():
                if n.op_type == "Add":
                    op_name = n.name
                    inp = n.input[0]
                if n.op_type == "Div":
                    out = n.output[0]
            hsigmoid = Op.from_onnx_node(
                Node.make_node(
                    op_name,
                    "Hsigmoid",
                    [inp],
                    [out],
                    [make_attribute("alpha", 1 / 6.0), make_attribute("beta", 0.5)],
                ).dump_to_onnx()
            )
            graph.insert_nodes([hsigmoid], graph.nodes.index(r["cons3"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()


class EliminateReshapeNode:
    """Eliminate specific *Reshape* node which alters initializer/constant's shape. It can help succesor node acquire weights directly."""

    def run(self, graph):
        candidate_node = ["Reshape", "Squeeze", "Unsqueeze"]
        for n in graph.nodes:
            if n.op_type in candidate_node:
                inp = n.input[0]
                prod = graph.get_tensor_producer(inp)
                if inp not in graph.initializer or (
                    isinstance(prod[0], Node) and prod[0].op_type != "Constant"
                ):
                    # target tensor is input or activation
                    continue

                if len(graph.get_tensor_consumer(inp)) != 1:
                    # if inp is referenced by multi nodes, we can not do resize directly.
                    continue

                if inp in graph.initializer:
                    # initializer
                    graph.initializer[inp].dims[:] = graph.get_tensor_shape(n.output[0])
                elif isinstance(prod[0], Node) and prod[0].op_type == "Constant":
                    # constant
                    prod[0].attributes["value"].t.dims[:] = graph.get_tensor_shape(
                        n.output[0]
                    )

                for o in graph.get_tensor_consumer(n.output[0]):
                    # replace input
                    if isinstance(o, Node):
                        o.replace_input_purely(n.output[0], inp)
                graph.del_node_purely(n)
        graph.update_topology()


class ExtractGeLU(Pass):
    """This pass find and convert GeLU constructure, which looks like
    input [Constant(3)]
      |   /
     / \ /
     | [Add]
     |    |
     |  [Clip(min=0, max=6)]
     \   /
     [Mul] [Constant(6)]
       \   /
       [Div]
         |
       output
    """

    def __init__(self):
        import numpy as np
        from onnx import numpy_helper
        from onnx.helper import make_attribute

        cons3 = ConstantMatcher("cons3", value=np.array(3).astype("float32"))
        pown = Node.make_node("pown", "Pow", ["0", "cons3"], ["2"])
        cons_0_044715 = ConstantMatcher(
            "cons_0_044715", value=np.array(0.044715).astype("float32")
        )
        mul0 = Node.make_node("mul0", "Mul", ["2", "cons_0_044715"], ["4"])
        add0 = Node.make_node("add0", "Add", ["0", "4"], ["5"])
        cons_rec_pi = ConstantMatcher(
            "cons_rec_pi", value=np.array(math.sqrt(2.0 / math.pi)).astype("float32")
        )
        mul1 = Node.make_node("mul1", "Mul", ["5", "cons_rec_pi"], ["7"])
        tanh = Node.make_node("tanh", "Tanh", ["7"], ["8"])
        cons1 = ConstantMatcher("cons1", value=np.array(1).astype("float32"))
        add1 = Node.make_node("add1", "Add", ["8", "cons1"], ["10"])
        cons_0_5 = ConstantMatcher("cons_0_5", value=np.array(0.5).astype("float32"))
        mul2 = Node.make_node("mul2", "Mul", ["0", "cons_0_5"], ["12"])
        mul3 = Node.make_node("mul3", "Mul", ["12", "10"], ["13"])

        self.pattern = Graph.make_graph(
            "gelu module",
            {},
            [
                cons3,
                pown,
                cons_0_044715,
                mul0,
                add0,
                cons_rec_pi,
                mul1,
                tanh,
                cons1,
                add1,
                cons_0_5,
                mul2,
                mul3,
            ],
            ["0"],
            ["13"],
        )
        self.pattern.update_topology()

    def run(self, graph):
        for r in graph.match(self.pattern):
            inp = r["pown"].input[0]
            out = r["mul3"].output[0]
            op_name = r["pown"].name

            gelu = Op.from_onnx_node(
                Node.make_node(op_name, "GeLU", [inp], [out]).dump_to_onnx()
            )
            graph.insert_nodes([gelu], graph.nodes.index(r["pown"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()


class ExtractLayerNorm(Pass):
    """This pass find and convert LayerNorm constructure, which looks like"""

    def __init__(self):
        import numpy as np
        from onnx import numpy_helper
        from onnx.helper import make_attribute

        reduce_0 = Node.make_node(
            "reduce_0", "ReduceMean", ["0"], ["1"], [make_attribute("axes", [-1])]
        )
        sub_0 = Node.make_node("sub_0", "Sub", ["0", "1"], ["2"])
        cons_2 = ConstantMatcher("cons_2", value=np.array(2).astype("float32"))
        pow_0 = Node.make_node("pow_0", "Pow", ["2", "cons_2"], ["4"])
        reduce_1 = Node.make_node(
            "reduce_1", "ReduceMean", ["4"], ["5"], [make_attribute("axes", [-1])]
        )
        cons_eps = ConstantMatcher("cons_eps", pshape=[1], tolerant_shape=True)
        add_0 = Node.make_node("add_0", "Add", ["5", "cons_eps"], ["7"])
        sqrt_0 = Node.make_node("sqrt_0", "Sqrt", ["7"], ["8"])
        div_0 = Node.make_node("div_0", "Div", ["2", "8"], ["9"])
        cons_w = ConstantMatcher("cons_w")
        mul_0 = Node.make_node("mul_0", "Mul", ["9", "cons_w"], ["11"])
        cons_b = ConstantMatcher("cons_b")
        add_1 = Node.make_node("add_1", "Add", ["11", "cons_b"], ["13"])

        self.pattern = Graph.make_graph(
            "layernorm module",
            {},
            [
                reduce_0,
                sub_0,
                cons_2,
                pow_0,
                reduce_1,
                cons_eps,
                add_0,
                sqrt_0,
                div_0,
                cons_w,
                mul_0,
                cons_b,
                add_1,
            ],
            ["0"],
            ["13"],
        )
        self.pattern.update_topology()

    def run(self, graph):
        for r in graph.match(self.pattern):
            inp = r["reduce_0"].input[0]
            out = r["add_1"].output[0]
            op_name = r["reduce_0"].name
            w = r["mul_0"].input[1]
            b = r["add_1"].input[1]

            layernorm = Op.from_onnx_node(
                Node.make_node(op_name, "LayerNorm", [inp, w, b], [out]).dump_to_onnx()
            )
            graph.insert_nodes([layernorm], graph.nodes.index(r["reduce_0"]))
            [graph.del_node_purely(n) for n in r.values()]
            graph.update_topology()
            graph.update_tensor_shape()
