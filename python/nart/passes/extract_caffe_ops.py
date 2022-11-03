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

""" Passes that extract CaffeXXX ops from standard ONNX graph.
"""

from . import Pass
from queue import Queue
from ..ops.op import Op, DELIM, Constant

import logging

logger = logging.getLogger("nart.passes")


class ExtractEltwise(Pass):
    """A subgraph of constant-mul-add nodes with consistent blob flow corresponds to one Eltwise layer in caffe.
    This pass extract this pattern and transform those nodes into one Eltwise node.

    **NOTE**: This pass will **NOT** remove any producer nodes of the Add/Sum node's inputs,
    so call *DeadCodeElimination* pass to remove dead nodes after this pass.
    """

    def __init__(self, eager=True):
        """Args:
        eager <bool>: by default, this pass will convert any op which satisfies the restriction of Eltwise,
            when eager is False, only those with coefficients will be converted.
        """
        self._eager = eager

    def run(self, graph):
        from onnx import numpy_helper, helper

        def get_const_scalar(tensor):
            # check if tensor is an constant scalar, returns the value if so, otherwise returns None.
            producer = graph.get_tensor_producer(tensor)
            producer = producer[0]
            if producer == "input" and tensor in graph.initializer:
                array = numpy_helper.to_array(graph.initializer[tensor])
            elif isinstance(producer, Op) and producer.op_type == "Constant":
                array = producer.to_ndarray()
            else:
                # not an initializer nor an output of Constant
                return None
            if all(x == 1 for x in array.shape):
                return array.item(0)
            else:
                return None

        def detect_coef(tensor):
            # detects the multiply pattern of one tensor.
            producer = graph.get_tensor_producer(tensor)
            assert len(producer) > 0, f"cannot find producer of tensor {tensor}"
            producer = producer[0]
            if isinstance(producer, Op) and producer.op_type == "Mul":
                # a multiply node, try to find coefficient
                mul = producer
                coeff = get_const_scalar(mul.input[0])
                if coeff is not None:
                    return mul.input[1], coeff
                coeff = get_const_scalar(mul.input[1])
                if coeff is not None:
                    return mul.input[0], coeff
            return tensor, 1.0

        for node in list(graph.nodes):
            if node.op_type in {"Add", "Sum", "Sub"}:
                inputs = []
                coeffs = []
                for idx, name in enumerate(node.input):
                    ipt, coeff = detect_coef(name)
                    inputs.append(ipt)
                    if node.op_type == "Sub" and idx == 1:
                        coeffs.append(-1.0 * coeff)
                    else:
                        coeffs.append(coeff)
                # check shape of inputs, Eltwise requires all inputs have the same shape.
                shape = graph.get_tensor_shape(inputs[0])
                if any(graph.get_tensor_shape(x) != shape for x in inputs):
                    logger.debug(
                        f"not extracting {node.name} as Eltwise since it's inputs' shapes aren't all same"
                    )
                    continue

                import math

                if all(math.isclose(x, 1.0) for x in coeffs) and not self._eager:
                    # if coefficients are all 1s, and eager mode if off, skip this.
                    continue

                # create Eltwise node.
                eltwise = Op.gen_op("Eltwise", node.name)
                eltwise.input.extend(inputs)
                eltwise.add_output(node.output[0])
                eltwise.add_attribute(helper.make_attribute("coeff", coeffs))
                idx = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_node(eltwise, idx)
        graph.update_topology()
        graph.update_tensor_shape()


class ExtractCaffeNormalize(Pass):
    r"""
    Extract caffe's Normalize layer, it will normalize across channel
    by dividing the L2 norm of vector.

    In ONNX, it's typically represented like:
          A
         /  \
        |   ReduceL2->B
        |      |   eps
        |       \  /
        |       Add->C
        |        |  -1
        |        |  /
         \      Pow->D
          \     /
            Mul->E
    """

    def __init__(self):
        from ..core.match_utils import ConstantMatcher, make_pattern
        import numpy as np

        reducel2 = Op.make_op(
            "ReduceL2", "reducel2", ["A"], ["B"], {"axes": [1], "keepdims": 1}
        )
        eps = ConstantMatcher("eps", [1], tolerant_shape=True)
        add = Op.make_op("Add", "add", ["B", "eps"], "C")
        minus_1 = ConstantMatcher(
            "minus_1", [1], tolerant_shape=True, value=np.array(-1, np.float32)
        )
        exp = Op.make_op("Pow", "pow", ["C", "minus_1"], ["D"])
        mul = Op.make_op("Mul", "mul", ["A", "D"], ["E"])
        nodes = [reducel2, eps, add, minus_1, exp, mul]
        self.pattern = make_pattern(nodes, ["A"], ["E"], "normalize_pettern")

        # the second patter, use Div insteand of Mul.
        reducel2 = Op.make_op(
            "ReduceL2", "reducel2", ["A"], ["B"], {"axes": [1], "keepdims": 1}
        )
        eps = ConstantMatcher("eps", [1], tolerant_shape=True)
        add = Op.make_op("Add", "add", ["B", "eps"], "C")
        div = Op.make_op("Div", "div", ["A", "C"], ["D"])
        nodes = [reducel2, eps, add, div]
        self.pattern2 = make_pattern(nodes, ["A"], ["D"], "normalize_pettern2")

    def run(self, graph):
        import numpy as np

        one = Constant.make_constant("one_for_norm", np.array([1.0], np.float32))
        first = True
        for match in graph.match(self.pattern):
            reducel2 = match["reducel2"]
            A = reducel2.input[0]
            add = match["add"]
            assert add.input[0] == reducel2.output[0], "input order not as expected"
            eps = add.input[1]
            eps = graph.get_const_tensor_as_array(eps).item(0)
            mul = match["mul"]
            E = mul.output[0]
            norm = Op.make_op(
                "CaffeNormalize",
                mul.name,
                [A, "one_for_norm"],
                [E],
                {"eps": eps, "across_spatial": False, "channel_shared": True},
            )
            index = graph.nodes.index(mul)
            graph.del_node(mul)
            nodes = [one, norm] if first else [norm]
            graph.insert_nodes(nodes, index)
            first = False
        for match in graph.match(self.pattern2):
            reducel2 = match["reducel2"]
            A = reducel2.input[0]
            add = match["add"]
            assert add.input[0] == reducel2.output[0], "input order not as expected"
            eps = add.input[1]
            eps = graph.get_const_tensor_as_array(eps).item(0)
            div = match["div"]
            D = div.output[0]
            norm = Op.make_op(
                "CaffeNormalize",
                div.name,
                [A, "one_for_norm"],
                [D],
                {"eps": eps, "across_spatial": False, "channel_shared": True},
            )
            index = graph.nodes.index(div)
            graph.del_node(div)
            nodes = [one, norm] if first else [norm]
            graph.insert_nodes(nodes, index)
            first = False


class ExtractCaffePower(Pass):
    """Extract caffe's Power layer from the pattern Pow(Add(Mul(A, scale), bias), power)."""

    def __init__(self):
        from ..core.match_utils import ConstantMatcher, make_pattern
        import numpy as np

        def gen_pattern(has_scale, has_bias):
            nodes = []
            name = "X"
            if has_scale:
                nodes.append(ConstantMatcher("scale", [1], tolerant_shape=True))
                nodes.append(Op.make_op("Mul", "mul", [name, "scale"], ["A"]))
                name = "A"
            if has_bias:
                nodes.append(ConstantMatcher("bias", [1], tolerant_shape=True))
                nodes.append(Op.make_op("Add", "add", [name, "bias"], ["B"]))
                name = "B"
            nodes.append(ConstantMatcher("exp", [1], tolerant_shape=True))
            nodes.append(Op.make_op("Pow", "pow", [name, "exp"], ["Y"]))
            pattern = make_pattern(nodes, ["X"], ["Y"], "power_pattern")
            return pattern

        self.patterns = [gen_pattern(True, False), gen_pattern(False, False)]

    def run(self, graph):
        import numpy as np

        for pattern in self.patterns:
            for match in graph.match(pattern):
                # FIXME: a better way to find input name
                X = None
                if "mul" in match:
                    scale = match["mul"].input[1]
                    scale = graph.get_const_tensor_as_array(scale)
                    scale = float(scale.item(0))
                    X = match["mul"].input[0]
                else:
                    scale = 1.0
                if "add" in match:
                    bias = match["add"].input[1]
                    bias = graph.get_const_tensor_as_array(bias)
                    bias = float(bias.item(0))
                    X = match["add"].input[0] if X is None else X
                else:
                    bias = 0.0
                pow_op = match["pow"]
                exp = pow_op.input[1]
                exp = graph.get_const_tensor_as_array(exp)
                exp = float(exp.item(0))
                X = pow_op.input[0] if X is None else X
                # make CaffePower op
                power = Op.make_op(
                    "CaffePower",
                    pow_op.name,
                    [X],
                    pow_op.output,
                    {"scale": scale, "shift": bias, "power": exp},
                )
                index = graph.nodes.index(pow_op)
                graph.del_node(pow_op)
                graph.insert_node(power, index)


class ExtractCaffeThreshold(Pass):
    """Extract Caffe's Threashold op, which can compare a tensor to an constant scalar.
    This pass is used to match those Greater Op that compare between a tensor and constant scalar,
    and convert to Threshold Op.
    """

    def __init__(self):
        from ..core.match_utils import ConstantMatcher, make_pattern
        import numpy as np

        thresh = ConstantMatcher("thresh", [1], tolerant_shape=True)
        greater = Op.make_op("Greater", "greater", ["X", "thresh"], ["Y"])
        self._pattern = make_pattern([thresh, greater], ["X"], ["Y"], "thresh-pattern")

    def run(self, graph):
        for match in graph.match(self._pattern):
            greater = match["greater"]
            thresh = greater.input[1]
            thresh = graph.get_const_tensor_as_array(thresh)
            assert thresh.size == 1, "threshold should be a 1-element tensor or scalar"
            thresh = thresh.item(0)
            thresh_op = Op.make_op(
                "CaffeThreshold",
                greater.name,
                [greater.input[0]],
                greater.output,
                {"threshold": thresh},
            )
            index = graph.nodes.index(greater)
            graph.del_node(greater)
            graph.insert_node(thresh_op, index)


class SqrtToCaffePower(Pass):
    """Convert Sqrt op to CaffePower op, using the equation sqrt(x) = x ^ 0.5."""

    def __init__(self):
        super().__init__()

    def run(self, graph):
        for node in graph.nodes:
            if node.op_type != "Sqrt":
                continue
            power = Op.make_op(
                "CaffePower",
                node.name,
                node.input,
                node.output,
                {"scale": 1.0, "shift": 0.0, "power": 0.5},
            )
            pos = graph.nodes.index(node)
            graph.del_node(node)
            graph.insert_node(power, pos)
