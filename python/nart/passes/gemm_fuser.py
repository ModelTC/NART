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

""" This file contains the pass to fuse linear op & BatchNormalization into Gemm op.
"""

from . import Pass
from ..ops.op import Op, DELIM, Constant
from ..core.match_utils import ConstantMatcher, OpMatcher, AnyAttrValue, make_pattern
import numpy as np

import logging

logger = logging.getLogger("nart.passes")


class GemmFuser(Pass):
    """A pass which can fuse linear ops & BatchNormalization into Gemm op.

    Args:

    """

    def __init__(self):
        self.patterns = []
        """ Add all patterns enabled.
            Requirements of new pattern:
                matching node of Gemm should be named as 'gemm',
                matching node of the node to be fused should be named as 'fused_node'
        """
        self.patterns.append(self.get_gemm_batchnorm_pattern())
        self.patterns.append(self.get_gemm_add_pattern())
        self.patterns.append(self.get_gemm_mul_pattern())

    def run(self, graph):
        modified = True
        # iterative until no modification is done
        while modified:
            modified = False
            for pattern, extractor in self.patterns:
                for match in graph.match(pattern):
                    gemm = match["gemm"]
                    consumers = graph.get_tensor_consumer(gemm.output[0])
                    if len(consumers) > 1:
                        # if the output of gemm is consumed by more than 1 node, skip the fusing.
                        logger.debug(
                            f"skip fusing node into `{gemm.name}` since its output is consumed by more than 1 node"
                        )
                        continue
                    B = graph.get_const_tensor_as_array(gemm.input[1], allow_none=True)
                    C = (
                        graph.get_const_tensor_as_array(gemm.input[2], allow_none=True)
                        if gemm.has_input(2)
                        else None
                    )
                    if B is None or (gemm.has_input(2) and C is None):
                        # the fusing process requires B and C be constant.
                        logger.debug(
                            f"can't fuse node into `{gemm.name}` since its weights are not constant"
                        )
                        continue
                    transB = gemm.get_attribute_value("transB", 0)
                    # compute new B and C of Gemm.
                    info = extractor(match, graph)
                    if not info["success"]:
                        continue
                    scale, bias = info["scale"], info["bias"]
                    scale = (
                        scale
                        if scale is not None
                        else np.array([1.0], dtype=np.float32)
                    )
                    bias = bias if bias is not None else np.array([0.0], np.float32)

                    if not transB:
                        B2 = np.multiply(B, scale)
                    else:
                        B2 = np.multiply(B, np.reshape(scale, list(scale.shape) + [1]))

                    if C is None:
                        C2 = bias
                    else:
                        C2 = C * scale + bias
                    # create new Gemm op.
                    nodes = []
                    B2_op = Constant.make_constant(f"{gemm.input[1]}_fused", B2)
                    nodes.append(B2_op)
                    inputs = [gemm.input[0], B2_op.output[0]]
                    fused_op = match["fused_op"]
                    if np.count_nonzero(C2) != 0:
                        bias_opname = (
                            f"{gemm.input[2]}_fused"
                            if gemm.has_input(2)
                            else f"{gemm.name}+{fused_op.name}_bias"
                        )
                        C2_op = Constant.make_constant(bias_opname, C2)
                        inputs.append(C2_op.output[0])
                        nodes.append(C2_op)
                    assert (
                        len(fused_op.output) == 1
                    ), f"cannot fuse a node with {len(fused_op.output)} into Gemm"
                    gemm2 = Op.make_op(
                        "Gemm",
                        f"{gemm.name}+{fused_op.name}",
                        inputs,
                        fused_op.output.copy(),
                        gemm.attributes.copy(),
                    )
                    nodes.append(gemm2)
                    idx = graph.nodes.index(fused_op)
                    graph.del_node(fused_op)
                    graph.insert_nodes(nodes, idx)

                    modified = True
        graph.update_topology()
        graph.update_tensor_shape()

    @staticmethod
    def get_gemm_batchnorm_pattern():
        r"""The pattern captures Gemm+BatchNormalization.
        X    W <const>
         \  /
         Gemm
           |
        BatchNormalization (and all its weights)
           |
           Y
        """
        nodes = [
            # match against any Gemm op.
            OpMatcher("gemm", {"Gemm"}, ["0"], ["1"]),
            ConstantMatcher("scale"),
            ConstantMatcher("bias"),
            ConstantMatcher("mean"),
            ConstantMatcher("var"),
            Op.make_op(
                "BatchNormalization",
                "fused_op",
                ["1", "scale", "bias", "mean", "var"],
                ["2"],
                {
                    "epsilon": AnyAttrValue("epsilon"),
                    "momentum": AnyAttrValue("momentum"),
                },
            ),
        ]
        pattern = make_pattern(nodes, ["0"], ["2"])

        # the extractor to extract scale&bias from a match
        def extractor(match, graph):
            batchnorm = match["fused_op"]
            scale = graph.get_const_tensor_as_array(
                batchnorm.input[1], allow_none=False
            )
            bias = graph.get_const_tensor_as_array(batchnorm.input[2], allow_none=False)
            mean = graph.get_const_tensor_as_array(batchnorm.input[3], allow_none=False)
            var = graph.get_const_tensor_as_array(batchnorm.input[4], allow_none=False)
            epsilon = batchnorm.get_attribute_value(
                "epsilon", batchnorm.attr_dict["epsilon"][1]
            )
            # compute the cononalified bias and mean
            std_var = np.sqrt(var + epsilon)
            adjusted_scale = scale / std_var
            adjusted_bias = bias - mean * adjusted_scale
            return {"scale": adjusted_scale, "bias": adjusted_bias, "success": True}

        return pattern, extractor

    @staticmethod
    def get_gemm_add_pattern():
        r"""The pattern captures Gemm+Add.
        NOTE: Add is commutative, but current graph matcher cannot capture this property.

        X    W <const>
         \  /
         Gemm   B <const>
           \   /
            Add
             |
             Y
        """
        nodes = [
            # match against any Gemm op.
            OpMatcher("gemm", {"Gemm"}, ["0"], ["1"]),
            ConstantMatcher("B"),
            Op.make_op("Add", "fused_op", ["1", "B"], ["2"]),
        ]
        pattern = make_pattern(nodes, ["0"], ["2"])

        # the extractor to extract scale&bias from a match
        def extractor(match, graph):
            add = match["fused_op"]
            bias = graph.get_const_tensor_as_array(add.input[1], allow_none=False)
            # the add must be broadcasting add except on channel dimension in order to be fused into Gemm.
            # that is bias's shape after aligning must all be 1s except on channel dimension.
            inp_shape = graph.get_tensor_shape(add.input[0])
            bias_shape = bias.shape
            bias_shape = [1] * (len(inp_shape) - len(bias_shape)) + list(bias_shape)
            if not bias_shape[0] == 1 or any(x != 1 for x in bias_shape[2:]):
                return {"success": False}
            assert (
                bias_shape[1] == inp_shape[1]
            ), "bad topology, add shape not broadcastable"
            bias = bias.ravel()
            return {"scale": None, "bias": bias, "success": True}

        return pattern, extractor

    @staticmethod
    def get_gemm_mul_pattern():
        r"""The pattern captures Gemm+Mul.
        NOTE: Mul is commutative, but current graph matcher cannot capture this property.

        X    W <const>
         \  /
         Gemm   B <const>
           \   /
            Mul
             |
             Y
        """
        nodes = [
            # match against any Gemm op.
            OpMatcher("gemm", {"Gemm"}, ["0"], ["1"]),
            ConstantMatcher("B"),
            Op.make_op("Mul", "fused_op", ["1", "B"], ["2"]),
        ]
        pattern = make_pattern(nodes, ["0"], ["2"])

        # the extractor to extract scale&bias from a match
        def extractor(match, graph):
            mul = match["fused_op"]
            mulitlier = graph.get_const_tensor_as_array(mul.input[1], allow_none=False)
            # the mul must be broadcasting mul except on channel dimension in order to be fused into Gemm.
            # that is mulitlier's shape after aligning must all be 1s except on channel dimension.
            inp_shape = graph.get_tensor_shape(mul.input[0])
            multi_shape = mulitlier.shape
            multi_shape = [1] * (len(inp_shape) - len(multi_shape)) + list(multi_shape)
            if not multi_shape[0] == 1 or any(x != 1 for x in multi_shape[2:]):
                return {"success": False}
            assert (
                multi_shape[1] == inp_shape[1]
            ), "bad topology, add shape not broadcastable"
            mulitlier = mulitlier.ravel()
            return {"scale": mulitlier, "bias": None, "success": True}

        return pattern, extractor
