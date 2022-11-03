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

""" This file contains the pass to fuse linear op & BatchNormalization into conv op.
"""

from . import Pass
from ..ops.op import Op, DELIM, Constant
from ..core.match_utils import ConstantMatcher, OpMatcher, AnyAttrValue, make_pattern
import numpy as np

import logging

logger = logging.getLogger("nart.passes")


class ConvFuser(Pass):
    """A pass which can fuse linear ops & BatchNormalization into conv op.

    Args:

    """

    def __init__(self):
        self.patterns = []
        """ Add all patterns enabled.
            Requirements of new pattern:
                matching node of Conv should be named as 'conv',
                matching node of the node to be fused should be named as 'fused_node'
        """
        self.patterns.append(self.get_conv_batchnorm_pattern())
        self.patterns.append(self.get_conv_add_pattern())
        self.patterns.append(self.get_conv_mul_pattern())
        self.patterns.append(self.get_conv_mul_symmetric_pattern())

    def run(self, graph):
        modified = True
        # iterative until no modification is done
        while modified:
            modified = False
            for pattern, extractor in self.patterns:
                for match in graph.match(pattern):
                    conv = match["conv"]
                    consumers = graph.get_tensor_consumer(conv.output[0])
                    if len(consumers) > 1:
                        # if the output of conv is consumed by more than 1 node, skip the fusing.
                        logger.debug(
                            f"skip fusing node into `{conv.name}` since its output is consumed by more than 1 node"
                        )
                        continue
                    W = graph.get_const_tensor_as_array(conv.input[1], allow_none=True)
                    B = (
                        graph.get_const_tensor_as_array(conv.input[2], allow_none=True)
                        if conv.has_input(2)
                        else None
                    )
                    if W is None or (conv.has_input(2) and B is None):
                        # the fusing process requires B and C be constant.
                        logger.debug(
                            f"can't fuse node into `{conv.name}` since its weights are not all constant"
                        )
                        continue
                    # compute new B and C of Conv.
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
                    # reshape before multiply, then we are multiplying scale[i] and W[i].
                    W2 = np.multiply(W, np.expand_dims(scale, axis=[1, 2, 3]))
                    if B is not None:
                        B2 = np.multiply(B, scale) + bias
                    else:
                        B2 = bias
                    # create new conv op.
                    nodes = []
                    W2_op = Constant.make_constant(f"{conv.input[1]}_fused", W2)
                    nodes.append(W2_op)
                    inputs = [conv.input[0], W2_op.output[0]]
                    fused_op = match["fused_op"]
                    if np.count_nonzero(B2) != 0:
                        bias_opname = (
                            f"{conv.input[2]}_fused"
                            if conv.has_input(2)
                            else f"{conv.name}+{fused_op.name}_bias"
                        )
                        B2_op = Constant.make_constant(bias_opname, B2)
                        inputs.append(B2_op.output[0])
                        nodes.append(B2_op)
                    assert (
                        len(fused_op.output) == 1
                    ), f"cannot fuse a node with {len(fused_op.output)} into conv"
                    conv2 = Op.make_op(
                        "Conv",
                        f"{conv.name}+{fused_op.name}",
                        inputs,
                        fused_op.output.copy(),
                        conv.attributes.copy(),
                    )
                    nodes.append(conv2)
                    idx = graph.nodes.index(fused_op)
                    graph.del_node(fused_op)
                    graph.insert_nodes(nodes, idx)

                    modified = True
        graph.update_topology()
        graph.update_tensor_shape()

    @staticmethod
    def get_conv_batchnorm_pattern():
        nodes = [
            # match against any Conv op.
            OpMatcher("conv", {"Conv"}, ["0"], ["1"]),
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
    def get_conv_add_pattern():
        r"""The pattern captures Conv+Add.
        NOTE: Add is commutative, but current graph matcher cannot capture this property.

        X   W&B <const>
         \  /
         Conv  C <const>
           \   /
            Add
             |
             Y
        """
        nodes = [
            # match against any Conv op.
            OpMatcher("conv", {"Conv"}, ["0"], ["1"]),
            ConstantMatcher("C"),
            Op.make_op("Add", "fused_op", ["1", "C"], ["2"]),
        ]
        pattern = make_pattern(nodes, ["0"], ["2"])

        def extractor(match, graph):
            add = match["fused_op"]
            bias = graph.get_const_tensor_as_array(add.input[1], allow_none=False)
            # the add must be broadcasting add except on channel dimension in order to be fused into Conv.
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
    def get_conv_mul_pattern():
        r"""The pattern captures Conv+Mul.
        NOTE: Mul is commutative, but current graph matcher cannot capture this property.

        X    W <const>
         \  /
         Conv   B <const>
           \   /
            Mul
             |
             Y
        """
        nodes = [
            # match against any Conv op.
            OpMatcher("conv", {"Conv"}, ["0"], ["1"]),
            ConstantMatcher("B"),
            Op.make_op("Mul", "fused_op", ["1", "B"], ["2"]),
        ]
        pattern = make_pattern(nodes, ["0"], ["2"])

        # the extractor to extract scale&bias from a match
        def extractor(match, graph):
            mul = match["fused_op"]
            mulitlier = graph.get_const_tensor_as_array(mul.input[1], allow_none=False)
            # the mul must be broadcasting mul except on channel dimension in order to be fused into Conv.
            # that is mulitlier's shape after aligning must all be 1s except on channel dimension.
            inp_shape = graph.get_tensor_shape(mul.input[0])
            multi_shape = mulitlier.shape
            multi_shape = [1] * (len(inp_shape) - len(multi_shape)) + list(multi_shape)
            if not multi_shape[0] == 1 or any(x != 1 for x in multi_shape[2:]):
                return {"success": False}
            # assert multi_shape[1] == inp_shape[1], "bad topology, add shape not broadcastable"
            mulitlier = mulitlier.ravel()
            return {"scale": mulitlier, "bias": None, "success": True}

        return pattern, extractor

    @staticmethod
    def get_conv_mul_symmetric_pattern():
        r"""The pattern captures Conv+Mul.
                  NOTE: Mul is commutative, but current graph matcher cannot capture this property.

                  X    W <const>
                   \  /
        <const> B  Conv
                 \ /
                 Mul
                  |
                  Y
        """
        nodes = [
            # match against any Conv op.
            OpMatcher("conv", {"Conv"}, ["0"], ["1"]),
            ConstantMatcher("B"),
            Op.make_op("Mul", "fused_op", ["B", "1"], ["2"]),
        ]
        pattern = make_pattern(nodes, ["0"], ["2"])

        # the extractor to extract scale&bias from a match
        def extractor(match, graph):
            mul = match["fused_op"]
            mulitlier = graph.get_const_tensor_as_array(mul.input[0], allow_none=False)
            # the mul must be broadcasting mul except on channel dimension in order to be fused into Conv.
            # that is mulitlier's shape after aligning must all be 1s except on channel dimension.
            inp_shape = graph.get_tensor_shape(mul.input[1])
            multi_shape = mulitlier.shape
            multi_shape = [1] * (len(inp_shape) - len(multi_shape)) + list(multi_shape)
            if not multi_shape[0] == 1 or any(x != 1 for x in multi_shape[2:]):
                return {"success": False}
            # assert multi_shape[1] == inp_shape[1], "bad topology, add shape not broadcastable"
            mulitlier = mulitlier.ravel()
            return {"scale": mulitlier, "bias": None, "success": True}

        return pattern, extractor
