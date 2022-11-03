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

""" Passes that transform CaffeXXX Ops (which mimic caffe layers) to standard ONNX ops.
"""
from . import Pass
from queue import Queue
from ..ops.op import Op, DELIM, Constant

import logging

logger = logging.getLogger("nart.passes")


class SplitShuffleChannel(Pass):
    """Split a ShuffleChannel op to reshape-transpose-reshape sequence.

    *NOTE*: In current implementation, this transform requires that input shape is fixed except for batch size axis.
    """

    def run(self, graph):
        from ..ops.op import Reshape, Transpose, ShuffleChannel

        warned = False

        for node in list(graph.nodes):
            if isinstance(node, ShuffleChannel):
                if not warned:
                    logger.warning(
                        (
                            " using <SplitShuffleChannel> pass to split ShuffleChannel to reshape-transpose-reshape sequence, "
                            "be careful this is valid only when HW are fixed"
                        )
                    )
                    warned = True
                data = node.input[0]
                data_shape = graph.get_tensor_shape(data)
                ops = []
                # first do the reshape to split into groups
                group = node.get_attribute_value("group")
                C = data_shape[1]
                new_shape = [0, group, -1] + data_shape[2:]
                ops.extend(
                    Reshape.make_reshape(
                        f"{node.name}{DELIM}reshape1",
                        data,
                        f"{data}_splited",
                        new_shape,
                    )
                )
                reshape1_op = ops[-1]
                # then do transpose
                transpose_op = Transpose.make_transpose(
                    f"{node.name}{DELIM}transpose",
                    reshape1_op.output[0],
                    f"{data}_transposed",
                    [0, 2, 1, 3, 4],
                )
                ops.append(transpose_op)
                # then reshape back
                ori_shape = [0] + data_shape[1:]
                ops.extend(
                    Reshape.make_reshape(
                        f"{node.name}{DELIM}reshape2",
                        transpose_op.output[0],
                        node.output[0],
                        ori_shape,
                    )
                )
                pos = graph.nodes.index(node)
                # since this pass don't use any topology, we can use purely APIs of Graph.
                graph.del_node_purely(node)
                graph.insert_nodes_purely(ops, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class SplitCaffeScale(Pass):
    """This pass convert CaffeScale node to Mul+Add."""

    def run(self, graph):
        from ..ops.op import CaffeScale
        import numpy as np

        for node in list(graph.nodes):
            if isinstance(node, CaffeScale):
                scale = graph.get_const_tensor_as_array(node.input[1])
                assert (
                    scale is not None
                ), "Non constant scale input of Caffe Scale layer is not supported"
                if len(node.input) >= 3:
                    bias = graph.get_const_tensor_as_array(node.input[2])
                    assert (
                        bias is not None
                    ), "Non constant bias input of Caffe Scale layer is not supported"
                else:
                    bias = None

                nb_dims = len(graph.get_tensor_shape(node.input[0]))
                axis = node.get_attribute_value("axis", 1)
                # adjust the shape of scale and bias.
                scale = np.reshape(
                    scale,
                    [1] * axis
                    + list(scale.shape)
                    + [1] * (nb_dims - axis - len(scale.shape)),
                )
                if bias is not None:
                    bias = np.reshape(
                        bias,
                        [1] * axis
                        + list(bias.shape)
                        + [1] * (nb_dims - axis - len(bias.shape)),
                    )
                # create nodes.
                from ..ops.op import Mul, Add

                ret = []
                scale_const = Constant.make_constant(
                    f"{node.name}{DELIM}scale_const", scale
                )
                mul_out_name = node.output[0] if bias is None else f"{node.name}_interm"
                mul_op = Mul.make_mul(
                    f"{node.name}{DELIM}mul",
                    node.input[0],
                    scale_const.output[0],
                    mul_out_name,
                )
                ret.extend([scale_const, mul_op])
                if bias is not None:
                    bias_const = Constant.make_constant(
                        f"{node.name}{DELIM}bias_const", bias
                    )
                    add_op = Add.make_add(
                        f"{node.name}{DELIM}add",
                        mul_out_name,
                        bias_const.output[0],
                        node.output[0],
                    )
                    ret.extend([bias_const, add_op])
                pos = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_nodes(ret, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class ConvertCaffeBatchNorm(Pass):
    """Convert CaffeBatchNorm to BatchNormalization."""

    def run(self, graph):
        from ..ops.op import CaffeBatchNorm, Constant
        import numpy as np

        for node in list(graph.nodes):
            new_nodes = []
            if not isinstance(node, CaffeBatchNorm):
                continue
            mean = node.input[1]
            var = node.input[2]
            mvf = node.input[3]
            mvf_const = graph.get_const_tensor_as_array(mvf)
            import math

            if mvf_const is not None and math.isclose(mvf_const.item(0), 1.0):
                # mvf is a constant, and equals 1.0
                pass
            else:
                div1 = Op.make_op(
                    "Div", f"{node.name}::div1", [mean, mvf], [f"{mean}_div_{mvf}"]
                )
                div2 = Op.make_op(
                    "Div", f"{node.name}::div2", [var, mvf], [f"{var}_div_{mvf}"]
                )
                mean = div1.output[0]
                var = div2.output[0]
                new_nodes.extend([div1, div2])
            shape = graph.get_tensor_shape(mean)
            scales_op = Constant.make_constant(
                f"{node.name}_scale_const", np.ones(shape, dtype=np.float32)
            )
            bias_op = Constant.make_constant(
                f"{node.name}_bias_const", np.ones(shape, dtype=np.float32)
            )
            new_nodes.extend([scales_op, bias_op])

            attrs = {"epsilon": node.get_attribute_value("eps", 1e-5)}
            batchnormalize = Op.make_op(
                "BatchNormalization",
                f"{node.name}",
                [node.input[0], scales_op.output[0], bias_op.output[0], mean, var],
                [node.output[0]],
                attrs,
            )
            new_nodes.append(batchnormalize)
            pos = graph.nodes.index(node)
            graph.del_node(node)
            graph.insert_nodes(new_nodes, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class ConvertDualInputScale(Pass):
    """CaffeScale op can have more than 1 input, which makes it an broadcast multiplication operation
    rather than a ordinary scale.
    This pass convert such op into Reshape + Mul ops.
    """

    def run(self, graph):
        from ..ops.op import Reshape

        for node in list(graph.nodes):
            if node.op_type != "CaffeScale":
                continue
            ipt = node.input[0]
            scale = node.input[1]
            if scale in graph.initializer:
                continue
            prod = graph.get_tensor_producer(scale)
            assert len(prod) == 1, "bad graph topology"
            prod = prod[0]
            if (prod == "input" and scale in graph.initializer) or (
                prod != "input" and prod.op_type == "Constant"
            ):
                continue
            # this is an dual input scale.
            new_nodes = []
            axis = node.get_attribute_value("axis")
            ndim_ipt = len(graph.get_tensor_shape(ipt))
            ndim_scale = len(graph.get_tensor_shape(scale))
            if axis + ndim_scale < ndim_ipt:
                # if the scale is not aligned with input when using Mul to multiply then,
                # a reshape op is required to expand the dimension of scale to (ndim_ipt - axis)
                aligned_scale = f"{scale}_reshaped_at_{node.name}_1"
                new_nodes.extend(
                    # unsqueeze scale to append spatial axes.
                    Reshape.make_reshape(
                        f"{node.name}::reshape_scale",
                        scale,
                        aligned_scale,
                        [0] * ndim_scale + [1] * (ndim_ipt - axis - ndim_scale),
                    )
                )
            else:
                aligned_scale = scale
            bias = node.input[2] if node.has_input(2) else None
            assert bias is None, "Scale with bias term not handled"
            mul_out = node.output[0]
            mul_op = Op.make_op(
                "Mul", f"{node.name}::mul", [ipt, aligned_scale], [mul_out]
            )
            new_nodes.append(mul_op)

            pos = graph.nodes.index(node)
            graph.del_node(node)
            graph.insert_nodes(new_nodes, pos)

        graph.update_topology()
        graph.update_tensor_shape()


class SplitCaffeExp(Pass):
    """This pass convert CaffeExp node to Mul,Add and a Pow. The calculation is Pow(base,Add(shift,Mul(scale,x)))"""

    def run(self, graph):
        from ..ops.op import CaffeExp
        import numpy as np
        import math

        for node in list(graph.nodes):
            if isinstance(node, CaffeExp):
                base = node.get_attribute_value("base")
                scale = node.get_attribute_value("scale")
                shift = node.get_attribute_value("shift")
                # create nodes.
                from ..ops.op import Mul, Add, Pow

                ret = []

                if math.isclose(scale, 1):
                    mul_out_name = node.input[0]
                else:
                    scale_const = Constant.make_constant(
                        f"{node.name}{DELIM}scale_const",
                        np.array(scale, dtype="float32"),
                    )
                    mul_out_name = f"{node.name}{DELIM}mul_after"
                    mul_op = Mul.make_mul(
                        f"{node.name}{DELIM}mul",
                        node.input[0],
                        scale_const.output[0],
                        mul_out_name,
                    )
                    ret.extend([scale_const, mul_op])

                if math.isclose(shift, 0):
                    add_out_name = mul_out_name
                else:
                    shift_const = Constant.make_constant(
                        f"{node.name}{DELIM}shift_const",
                        np.array(shift, dtype="float32"),
                    )
                    add_out_name = f"{node.name}{DELIM}add_after"
                    add_op = Add.make_add(
                        f"{node.name}{DELIM}add",
                        mul_out_name,
                        shift_const.output[0],
                        add_out_name,
                    )
                    ret.extend([shift_const, add_op])

                if math.isclose(base, -1):
                    # by caffe Exp layer design, when base==-1, interpret it as base=e
                    pow_out_name = node.output[0]
                    exp_op = Op.make_op(
                        "Exp", f"{node.name}{DELIM}exp", [add_out_name], [pow_out_name]
                    )
                    ret.append(exp_op)
                else:
                    base_const = Constant.make_constant(
                        f"{node.name}{DELIM}base_const", np.array(base, dtype="float32")
                    )
                    pow_out_name = node.output[0]
                    pow_op = Pow.make_pow(
                        f"{node.name}{DELIM}pow",
                        base_const.output[0],
                        add_out_name,
                        pow_out_name,
                    )
                    ret.extend([base_const, pow_op])

                pos = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_nodes(ret, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class SplitPixelShuffle(Pass):
    """
    This pass convert DepthToSpace or SpaceToDepth ops to reshape-transpose-reshape sequence.
    """

    def run(self, graph):
        from ..ops.op import Reshape, Transpose, DepthToSpace, SpaceToDepth

        warned = False

        for node in list(graph.nodes):
            if isinstance(node, DepthToSpace):
                if not warned:
                    logger.warning(
                        (
                            " using <SplitPixelShuffle> pass to split PixelShuffle to reshape-transpose-reshape sequence"
                        )
                    )
                    warned = True

                ops = []
                data = node.input[0]
                n, c, h, w = graph.get_tensor_shape(data)
                blocksize = node.get_attribute_value("blocksize")
                blocksize_squared = blocksize * blocksize
                mode = node.get_attribute_value("mode")
                if mode == "DCR":
                    shape1 = [n, blocksize, blocksize, c // blocksize_squared, h, w]
                    permute = [0, 3, 4, 1, 5, 2]
                elif mode == "CRD":
                    shape1 = [n, c // blocksize_squared, blocksize, blocksize, h, w]
                    permute = [0, 1, 4, 2, 5, 3]
                else:
                    raise NotImplementedError(
                        "DepthToSpace do not support mode %s" % mode
                    )
                shape2 = [n, c // blocksize_squared, h * blocksize, w * blocksize]

                # First, reshape to split the channels dim from c into 3 separate dims: (c // blocksize_squared,
                # blocksize, blocksize). This allows shuffling to be done next by permuting dims.
                ops.extend(
                    Reshape.make_reshape(
                        f"{node.name}{DELIM}reshape1", data, f"{data}_splited", shape1
                    )
                )
                reshape1_op = ops[-1]

                # Next, shuffle by permuting the new block_size dims alongside the height and width dims.
                transpose_op = Transpose.make_transpose(
                    f"{node.name}{DELIM}transpose",
                    reshape1_op.output[0],
                    f"{data}_transposed",
                    permute,
                )
                ops.append(transpose_op)

                # Finally, upscale by collapsing (h, blocksize) -> a single dim
                ops.extend(
                    Reshape.make_reshape(
                        f"{node.name}{DELIM}reshape2",
                        transpose_op.output[0],
                        node.output[0],
                        shape2,
                    )
                )
                pos = graph.nodes.index(node)
                # since this pass don't use any topology, we can use purely APIs of Graph.
                graph.del_node_purely(node)
                graph.insert_nodes_purely(ops, pos)

            if isinstance(node, SpaceToDepth):
                if not warned:
                    logger.warning(
                        (
                            " using <SplitPixelShuffle> pass to split PixelShuffle to reshape-transpose-reshape sequence"
                        )
                    )
                    warned = True

                ops = []
                data = node.input[0]
                n, c, h, w = graph.get_tensor_shape(data)
                blocksize = node.get_attribute_value("blocksize")
                blocksize_squared = blocksize * blocksize

                # First, reshape to split height dim into (h // blocksize, blocksize) dims and
                # width dim into (w // blocksize, blocksize) dims. This allows unshuffling to be
                # done next by permuting dims.
                shape = [n, c, h // blocksize, blocksize, w // blocksize, blocksize]
                ops.extend(
                    Reshape.make_reshape(
                        f"{node.name}{DELIM}reshape1", data, f"{data}_splited", shape
                    )
                )
                reshape1_op = ops[-1]

                # Next, unshuffle by permuting the downscale_factor dims alongside the channel dim.
                permute = [0, 1, 3, 5, 2, 4]
                transpose_op = Transpose.make_transpose(
                    f"{node.name}{DELIM}transpose",
                    reshape1_op.output[0],
                    f"{data}_transposed",
                    permute,
                )
                ops.append(transpose_op)

                # Finally, downscale by collapsing (c, blocksize, blocksize) -> a single dim
                shape = [n, c * blocksize_squared, h // blocksize, w // blocksize]
                ops.extend(
                    Reshape.make_reshape(
                        f"{node.name}{DELIM}reshape2",
                        transpose_op.output[0],
                        node.output[0],
                        shape,
                    )
                )
                pos = graph.nodes.index(node)
                # since this pass don't use any topology, we can use purely APIs of Graph.
                graph.del_node_purely(node)
                graph.insert_nodes_purely(ops, pos)

        graph.update_topology()
        graph.update_tensor_shape()
