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

from queue import Queue
import copy
from ..ops.op import Op, DELIM
import numpy as np

import logging

logger = logging.getLogger("nart.passes")


class Pass:
    def __init__(self):
        pass

    def run(self, graph):
        pass


from .split_caffe_ops import (
    SplitShuffleChannel,
    SplitPixelShuffle,
    ConvertCaffeBatchNorm,
    SplitCaffeScale,
    ConvertDualInputScale,
    SplitCaffeExp,
)
from .extract_caffe_ops import (
    ExtractEltwise,
    ExtractCaffeNormalize,
    ExtractCaffePower,
    ExtractCaffeThreshold,
    SqrtToCaffePower,
)
from .gemm_fuser import GemmFuser
from .conv_fuser import ConvFuser
from .misc import *


def get_standardize_op_passes():
    """Get list of passes to standardize ops.
    Those passes will convert Caffe mimic Ops and custom ops to standard ONNX ops.
    """
    ret = [
        MergeBatchNormScale(),
        ConvertDualInputScale(),
        ConvertCaffeBatchNorm(),
        SplitCaffeScale(),
        SplitShuffleChannel(),
        SplitCaffeExp(),
    ]
    return ret


class DeadCodeElimination(Pass):
    """This pass removes dead nodes from graph.

    A node is dead if it doesn't contribute to any output.
    """

    def run(self, graph):
        node_mark = set()
        q = Queue()
        for o in graph.output:
            pro = graph.get_tensor_producer(o)
            if len(pro) != 1:
                logger.warn(f"can not provide output {o}")
            elif pro[0] == "input":
                logger.debug(f"ignore output `{o}` produced by input")
            else:
                q.put(pro[0])
                node_mark.add(pro[0])

        while not q.empty():
            n = q.get()
            for i in n.input:
                if i == "":
                    logger.warn(f"node {n} has empty input tensor")
                    continue
                # assume: each tensor has only one producer
                pro = graph.get_tensor_producer(i)
                if pro[0] != "input" and pro[0] not in node_mark:
                    node_mark.add(pro[0])
                    q.put(pro[0])

        for n in list(graph.nodes):
            if n not in node_mark:
                graph.del_node(n)

        graph.update_topology()
        for i in list(graph.input):
            if len(graph.get_tensor_consumer(i)) == 0:
                graph.del_input(i)
        for o in list(graph.output):
            if len(graph.get_tensor_producer(o)) == 0:
                graph.del_output(o)

        graph.update_topology()
        graph.update_tensor_shape()


class ConstantToInitializer(Pass):
    """This pass detects Constant nodes, and fold its value into initializers. The original Constant node will be removed."""

    def run(self, graph):
        def const2init(node):
            assert node.op_type == "Constant"
            init = copy.deepcopy(node.attributes["value"].t)
            init.name = node.output[0]
            return init

        for n in list(graph.nodes):
            if n.op_type == "Constant":
                graph.del_node(n)
                graph.add_initializer(const2init(n))
        graph.update_topology()
        graph.update_tensor_shape()


class EliminateNodes(Pass):
    """Removes nodes which satisfies certain condition from graph.
    **NOTE**: the node to be removed can have only 1 input and 1 output.

    Args:
        pred (callable object): a predication which checkes whether a node should be removed. the signature should be: ``bool(Node, Graph)``
    """

    def __init__(self, pred):
        self.pred = pred

    def run(self, graph):
        for node in list(graph.nodes):
            if self.pred(node, graph):
                if len(node.input) != 1:
                    logger.info(
                        "Eliminating node `{0}`, which has {1} inputs, "
                        "the first one will be used to replace output".format(
                            node.name, len(node.input)
                        )
                    )
                assert len(node.output) == 1, "only 1 output nodes should be eliminated"
                # is a dropout node
                inp = node.input[0]
                output = node.output[0]
                for consumer in graph.get_tensor_consumer(output):
                    if consumer == "output":
                        graph.del_output(output)
                        if inp not in graph.output:
                            graph.add_output(inp)
                    else:
                        consumer.replace_input(output, inp)
                graph.del_node(node)
        # graph.update_topology()
        # graph.update_tensor_shape()


def RemoveCertainOp(op_types):
    def pred(node, graph):
        return node.op_type in op_types

    return EliminateNodes(pred)


class SumToAdd(Pass):
    """ONNX models can have Sum Ops, which some backends don't support.
    This pass will detect Sum node, and replace it with a series of Add nodes.
    """

    def run(self, graph):
        from ..ops.op import Sum

        def gen_add(name, a, b, output):
            op = Op.gen_op("Add", name)
            op.add_input(a)
            op.add_input(b)
            op.add_output(output)
            return op

        def create_adds_from_sum(sum):
            ops = []
            temp_out = sum.input[0]
            output = sum.output[0]
            for idx, input in enumerate(sum.input[1:]):
                if idx + 1 != len(sum.input) - 1:
                    # not the last add op
                    new_out = f"{output}_mid{idx}"
                    ops.append(
                        gen_add(f"{sum.name}{DELIM}add_{idx}", temp_out, input, new_out)
                    )
                    temp_out = new_out
                else:
                    # the last add op
                    ops.append(
                        gen_add(f"{sum.name}{DELIM}add_{idx}", temp_out, input, output)
                    )
            return ops

        for node in list(graph.nodes):
            if isinstance(node, Sum):
                pos = graph.nodes.index(node)
                graph.del_node(node)
                ops = create_adds_from_sum(node)
                # insert ops
                graph.insert_nodes(ops, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class ConvertGlobalPool(Pass):
    """Convert global pool nodes to ordinary pool, the pooling window size will be infered from input shape.

    NOTE: only valid when input shape is static.
    """

    op_type_map = {"GlobalAveragePool": "AveragePool", "GlobalMaxPool": "MaxPool"}

    def run(self, graph):
        # this pass rely on tensor shapes, so call update_tensor_shape at beginning
        graph.update_tensor_shape()

        def convert(node):
            input_shape = graph.get_tensor_shape(node.input[0])
            nb_dim = len(input_shape)
            kernel_shape = input_shape[2:]
            pads = [0] * ((nb_dim - 2) * 2)
            strides = [1] * (nb_dim - 2)
            new_name = f"{node.name}{DELIM}mod"
            attrs = {
                "kernel_shape": kernel_shape,
                "pads": pads,
                "strides": strides,
                "count_include_pad": 1,
            }

            op_type = self.op_type_map[node.op_type]
            ret = Op.make_op(
                op_type, new_name, node.input.copy(), node.output.copy(), attrs
            )

            return ret

        for node in list(graph.nodes):
            if node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
                pos = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_node(convert(node), pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


class ConvertToGlobalPool(Pass):
    """Convert ordinary pool to global pool nodes , if input_shape[-2:] == kernel_shape and pads == [0, 0, 0, 0].

    NOTE: only valid when input shape is static.
    """

    op_type_map = {"AveragePool": "GlobalAveragePool", "MaxPool": "GlobalMaxPool"}

    def run(self, graph):
        # this pass rely on tensor shapes, so call update_tensor_shape at beginning
        graph.update_tensor_shape()

        def convert(node):
            input_shape = graph.get_tensor_shape(node.input[0])
            nb_dim = len(input_shape)
            kernel_shape = node.get_attribute_value("kernel_shape")
            pads = [0] * ((nb_dim - 2) * 2)
            strides = [1] * (nb_dim - 2)
            if input_shape[-2:] == kernel_shape and pads == node.get_attribute_value(
                "pads"
            ):
                op_type = self.op_type_map[node.op_type]
                ret = Op.make_op(
                    op_type, node.name, node.input.copy(), node.output.copy()
                )
            else:
                ret = node

            return ret

        for node in list(graph.nodes):
            if node.op_type in ["MaxPool", "AveragePool"]:
                pos = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_node(convert(node), pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


class FoldConstantToAttr(Pass):
    def run(self, graph):
        from onnx import helper
        from onnx import numpy_helper

        for node in list(graph.nodes):
            if node.op_type == "Reshape":
                if len(node.input) == 1 and "shape" in node.attributes:
                    pass
                elif len(node.input) == 2:
                    if node.input[1] in graph.initializer:
                        shape = numpy_helper.to_array(graph.initializer[node.input[1]])
                    else:
                        constant_node = graph.get_tensor_producer(node.input[1])[-1]
                        shape = constant_node.to_ndarray()
                    shape = [int(i) for i in shape]
                    node.add_attribute(helper.make_attribute("shape", shape))
                    node.del_input(node.input[1])
                else:
                    raise
        graph.update_topology()
        graph.update_tensor_shape()


class SubToAdd(Pass):
    """Convert a sub node to mul-add sequence.

    This pass can be used to convert subtract operation for those backends without subtract support, or before converting to caffe's Eltwise.
    """

    def run(self, graph):
        from ..ops.op import Sub, Constant
        import numpy as np

        for node in list(graph.nodes):
            if isinstance(node, Sub):
                op1 = node.input[0]
                op2 = node.input[1]
                ndim = len(graph.get_tensor_shape(op2))
                minus1 = np.array(-1, dtype=np.float32).reshape([1] * ndim)
                minus1_node = Constant.make_constant(
                    f"{node.name}{DELIM}minus1", minus1
                )
                mul_op = Op.gen_op("Mul", f"{node.name}{DELIM}mul")
                mul_op.input.extend([op2, minus1_node.output[0]])
                mul_op.add_output(f"{op2}_mul_-1")

                add_op = Op.gen_op("Add", f"{node.name}{DELIM}add")
                add_op.input.extend([op1, mul_op.output[0]])
                add_op.add_output(node.output[0])

                idx = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_nodes([minus1_node, mul_op, add_op], idx)
        graph.update_topology()
        graph.update_tensor_shape()


class InsertUnsqueezeForBroadcastOp(Pass):
    """Insert Unsqueeze Op before BroadcastOp to ensure all operands are of same number of dimension."""

    def __init__(self, extra_op_types=set()):
        self.extra_op_types = extra_op_types

    def run(self, graph):
        from ..ops import BroadcastOp
        from ..ops.op import Unsqueeze

        for node in list(graph.nodes):
            if isinstance(node, BroadcastOp) or node.op_type in self.extra_op_types:
                # all BroadcastOp needs to be processed
                # the maximum number of dimension among all operands.
                max_ndim = 0
                for input in node.input:
                    ndim = len(graph.get_tensor_shape(input))
                    max_ndim = max(max_ndim, ndim)
                # process all operands
                for idx, input in enumerate(node.input):
                    ndim = len(graph.get_tensor_shape(input))
                    if ndim < max_ndim:
                        # if needs to be unsqueezed (expand dimension).
                        node_name = f"{node.name}{DELIM}unsqueeze{idx}"
                        out_name = f"{input}_unsqueezed_at_{node.name}_{idx}"
                        axes = range(0, max_ndim - ndim)
                        unsqueeze_node = Unsqueeze.make_unsqueeze(
                            node_name, input, out_name, axes
                        )
                        # replace original operand by unsqueezed result.
                        node.input[idx] = out_name
                        # insert the unsqueeze node
                        pos = graph.nodes.index(node)
                        graph.insert_node_purely(unsqueeze_node, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class FuseFloorDiv(Pass):
    """Detects Floor(Div(a, b)) pattern and transform them into a FloorDiv Op, used by tensorrt."""

    def run(self, graph):
        from ..ops.op import Floor, Div, FloorDiv

        for node in list(graph.nodes):
            if isinstance(node, Floor):
                producer = graph.get_tensor_producer(node.input[0])
                assert (
                    len(producer) == 1
                ), f"cannot determine producer of tensor {node.input[0]}"
                producer = producer[0]
                if isinstance(producer, Div):
                    # if producer of Floor input is Div op.
                    fused = Op.gen_op("FloorDiv", f"{node.name}{DELIM}fused_div")
                    fused.input.extend(producer.input)
                    fused.add_output(node.output[0])
                    pos = graph.nodes.index(node)
                    # replace the original floor node.
                    graph.insert_node(fused, pos)
                    graph.del_node(node)
        graph.update_topology()
        graph.update_tensor_shape()


class SoftmaxToCaffeSoftmax(Pass):
    """Convert from onnx softmax to CaffeSoftmax."""

    def run(self, graph):
        for node in list(graph.nodes):
            if node.op_type == "Softmax" and node.version == 9:
                input_shape = graph.get_tensor_shape(node.input[0])
                input_nb_dim = len(input_shape)
                axis = node.get_attribute_value("axis")
                axis = axis % input_nb_dim
                if all(x == 1 for x in input_shape[axis + 1 :]):
                    # the condition that caffe's softmax can be directly converted to onnx softmax
                    # is `all(x == 1 for x in input_shape[axis + 1:])`, because onnx's softmax-9 is done after
                    # all dimension after `aixs`.
                    if axis != input_nb_dim - 1:
                        logger.warning(
                            f"{node.name} does softmax on {node.input[0]}'s {axis}-th axis, "
                            f"the tensor shape is {input_shape}"
                        )
                    # can be directly converted to CaffeSoftmax
                    op = Op.make_op(
                        "CaffeSoftmax",
                        node.name,
                        node.input,
                        node.output,
                        {"axis": axis},
                    )
                    pos = graph.nodes.index(node)
                    graph.del_node(node)
                    graph.insert_node(op, pos)
                else:
                    # TODO: use reshape->CaffeSoftmax->reshape to do the transform.
                    # but the shape input of second reshape op can't be determined in current implementation.
                    raise NotImplementedError(
                        "cannot do the transform when softmax.axis is not the last axis"
                    )
        graph.update_topology()
        graph.update_tensor_shape()


class CollapseConsecutiveCasts(Pass):
    """This pass collapses consecutive casts into one.

    *NOTE*: When fusing two cast nodes, this pass will not eliminate the former one,
    so please call *DeadCodeElimination* pass to remove dead nodes after this pass.
    """

    def run(self, graph):
        from ..ops.op import Cast

        for node in list(graph.nodes):
            if isinstance(node, Cast):
                producer = graph.get_tensor_producer(node.input[0])
                assert len(producer) == 1, "bad graph topology"
                producer = producer[0]
                if not isinstance(producer, Cast):
                    # the producer is not Cast node, skip.
                    continue
                data = producer.input[0]
                to = node.get_attribute_value("to")
                from ..ops.op import DataType

                cast = Cast.make_cast(
                    f"{node.name}::fused", data, node.output[0], DataType(to)
                )
                # replace the original node
                pos = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_node(cast, pos)
        graph.update_topology()
        graph.update_tensor_shape()


class CanonifyNodeNames(Pass):
    """This pass can be used to rename nodes.

    Args:
        transformer(str(str)): Used to transform node name.
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def run(self, graph):
        for node in list(graph.nodes):
            node._name = self.transformer(node.name)
        graph.update_topology()
        graph.update_tensor_shape()


class ConvertDynamicUpsample(Pass):
    """This pass convert DynamicUpsample to Upsample,
    for those backends who don't support dynamic upsample.

    **NOTE**: This pass depend on the tensor shapes.
    """

    def run(self, graph):
        from ..ops.op import DynamicUpsample, Constant
        from ..utils.caffe_utils.caffe_to_onnx import adjusted_scale

        for node in list(graph.nodes):
            if not isinstance(node, DynamicUpsample):
                continue
            # a dynamic upsample node encountered
            shape = graph.get_tensor_shape(node.input[0])
            target_shape = graph.get_tensor_shape(node.input[1])
            assert len(shape) == len(
                target_shape
            ), "dynamic upsample requires two inputs have same dimension"
            scales = [adjusted_scale(t, s) for t, s in zip(target_shape[2:], shape[2:])]
            scales = [1.0, 1.0] + scales
            scales_op = Constant.make_constant(
                f"{node.name}::scales_const", np.array(scales, np.float32)
            )
            upsample_op = Op.make_op(
                "Upsample", node.name, [node.input[0], scales_op.output[0]], node.output
            )
            upsample_op.attributes.update(node.attributes)

            pos = graph.nodes.index(node)
            graph.del_node_purely(node)
            graph.insert_nodes_purely([scales_op, upsample_op], pos)
        graph.update_topology()
        graph.update_tensor_shape()


class MergeBatchNormScale(Pass):
    """This pass merge CaffeBatchNorm+CaffeScale sequence into BatchNormalize."""

    def __init__(self):
        from ..core import Graph
        from ..core.match_utils import AnyAttrValue

        batch_norm = Op.make_op(
            "CaffeBatchNorm", "batchnorm", ["0", "mean", "var", "mvf"], ["1"]
        )
        batch_norm.attributes["eps"] = AnyAttrValue("eps")
        scale = Op.make_op(
            "CaffeScale", "scale", ["1", "scale", "bias"], ["2"], {"axis": 1}
        )
        self.pattern1 = Graph.make_graph(
            "BatchNormalization module",
            {},
            [batch_norm, scale],
            ["0", "mean", "var", "mvf", "scale", "bias"],
            ["2"],
        )
        self.pattern1.update_topology()

        from copy import deepcopy

        batch_norm2 = deepcopy(batch_norm)
        scale2 = Op.make_op("CaffeScale", "scale", ["1", "scale"], ["2"], {"axis": 1})
        self.pattern2 = Graph.make_graph(
            "BatchNormalization module",
            {},
            [batch_norm2, scale2],
            ["0", "mean", "var", "mvf", "scale", "bias"],
            ["2"],
        )
        self.pattern2.update_topology()

    def run(self, graph):
        from ..ops.op import CaffeBatchNorm, CaffeScale, Constant

        def _proc(match):
            # the created nodes.
            new_nodes = []
            batch_norm = match["batchnorm"]
            mean = batch_norm.input[1]
            var = batch_norm.input[2]
            mvf = batch_norm.input[3]
            mvf_const = graph.get_const_tensor_as_array(mvf)
            import math

            if mvf_const is not None and math.isclose(mvf_const.item(0), 1.0):
                # mvf is a constant, and equals 1.0
                pass
            else:
                div1 = Op.make_op(
                    "Div",
                    f"{batch_norm.name}::div1",
                    [mean, mvf],
                    [f"{mean}_div_{mvf}"],
                )
                div2 = Op.make_op(
                    "Div", f"{batch_norm.name}::div2", [var, mvf], [f"{var}_div_{mvf}"]
                )
                mean = div1.output[0]
                var = div2.output[0]
                new_nodes.extend([div1, div2])

            scale_op = match["scale"]
            scale = scale_op.input[1]
            if len(scale_op.input) > 2:
                bias = scale_op.input[2]
            else:
                scale_shape = graph.get_tensor_shape(scale)
                assert len(scale_shape) == 1, "scale of Scale layer should be 1D tensor"
                bias_const = Constant.make_constant(
                    f"{scale_op.name}_zeros", np.zeros(scale_shape, dtype=np.float32)
                )
                bias = bias_const.output[0]
                new_nodes.append(bias_const)

            attrs = {"epsilon": batch_norm.get_attribute_value("eps", 1e-5)}

            batchnormalize = Op.make_op(
                "BatchNormalization",
                f"{batch_norm.name}_{scale_op.name}",
                [batch_norm.input[0], scale, bias, mean, var],
                [scale_op.output[0]],
                attrs,
            )
            new_nodes.append(batchnormalize)
            pos = graph.nodes.index(scale_op)
            graph.del_node(scale_op)
            graph.insert_nodes(new_nodes, pos)

        for m in graph.match(self.pattern1):
            _proc(m)
        for m in graph.match(self.pattern2):
            _proc(m)


class ReplaceRoundByFloor(Pass):
    """Some backend don't have round operator, this pass replace it by floor:
    round(a) == floor(a+0.5)
    """

    def run(self, graph):
        from ..ops.op import Constant

        const_0_5 = None
        for node in list(graph.nodes):
            if node.op_type != "Round":
                continue
            new_nodes = []
            if const_0_5 is None:
                # create a 0.5 constant
                const_0_5 = Constant.make_constant(
                    f"{node.name}_const_half", np.array([0.5], dtype=np.float32)
                )
                new_nodes.append(const_0_5)
            add = Op.make_op(
                "Add",
                f"{node.name}{DELIM}add",
                [node.input[0], const_0_5.output[0]],
                [f"{node.input[0]}_add_half"],
            )
            floor = Op.make_op(
                "Floor", f"{node.name}{DELIM}floor", [add.output[0]], [node.output[0]]
            )
            new_nodes.append(add)
            new_nodes.append(floor)
            pos = graph.nodes.index(node)
            graph.del_node_purely(node)
            graph.insert_nodes_purely(new_nodes, pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


class ConvertBGRFilterToRGB(Pass):
    """Some network is trained with BGR input tensor, but there is backend that does poorly in
    BGR image preprocssing. This pass can modify the graph, making any op that originally consumes
    BGR tensor into consume RGB tensor instead, by shuffling its filter(weight).
    """

    def __init__(self, bgr_tensors):
        self.bgr_tensors = bgr_tensors

    def run(self, graph):
        from ..ops.op import Constant

        for node in list(graph.nodes):
            if not any(x in self.bgr_tensors for x in node.input):
                # no input is bgr tensor, skip
                continue
            if node.op_type == "Conv":
                tensor = node.input[0]
                assert tensor in self.bgr_tensors
                kernel = graph.get_const_tensor_as_array(node.input[1])
                assert kernel.shape[1] == 3
                kernel_b = kernel[:, 0, :, :]
                kernel_g = kernel[:, 1, :, :]
                kernel_r = kernel[:, 2, :, :]
                kernel_rgb = np.stack([kernel_r, kernel_g, kernel_b], axis=1)
                kernel_name = f"{node.input[1]}_rgb"
                kernel_const = Constant.make_constant(kernel_name, kernel_rgb)
                pos = graph.nodes.index(node)
                graph.insert_node(kernel_const, pos)
                node.replace_input(node.input[1], kernel_name)
            else:
                raise RuntimeError(
                    f"node '{node.name}' of type '{node.op_type}' has BGR input, cannot handle"
                )


class SplitReduceL2ToGeneralOps(Pass):
    """ """

    def run(self, graph):
        for node in list(graph.nodes):
            if node.op_type == "ReduceL2":
                sqr_name = f"{node.name}{DELIM}sqr"
                sqr = Op.make_op(
                    "Mul", sqr_name, [node.input[0], node.input[0]], [sqr_name]
                )
                reduce_sum_name = f"{node.name}{DELIM}sum"
                reduce_sum = Op.make_op(
                    "ReduceSum",
                    reduce_sum_name,
                    [sqr_name],
                    [reduce_sum_name],
                    node.attributes,
                )
                sqrt_name = node.output[0]
                sqrt = Op.make_op("Sqrt", sqrt_name, [reduce_sum_name], [sqrt_name])
                pos = graph.nodes.index(node)
                graph.insert_nodes([sqr, reduce_sum, sqrt], pos)
                graph.del_node(node)
            graph.update_topology()
            graph.update_tensor_shape()


def removeInitializer(graph, node):
    # delete x_scale and x_zero_point, these two maybe reuse in other node
    scale = graph.initializer[node.input[1]]
    if len(graph.get_tensor_consumer(scale.name)) == 0:
        graph.del_initializer(scale)
    if len(node.input) == 3:
        zero_point = graph.initializer[node.input[2]]
        if len(graph.get_tensor_consumer(zero_point.name)) == 0:
            graph.del_initializer(zero_point)


class RemoveInitializerDequant(Pass):
    """
    merge initializer and dequant op into sigle node
    """

    def run(self, graph):
        from onnx import helper, TensorProto

        for node in list(graph.nodes):
            if (
                node.op_type == "DequantizeLinear"
                and node.input[0] in graph.initializer
            ):
                assert len(node.input) >= 2, "wrong DequantizeLinear op"
                x, x_scale = (
                    graph.initializer[node.input[0]],
                    graph.initializer[node.input[1]],
                )
                x_value, x_scale_value = graph.get_const_tensor_as_array(
                    x.name
                ), graph.get_const_tensor_as_array(x_scale.name)
                # optimal input
                if len(node.input) == 3:
                    x_zero_point = graph.initializer[node.input[2]]
                    x_zero_point_value = graph.get_const_tensor_as_array(
                        x_zero_point.name
                    )
                else:
                    x_zero_point_value = 0
                # OVERFLOW CAUTION
                x_value = (x_value.astype("int32") - x_zero_point_value) * x_scale_value
                # DequantizeLinear only output float tensor, cast x_value from float64 to float32
                graph.add_initializer(
                    helper.make_tensor(
                        x.name,
                        TensorProto.FLOAT,
                        x_value.shape,
                        x_value.astype("float32").tobytes(),
                        raw=True,
                    )
                )

                graph.del_node(node)
                removeInitializer(graph, node)

                # delete DequantizeLinear
                consumers = graph.get_tensor_consumer(node.output[0])
                for consumer in consumers:
                    consumer.replace_input(node.output[0], x.name)
        graph.update_topology()


class RemoveBackTOBackQuantDequant(Pass):
    """
    remove adjacent quant and dequant op
    """

    def run(self, graph):
        from onnx import helper
        from ..core import Node

        def float_equal(a, b):
            return abs(a - b) < 1e-9

        for node in list(graph.nodes):
            if node.op_type == "QuantizeLinear":
                next_nodes = graph.get_tensor_consumer(node.output[0])
                if len(next_nodes) == 1:
                    next_node = next_nodes[0]
                    # maybe output(str)
                    if (
                        isinstance(next_node, Node)
                        and next_node.op_type == "DequantizeLinear"
                    ):
                        quant_scale, dequant_scale = (
                            graph.initializer[node.input[1]],
                            graph.initializer[next_node.input[1]],
                        )
                        (
                            quant_scale_value,
                            dequant_scale_value,
                        ) = graph.get_const_tensor_as_array(
                            quant_scale.name
                        ), graph.get_const_tensor_as_array(
                            dequant_scale.name
                        )
                        if len(node.input) == 3:
                            quant_zero_point = graph.initializer[node.input[2]]
                            quant_zero_point_value = graph.get_const_tensor_as_array(
                                quant_zero_point.name
                            )
                        else:
                            quant_zero_point_value = 0

                        if len(next_node.input) == 3:
                            dequant_zero_point = graph.initializer[next_node.input[2]]
                            dequant_zero_point_value = graph.get_const_tensor_as_array(
                                dequant_zero_point.name
                            )
                        else:
                            dequant_zero_point_value = 0

                        if (
                            float_equal(quant_scale_value, dequant_scale_value)
                            and quant_zero_point_value == dequant_zero_point_value
                        ):
                            if quant_zero_point_value == 0:
                                relu_op = Op.make_op(
                                    "Relu", node.name, [node.input[0]], next_node.output
                                )
                                pos = graph.nodes.index(node)
                                graph.del_node(node)
                                removeInitializer(graph, node)
                                graph.del_node(next_node)
                                removeInitializer(graph, next_node)
                                graph.insert_node(relu_op, pos=pos)
                            else:
                                consumers = graph.get_tensor_consumer(
                                    next_node.output[0]
                                )
                                for consumer in consumers:
                                    consumer.replace_input(
                                        next_node.output[0], node.input[0]
                                    )
                                graph.del_node(node)
                                removeInitializer(graph, node)
                                graph.del_node(next_node)
                                removeInitializer(graph, next_node)


class RemoveInputDequantOutputQuant(Pass):
    """
    remove dequant after input and quant before output,
    this pass should applied after RemoveBackTOBackQuantDequant pass
    """

    def run(self, graph):
        inputs, outputs = graph.network_inputs, graph.output
        for i in inputs:
            for input_node in graph.get_tensor_consumer(i):
                if input_node.op_type == "DequantizeLinear":
                    for next_node in graph.get_tensor_consumer(input_node.output[0]):
                        next_node.replace_input(input_node.output[0], i)
                    graph.del_node(input_node)
                    removeInitializer(graph, input_node)
                elif input_node.op_type == "Transpose":
                    dequant_node = graph.get_tensor_consumer(input_node.output[0])[0]
                    if dequant_node.op_type == "DequantizeLinear":
                        for next_node in graph.get_tensor_consumer(
                            dequant_node.output[0]
                        ):
                            next_node.replace_input(
                                dequant_node.output[0], input_node.output[0]
                            )
                        graph.del_node(dequant_node)
                        removeInitializer(graph, dequant_node)
                    # remove transpose op and change input layout to NCHW
                    perm = input_node.get_attribute_value("perm")
                    if perm != [0, 3, 1, 2]:
                        logger.debug(
                            "skip process the transpose, which is not nhwc->nchw."
                        )
                        continue
                    input_shape = graph.get_tensor_shape(input_node.input[0])
                    n, h, w, c = (
                        input_shape[0],
                        input_shape[1],
                        input_shape[2],
                        input_shape[3],
                    )
                    new_shape = [n, c, h, w]
                    graph.set_tensor_shape(input_node.input[0], new_shape)
                    for next_node in graph.get_tensor_consumer(input_node.output[0]):
                        next_node.replace_input(
                            input_node.output[0], input_node.input[0]
                        )
                    graph.del_node(input_node)
        for o in outputs:
            for output_node in graph.get_tensor_producer(o):
                if output_node.op_type == "QuantizeLinear":
                    pre_out_node = graph.get_tensor_producer(output_node.input[0])[0]
                    pre_out_node.replace_output(pre_out_node.output[0], o)
                    graph.del_node(output_node)
                    removeInitializer(graph, output_node)
        graph.update_topology()


class RemoveExtraQuantDequant(Pass):
    """
    remove extra quant and dequant,
    this pass should be applied after RemoveInputDequantOutputQuant pass
    """

    def run(self, graph):
        for node in list(graph.nodes):
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                pre_node = graph.get_tensor_producer(node.input[0])
                if pre_node:
                    pre_out_node = pre_node[0]
                    next_node = graph.get_tensor_consumer(node.output[0])[0]
                    next_node.replace_input(node.output[0], pre_out_node.output[0])
                    graph.del_node(node)
                    removeInitializer(graph, node)
                else:
                    graph.del_node(node)
                    removeInitializer(graph, node)
        graph.update_topology()


from .standardize_onnx_ops import StandardizeUpsample


class ConvertGlobalPoolToReduce(Pass):
    """Convert global pool nodes to reduce op."""

    op_type_map = {"GlobalAveragePool": "ReduceMean", "GlobalMaxPool": "ReduceMax"}

    def run(self, graph):
        def convert(node):
            attrs = {
                "axes": [2, 3],
                "keepdims": 1,
            }
            op_type = self.op_type_map[node.op_type]
            ret = Op.make_op(
                op_type, node.name, node.input.copy(), node.output.copy(), attrs
            )
            return ret

        for node in list(graph.nodes):
            if node.op_type in ["GlobalAveragePool", "GlobalMaxPool"]:
                pos = graph.nodes.index(node)
                graph.del_node(node)
                graph.insert_node(convert(node), pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


class CovertReshapeToFlatten(Pass):
    """Convert Reshape nodes to flatten op."""

    def run(self, graph):
        for node in list(graph.nodes):
            if node.op_type == "Reshape":
                pos = graph.nodes.index(node)
                graph.del_node(node)
                attrs = {"axis": 1}
                flatten = Op.make_op(
                    "Flatten", node.name, [node.input[0]], node.output.copy(), attrs
                )
                graph.insert_node(flatten, pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


class AddFlattenBeforeGemm(Pass):
    """AddFlattenBeforeGemm"""

    def run(self, graph):
        for node in list(graph.nodes):
            if node.op_type == "Gemm":
                # 如果有前序节点且节点是Flatten，跳过
                pre_node = graph.get_tensor_producer(node.input[0])
                if pre_node:
                    pre_node = pre_node[0]
                    if pre_node.op_type == "Flatten":
                        continue
                # 没有前序节点/前序节点是别的，插入Flatten
                node_name = f"{node.name}{DELIM}flatten0"
                out_name = f"{pre_node.name}_flatten_at_{node_name}"
                attrs = {"axis": 1}
                flatten = Op.make_op(
                    "Flatten", node_name, [node.input[0]], [out_name], attrs
                )
                node.input[0] = out_name
                pos = graph.nodes.index(node)
                graph.insert_node_purely(flatten, pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


class SimplifyReisze(Pass):
    """simplify Resize Node."""

    def run(self, graph):
        from onnx import numpy_helper

        for node in list(graph.nodes):
            if node.op_type == "Resize":
                # roi
                if node.input[1] in graph.initializer:
                    roi = numpy_helper.to_array(graph.initializer[node.input[1]])
                else:
                    constant_node = graph.get_tensor_producer(node.input[1])[-1]
                    roi = constant_node.to_ndarray()
                if len(roi) != 0:
                    continue

                # mode
                mode = node.get_attribute_value("mode")

                # scales
                if node.input[2] in graph.initializer:
                    scales = numpy_helper.to_array(graph.initializer[node.input[2]])
                else:
                    constant_node = graph.get_tensor_producer(node.input[2])[-1]
                    scales = constant_node.to_ndarray()

                # sizes
                if node.input[3] in graph.initializer:
                    sizes = numpy_helper.to_array(graph.initializer[node.input[3]])
                else:
                    constant_node = graph.get_tensor_producer(node.input[3])[-1]
                    sizes = constant_node.to_ndarray()

                # upsample
                if len(scales) != 0:
                    upsample = Op.make_op(
                        "Upsample",
                        node.name,
                        [node.input[0], node.input[2]],
                        [node.output[0]],
                        {"mode": mode},
                    )
                elif len(sizes) != 0:
                    upsample = Op.make_op(
                        "Upsample",
                        node.name,
                        [node.input[0]],
                        [node.output[0]],
                        {"mode": mode, "height": sizes[2], "width": sizes[3]},
                    )

                pos = graph.nodes.index(node)
                graph.insert_node_purely(upsample, pos=pos)
                graph.del_node(node)
        graph.update_topology()
        graph.update_tensor_shape()


class SimplifyExpand(Pass):
    """simplify Expand Node."""

    def run(self, graph):
        from onnx import numpy_helper

        for node in list(graph.nodes):
            if node.op_type == "Expand":
                input_shape = graph.get_tensor_shape(node.input[0])
                output_shape = graph.get_tensor_shape(node.output[0])
                print(node, input_shape, output_shape)
                if input_shape != output_shape:
                    continue
                reshape = Op.make_op(
                    "Reshape",
                    node.name,
                    [node.input[0]],
                    [node.output[0]],
                    {"shape": output_shape},
                )
                pos = graph.nodes.index(node)
                graph.insert_node_purely(reshape, pos=pos)
                graph.del_node(node)
        graph.update_topology()
        graph.update_tensor_shape()


class CovertTopKToArgMax(Pass):
    """Convert TopK nodes to ArgMax op."""

    def run(self, graph):
        for node in list(graph.nodes):
            if node.op_type == "TopK":
                pos = graph.nodes.index(node)
                graph.del_node(node)
                # print(node.attributes['axis'])
                # print(node.attributes['k'])
                assert node.attributes["axis"].i == 1
                assert node.attributes["k"].i == 1
                attrs = {"axis": 1}
                argMax = Op.make_op(
                    "ArgMax", node.name, [node.input[0]], [node.output[1]], attrs
                )
                graph.insert_node(argMax, pos=pos)
        graph.update_topology()
        graph.update_tensor_shape()


from .simplify import *


class FuseAddDivClipCast(Pass):
    """Detect Cast(Clip(Div(Add(x, addend), divisor), min, max)) pattern and transform them into a AddDivClipCast Op, used with cuda."""

    def run(self, graph):
        from ..ops.op import Add, AddDivClipCast, Cast, Clip, Div
        from onnx import helper

        def check_if_only_one_producer(node):
            producer = graph.get_tensor_producer(node.input[0])
            assert (
                len(producer) == 1
            ), f"expect 1, got {len(producer)} producers for {node.input[0]}"
            return producer[0]

        for node in list(graph.nodes):
            if isinstance(node, Cast):
                producer = check_if_only_one_producer(node)
                if not isinstance(producer, Clip) or not len(producer.input) == 3:
                    continue
                clip_producer = check_if_only_one_producer(producer)
                # divisor should be `input` node
                if (
                    not isinstance(clip_producer, Div)
                    or clip_producer.input[1] not in graph.initializer
                    or len(clip_producer.input) != 2
                ):
                    continue
                div_producer = check_if_only_one_producer(clip_producer)
                # addend should be `input` node
                if (
                    not isinstance(div_producer, Add)
                    or div_producer.input[1] not in graph.initializer
                    or len(div_producer.input) != 2
                ):
                    continue
                addend_shape = graph.get_tensor_shape(div_producer.input[1])
                divisor_shape = graph.get_tensor_shape(clip_producer.input[1])
                assert (
                    addend_shape == divisor_shape
                ), f"""shape of second input of {div_producer.name} and {clip_producer.name}'
                    should be equal, but got {addend_shape} v.s. {divisor_shape}"""
                fused = Op.gen_op(
                    "AddDivClipCast", f"{node.name}{DELIM}fused_add_div_clip_cast"
                )
                cast_dtype = node.get_attribute_value("to")
                fused.attributes["to"] = helper.make_attribute("to", cast_dtype)
                fused.input.extend(div_producer.input)
                fused.input.extend(clip_producer.input[1:])
                fused.input.extend(producer.input[1:3])
                fused.add_output(node.output[0])
                pos = graph.nodes.index(node)
                # replace the original Cast node
                graph.insert_node(fused, pos)
                graph.del_node(node)
                graph.del_node(producer)
                graph.del_node(clip_producer)
                graph.del_node(div_producer)
        graph.update_topology()
        graph.update_tensor_shape()


class FuseClipCast(Pass):
    """Detect Cast(Clip(x, min, max)) pattern and transform them into a ClipCast Op, used with cuda."""

    def run(self, graph):
        from ..ops.op import Cast, Clip, ClipCast
        from onnx import helper

        for node in list(graph.nodes):
            if isinstance(node, Cast):
                producer = graph.get_tensor_producer(node.input[0])
                assert (
                    len(producer) == 1
                ), f"expect 1, got {len(producer)} producers for {node.input[0]}"
                producer = producer[0]
                # if producer of Cast input is Clip Op
                if isinstance(producer, Clip) and len(producer.input) == 3:
                    fused = Op.gen_op("ClipCast", f"{node.name}{DELIM}fused_clip_cast")
                    cast_dtype = node.get_attribute_value("to")
                    fused.attributes["to"] = helper.make_attribute("to", cast_dtype)
                    fused.input.extend(producer.input)
                    fused.add_output(node.output[0])
                    pos = graph.nodes.index(node)
                    # replace the original Clip node
                    graph.insert_node(fused, pos)
                    graph.del_node(node)
                    graph.del_node(producer)
        graph.update_topology()
        graph.update_tensor_shape()
