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

""" Passes that are hard to category.
"""
from ..core.graph import Graph
from ..ops import Reshape
from . import Pass


class LiftConstantTensors(Pass):
    """Some backends don't support constant tensor, but models may have this kind of pattern,
    for example, the `Add` op of Ascend310 requires both inputs to be tensor,
    which means `Add(x, Constant([1.0]))` is not valid.

    This pass is used to detect constant tensors operands of those operators which requires operands
    to be tensors rather than constants.
    """

    def __init__(self, pred):
        """Args:
        pred ((core.Node node, int i) -> bool): Is the i-th input of `node` allowed to be constant.
        """
        super().__init__()
        self.pred_ = pred
        self.collected_tensors = dict()

    def run(self, graph):
        import logging

        logger = logging.getLogger("nart.passes.LiftConstantTensors")

        for node in list(graph.nodes):
            for idx, ipt in enumerate(node.input):
                if graph.is_const_tensor(ipt) and not self.pred_(node, idx):
                    logger.debug(
                        f"detected constant input of node {node.name}, "
                        "which is not supported, lifting to nart tensor"
                    )
                    new_ipt = ipt + "_const"
                    if new_ipt not in self.collected_tensors:
                        # add new input to graph, and save it to collected_tensors
                        value = graph.get_const_tensor_as_array(ipt, allow_none=False)
                        if len(value.shape) == 0:
                            # nart doesnot support scalar, so reshape all scalar to tensor with shape [1].
                            value = value.reshape([1])
                        self.collected_tensors[new_ipt] = value
                        graph.add_input(new_ipt)
                        graph.set_tensor_shape(new_ipt, list(value.shape))
                    node.replace_input(ipt, new_ipt)
        graph.update_tensor_shape()


class UnsqueezeOutputTo4D(Pass):
    """In some sdk (senseauto-inference, etc), the model output is always expected to be 4D,
    which sometimes conflict with original model's semantic, this pass is used to append
    reshapes to make all output 4D.
    """

    def __init__(self):
        super().__init__()

    def run(self, graph: Graph):
        import logging

        logger = logging.getLogger("nart.passes.UnsqueezeOutputTo4D")
        from ..core import Node

        for output in graph.output:
            shape = graph.get_tensor_shape(output)
            if len(shape) >= 4:
                # if output shape is greater than or equal 4D, skip.
                continue
            producer = graph.get_tensor_producer(output)
            assert (
                len(producer) == 1
            ), f"bad graph topology, {output} has {len(producer)} outputs"
            producer = producer[0]
            assert isinstance(producer, Node)
            output_ver1 = f"{output}_ver1"
            producer.replace_output(output, output_ver1)
            # modify all consumers of `output` to use `output_ver1` instead.
            for consumer in graph.get_tensor_consumer(output):
                if isinstance(consumer, str) and consumer == "output":
                    # skip the graph output ifself
                    continue
                consumer.replace_input(output, output_ver1)
            # add reshape layer
            reshape = Reshape.make_reshape(
                f"unsqueeze_{output}", output_ver1, output, [0, 0, 1, 1]
            )
            graph.insert_nodes(reshape)


class TrimForCalibration(Pass):
    """Trim graph from the end of graph before calibration.
    Because some post process op is not implemented on CPU/GPU, the calibration process will encounter error,
    this can be avoided by safely remove those ops (they don't need calibration themself).

    Args:
        need_quant: a callable object which predicts whether an Op requires quantization.
    """

    def __init__(self, need_quant):
        self._need_quant = need_quant

    def run(self, graph):
        to_remove = set()

        def dfs(graph, node):
            """This method is used to collect nodes that should be removed."""
            if node == "input":
                # since producer "Input" is a string rather than node, special handling is required.
                return
            if self._need_quant(node):
                # this node requires quant, so itself and any node ahead of it must be kept.
                return
            to_remove.add(node)
            for ipt in node.input:
                producers = graph.get_tensor_producer(ipt)
                assert len(producers) == 1, "bad topology"
                dfs(graph, producers[0])

        # first do dfs from all outputs, to collect all nodes that can be removed.
        # a node can be removed if none of the nodes on the path from itself to output requires calibration.
        for tensor in graph.output:
            producers = graph.get_tensor_producer(tensor)
            assert len(producers) == 1, "bad topology"
            dfs(graph, producers[0])

        for node in to_remove:
            graph.del_node_purely(node)
        # then rebuild outputs
        access_count = dict()
        for node in graph.nodes:
            for ipt in node.input:
                access_count.setdefault(ipt, 0)
                access_count[ipt] += 1
            for output in node.output:
                access_count[output] = 0
        outputs = [x for x in access_count.keys() if access_count[x] == 0]
        graph.output[:] = outputs
        graph.update_topology()
