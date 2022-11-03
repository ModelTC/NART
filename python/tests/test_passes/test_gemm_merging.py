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

from .pass_test_unit import PassTestUnit, random_constant
import numpy as np
from nart.core import Graph
from nart.ops import Op, Constant

import logging

logger = logging.getLogger("nart.unittest.test_passes")


class GemmMergingPassUnitTest(PassTestUnit):
    def test_merge_gemm_batchnormalization(self):
        # TODO(hujian): art's ip op only support transB == True, so add the test where transB == False later.
        N, K, M = 16, 32, 16
        channel = M
        nodes = [
            random_constant("b", [M, K]),
            random_constant("c", [M]),
            Op.make_op("Gemm", "gemm", ["0", "b", "c"], ["1"], {"transB": 1}),
            random_constant("scale", [channel]),
            random_constant("bias", [channel]),
            random_constant("mean", [channel]),
            Constant.make_constant(
                "var", np.maximum(np.random.rand(channel).astype(np.float32), 1e-3)
            ),
            # random_constant("var", [channel]),
            Op.make_op(
                "BatchNormalization",
                "batchnorm",
                ["1", "scale", "bias", "mean", "var"],
                ["2"],
                {},
            ),
        ]
        graph_a = Graph.make_graph("graph_a", {}, nodes, {"0": [N, K]}, ["2"])
        graph_a.update_topology()
        graph_a.update_tensor_shape()
        from copy import deepcopy

        graph_b = deepcopy(graph_a)
        graph_b._name = "graph_b"

        from nart.passes import GemmFuser, DeadCodeElimination

        GemmFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        self.compare(graph_a, graph_b, atol=1e-5)

    def test_merge_gemm_add(self):
        N, K, M = 16, 32, 16
        channel = M
        nodes = [
            random_constant("b", [M, K]),
            random_constant("c", [M]),
            Op.make_op("Gemm", "gemm", ["0", "b", "c"], ["1"], {"transB": 1}),
            random_constant("bias", [channel]),
            Op.make_op("Add", "add", ["1", "bias"], ["2"]),
        ]
        graph_a, graph_b = self.make_graph(nodes, {"0": [N, K]}, ["2"])

        from nart.passes import GemmFuser, DeadCodeElimination

        GemmFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        # logger.warning("Gemm+Add mergeing test disabled due to lack of Add support in switch, enable this test after supported")
        self.compare(graph_a, graph_b, atol=1e-5)

    def test_merge_gemm_mul(self):
        N, K, M = 16, 32, 16
        channel = M
        nodes = [
            random_constant("b", [M, K]),
            random_constant("c", [M]),
            Op.make_op("Gemm", "gemm", ["0", "b", "c"], ["1"], {"transB": 1}),
            random_constant("multiplier", [channel]),
            Op.make_op("Mul", "mul", ["1", "multiplier"], ["2"]),
        ]
        graph_a, graph_b = self.make_graph(nodes, {"0": [N, K]}, ["2"])

        from nart.passes import GemmFuser, DeadCodeElimination

        GemmFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        # logger.warning("Gemm+Mul mergeing test disabled due to lack of Mul support in switch, enable this test after supported")
        self.compare(graph_a, graph_b, atol=1e-5)
