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


class ConvMergingPassUnitTest(PassTestUnit):
    def test_merge_conv_batchnormalization(self):
        N, C, H, W = 2, 8, 28, 28
        M = 32
        group = 2
        kh, kw = 3, 3
        strides = [2, 2]
        padding = [1, 1, 1, 1]

        # biased
        nodes = [
            random_constant("w", [M, C // group, kh, kw]),
            random_constant("b", [M]),
            Op.make_op(
                "Conv",
                "conv",
                ["0", "w", "b"],
                ["1"],
                {
                    "group": group,
                    "kernel_shape": [kh, kw],
                    "pads": padding,
                    "strides": strides,
                },
            ),
            random_constant("scale", [M]),
            random_constant("bias", [M]),
            random_constant("mean", [M]),
            random_constant("var", [M], loc=1.0, scale=0.25),
            Op.make_op(
                "BatchNormalization",
                "batchnorm",
                ["1", "scale", "bias", "mean", "var"],
                ["2"],
                {"epsilon": 2e-5},
            ),
        ]

        graph_a, graph_b = self.make_graph(nodes, {"0": [N, C, H, W]}, ["2"])
        from nart.passes import ConvFuser, DeadCodeElimination

        ConvFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        self.compare(graph_a, graph_b, atol=1e-5, rtol=1e-5)

        # non-biased
        nodes = [
            random_constant("w", [M, C // group, kh, kw]),
            Op.make_op(
                "Conv",
                "conv",
                ["0", "w"],
                ["1"],
                {
                    "group": group,
                    "kernel_shape": [kh, kw],
                    "pads": padding,
                    "strides": strides,
                },
            ),
            random_constant("scale", [M]),
            random_constant("bias", [M]),
            random_constant("mean", [M]),
            random_constant("var", [M], loc=1.0, scale=0.25),
            Op.make_op(
                "BatchNormalization",
                "batchnorm",
                ["1", "scale", "bias", "mean", "var"],
                ["2"],
                {"epsilon": 2e-5},
            ),
        ]

        graph_a, graph_b = self.make_graph(nodes, {"0": [N, C, H, W]}, ["2"])
        from nart.passes import ConvFuser, DeadCodeElimination

        ConvFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        self.compare(graph_a, graph_b, atol=1e-5, rtol=1e-5)

    def test_merge_conv_add(self):
        N, C, H, W = 2, 8, 28, 28
        M = 32
        group = 2
        kh, kw = 3, 3
        strides = [2, 2]
        padding = [1, 1, 1, 1]

        # biased
        nodes = [
            random_constant("w", [M, C // group, kh, kw]),
            random_constant("b", [M]),
            Op.make_op(
                "Conv",
                "conv",
                ["0", "w", "b"],
                ["1"],
                {
                    "group": group,
                    "kernel_shape": [kh, kw],
                    "pads": padding,
                    "strides": strides,
                },
            ),
            random_constant("bias", [M, 1, 1]),
            Op.make_op("Add", "add", ["1", "bias"], ["2"]),
        ]

        graph_a, graph_b = self.make_graph(nodes, {"0": [N, C, H, W]}, ["2"])
        from nart.passes import ConvFuser, DeadCodeElimination

        ConvFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        # logger.warning("Conv+Mul mergeing test disabled due to lack of Mul support in switch, enable this test after supported")
        self.compare(graph_a, graph_b, rtol=1e-6, atol=1e-5)

        # non-biased
        nodes = [
            random_constant("w", [M, C // group, kh, kw]),
            Op.make_op(
                "Conv",
                "conv",
                ["0", "w"],
                ["1"],
                {
                    "group": group,
                    "kernel_shape": [kh, kw],
                    "pads": padding,
                    "strides": strides,
                },
            ),
            random_constant("bias", [M, 1, 1]),
            Op.make_op("Add", "add", ["1", "bias"], ["2"]),
        ]

        graph_a, graph_b = self.make_graph(nodes, {"0": [N, C, H, W]}, ["2"])
        from nart.passes import ConvFuser, DeadCodeElimination

        ConvFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        # logger.warning("Conv+Mul mergeing test disabled due to lack of Mul support in switch, enable this test after supported")
        self.compare(graph_a, graph_b, rtol=1e-6, atol=1e-5)

    def test_merge_conv_mul(self):
        N, C, H, W = 2, 8, 28, 28
        M = 32
        group = 2
        kh, kw = 3, 3
        strides = [2, 2]
        padding = [1, 1, 1, 1]

        # biased
        nodes = [
            random_constant("w", [M, C // group, kh, kw]),
            random_constant("b", [M]),
            Op.make_op(
                "Conv",
                "conv",
                ["0", "w", "b"],
                ["1"],
                {
                    "group": group,
                    "kernel_shape": [kh, kw],
                    "pads": padding,
                    "strides": strides,
                },
            ),
            random_constant("multiplier", [M, 1, 1]),
            Op.make_op("Mul", "mul", ["1", "multiplier"], ["2"]),
        ]

        graph_a, graph_b = self.make_graph(nodes, {"0": [N, C, H, W]}, ["2"])
        from nart.passes import ConvFuser, DeadCodeElimination

        ConvFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        # logger.warning("Conv+Mul mergeing test disabled due to lack of Mul support in switch, enable this test after supported")
        self.compare(graph_a, graph_b, rtol=1e-6, atol=1e-5)

        # non-biased
        nodes = [
            random_constant("w", [M, C // group, kh, kw]),
            Op.make_op(
                "Conv",
                "conv",
                ["0", "w"],
                ["1"],
                {
                    "group": group,
                    "kernel_shape": [kh, kw],
                    "pads": padding,
                    "strides": strides,
                },
            ),
            random_constant("multiplier", [M, 1, 1]),
            Op.make_op("Mul", "mul", ["1", "multiplier"], ["2"]),
        ]

        graph_a, graph_b = self.make_graph(nodes, {"0": [N, C, H, W]}, ["2"])
        from nart.passes import ConvFuser, DeadCodeElimination

        ConvFuser().run(graph_b)
        DeadCodeElimination().run(graph_b)

        self.assertFalse(
            graph_a.check_similarity(graph_b),
            msg="the graph was not altered after passes",
        )
        # logger.warning("Conv+Mul mergeing test disabled due to lack of Mul support in switch, enable this test after supported")
        self.compare(graph_a, graph_b, rtol=1e-6, atol=1e-5)
