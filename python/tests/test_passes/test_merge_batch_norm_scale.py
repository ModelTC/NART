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


class MergeNormScaleUnitTest(PassTestUnit):
    def _common(self, with_bias):
        batch_size = 4
        channel = 32
        ops = [
            random_constant("mean", [channel]),
            # variance cannot be too small
            Constant.make_constant(
                "var", np.maximum(np.random.rand(channel).astype(np.float32), 1e-3)
            ),
            Constant.make_constant(
                "mvf",
                np.ones(
                    [
                        1,
                    ],
                    dtype=np.float32,
                ),
            ),
            # Constant.make_constant("mvf", np.array([0.9], dtype=np.float32)),
            Op.make_op(
                "CaffeBatchNorm", "a", ["0", "mean", "var", "mvf"], ["1"], {"eps": 2e-5}
            ),
            random_constant("scale", [channel]),
        ]
        if with_bias:
            ops.extend(
                [
                    random_constant("bias", [channel]),
                    Op.make_op("CaffeScale", "b", ["1", "scale", "bias"], ["2"]),
                ]
            )
        else:
            ops.extend([Op.make_op("CaffeScale", "b", ["1", "scale"], ["2"])])
        graph = Graph.make_graph(
            "graph0", {}, ops, {"0": [batch_size, channel, 1, 1]}, ["2"]
        )
        graph.update_topology()
        graph.update_tensor_shape()
        from copy import deepcopy

        graph2 = deepcopy(graph)
        graph2._name = "graph1"

        from nart.passes import MergeBatchNormScale, DeadCodeElimination

        MergeBatchNormScale().run(graph2)
        DeadCodeElimination().run(graph2)

        # ensure the graph was modified
        self.assertFalse(
            graph.check_similarity(graph2), msg="the graph was not altered after passes"
        )

        self.compare(graph, graph2, rtol=2e-05, atol=2e-5)

    def test_merge_norm_scale(self):
        self._common(True)

    # default don't support scale without bias now.
    # def test_merge_norm_scale_without_bias(self):
    #     self._common(False)
