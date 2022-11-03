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

import numpy as np
import unittest
import logging

logger = logging.getLogger("nart.test")
from nart.ops import Constant
from nart.core import Net, Graph


def random_constant(name, shape, loc=0.0, scale=1.0):
    return Constant.make_constant(
        name, np.random.normal(size=shape, loc=loc, scale=scale).astype(np.float32)
    )


def cosine_similarity(a, b):
    a = a.ravel()
    b = b.ravel()
    return np.dot(a, b) / (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))


class PassTestUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(PassTestUnit, self).__init__(*args, **kwargs)

    def compare(self, graph_a, graph_b, rtol=1e-05, atol=1e-7):
        """Compare two core.Graph by compiling to default(CPU) backend,
        and compare the result on random input.

        Args:
            graph_a(core.Graph): graph A.
            graph_b(core.Graph): graph B.
        """
        self.assertEqual(
            graph_a.input, graph_b.input, msg="inputs of two graph don't match"
        )
        self.assertEqual(
            graph_a.output, graph_b.output, msg="outputs of two graph don't match"
        )
        # create network
        net_a = Net.from_graph(graph_a, output_all=False)
        net_b = Net.from_graph(graph_b, output_all=False)
        # create random input
        inputs_a = net_a.get_input_binding()
        inputs_b = net_b.get_input_binding()
        assert set(inputs_a.keys()) == set(
            inputs_b.keys()
        ), f"networks to be compared have different inputs: {inputs_a.keys()} vs {inputs_b.keys()}"
        inputs = {x: np.random.randn(*val.shape) for x, val in inputs_a.items()}
        for name, value in inputs.items():
            np.copyto(inputs_a[name], value)
            np.copyto(inputs_b[name], value)
        # run
        net_a.forward()
        net_b.forward()
        # compare output
        outputs_a = net_a.get_output_binding()
        outputs_b = net_b.get_output_binding()
        assert set(outputs_a.keys()) == set(
            outputs_b.keys()
        ), f"networks to be compared have different outputs: {outputs_a.keys()} vs {outputs_b.keys()}"
        for name in outputs_a.keys():
            a = outputs_a[name]
            b = outputs_b[name]
            allclose = np.allclose(b, a, rtol=rtol, atol=atol)
            if not allclose:
                # print(a)
                # print(b)
                diff = np.abs(a - b)
                # print(diff)
                # print(diff > 1e-6)
                print(
                    "max(abs(a - b)) = ",
                    np.max(diff),
                    "; cosine(a, b) = ",
                    cosine_similarity(a.ravel(), b.ravel()),
                )

            self.assertTrue(allclose, msg=f"the network output `{name}` does not match")

    def make_graph(self, nodes, inputs, outputs):
        graph_a = Graph.make_graph("graph_a", {}, nodes, inputs, outputs)
        graph_a.update_topology()
        graph_a.update_tensor_shape()
        from copy import deepcopy

        graph_b = deepcopy(graph_a)
        graph_b._name = "graph_b"
        graph_b.update_topology()
        graph_b.update_tensor_shape()
        return graph_a, graph_b
