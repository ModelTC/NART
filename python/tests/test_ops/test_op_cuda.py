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

from nart.ops import Op, Constant
from nart.core import Graph, Net

import unittest
import numpy as np
import logging
import random

logger = logging.getLogger("nart.unittest.test_ops")


def cosine_similarity(a, b):
    if a.size != b.size:
        return 0
    a = a.ravel()
    b = b.ravel()
    from numpy.linalg import norm

    return np.dot(a, b) / (norm(a) * norm(b))


class OpCudaHeatmap2coordTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        pass

    def _random_heatmap2coord_with_shapes(self, input_shape):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create heatmap2coord op
        heatmap2coord_op = Op.make_op(
            "HeatMap2Coord", "heatmap2coord", ["input_0"], ["output"]
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, heatmap2coord_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net_default = Net.from_graph(graph, backend="default")
        net_default.forward()
        net_cuda = Net.from_graph(graph, backend="cuda")
        net_cuda.forward()
        # get result and check
        output_default = net_default.get_binding("output")
        output_cuda = net_cuda.get_binding("output")
        return (input_0, output_default, output_cuda)

    def test_dummy_heatmap2coord(self):
        """when we use input shape as (4, 4, 4, 4) result should be ( 4, 4 * 3)"""
        input_shape_0 = (4, 4, 4, 4)
        expected_output_shape = (4, 4 * 3)

        input_0, output_default, output_cuda = self._random_heatmap2coord_with_shapes(
            input_shape_0
        )

        self.assertTrue(
            np.allclose(output_cuda, output_default),
            msg=self._make_error_message([input_0], [output_cuda]),
        )


class OpCudaExpTest(unittest.TestCase):
    def _make_error_message(self, inputs, base, scale, shift, outputs, true_outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        msg += "base" + ":\n"
        msg += str(base)
        msg += "\n"

        msg += "scale" + ":\n"
        msg += str(scale)
        msg += "\n"

        msg += "shift" + ":\n"
        msg += str(shift)
        msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(true_outputs):
            msg += "true_output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def _test_base(self, input_const, base, scale, shift):
        """The common test function."""
        input_constant = Constant.make_constant("input", input_const)
        # create exp op
        exp = Op.make_op(
            "CaffeExp",
            "exp",
            ["input"],
            ["output"],
            {"base": base, "scale": scale, "shift": shift},
        )
        # make graph
        graph = Graph.make_graph("graph", {}, [input_constant, exp], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        true_output = base ** (input_const * scale + shift)
        self.assertTrue(
            np.allclose(output, true_output),
            msg=self._make_error_message(
                [input_const], base, scale, shift, [output], [true_output]
            ),
        )

    def test_dummy(self):
        """simplest base=0.5,scale=1.0,shift=0.0 situation"""

        # dummy test:
        input_shape = [1, 64, 4, 4]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        base = 0.5
        scale = 1
        shift = 0
        self._test_base(input_data, base, scale, shift)

    def test_no_shift(self):
        """CaffeExp when no shift exists: base=0.5,scale=1.5,shift=0.0 situation"""

        # dummy test:
        input_shape = [1, 64, 4, 4]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        base = 0.5
        scale = 1.5
        shift = 0
        self._test_base(input_data, base, scale, shift)

    def test_no_scale(self):
        """CaffeExp when no scale exists: base=0.5,scale=1.0,shift=0.7 situation"""

        # dummy test:
        input_shape = [1, 64, 4, 4]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        base = 0.5
        scale = 1
        shift = 0.7
        self._test_base(input_data, base, scale, shift)

    def test_random_data(self):
        """CaffeExp test with random base,scale and shift"""
        input_shape = [1, 64, 4, 4]
        input_data = np.random.randn(*input_shape).astype(np.float32)
        base = random.uniform(0.1, 2)
        scale = random.uniform(-1, 2)
        shift = random.uniform(0, 1)
        self._test_base(input_data, base, scale, shift)


class OpCudaAddTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        """simplest (4) + (4) situation"""

        # dummy test:
        # create input op
        input_0 = np.array([1, 2, 3, 4], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.array([5, 6, 7, 8], np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create add op
        add_op = Op.make_op("Add", "add", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, add_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, input_0 + input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_wrong_broadcast_shape(self):
        """[1,2] cannot broadcat to [1,4], so we should raise error"""
        # create input nodes with wrong broadcast shape
        input_0 = np.array([[1, 2]], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.array([[5, 6, 7, 8]], np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create add op
        add_op = Op.make_op("Add", "add", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, add_op], [], ["output"]
        )
        graph.update_topology()
        # make model
        # make net and do forward
        with self.assertRaises(ValueError):
            Net.from_graph(graph, backend="cuda")

    def _random_add_with_shapes(self, input_shape_0, input_shape_1):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.random.randint(size=input_shape_1, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create add op
        add_op = Op.make_op("Add", "add", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, add_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, input_1, output)

    def test_dummy_broadcast(self):
        """when we use input shape as (1,4) + (4,4) result should be (4,4)"""
        input_shape_0 = (1, 4)
        input_shape_1 = (4, 4)
        expected_output_shape = (4, 4)

        input_0, input_1, output = self._random_add_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0 + input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_0(self):
        """when we use input shape as (2,3,4,5) + (5,) result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4, 5)
        input_shape_1 = 5
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_add_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0 + input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_1(self):
        """when we use input shape as (4, 5) + (2,3,4,5) result should be (2,3,4,5)"""
        input_shape_0 = (4, 5)
        input_shape_1 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_add_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0 + input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_2(self):
        """when we use input shape as (1,4,5) + (2,3,1,1) result should be (2,3,4,5)"""
        input_shape_0 = (1, 4, 5)
        input_shape_1 = (2, 3, 1, 1)
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_add_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0 + input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_3(self):
        """when we use input shape as (3,4,5) + (2,1,1,1) result should be (2,3,4,5)"""
        input_shape_0 = (3, 4, 5)
        input_shape_1 = (2, 1, 1, 1)
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_add_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0 + input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )


class OpDefaultAbsTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        """simplest shape = (4) situation"""
        # dummy test:
        # create input op
        input_0 = np.array([1, 0, -1, -2], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create abs op
        abs_op = Op.make_op("Abs", "abs", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, abs_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, abs(input_0)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_wrong_input_infer(self):
        """If Abs get binary input, we should raise error"""
        # create input op
        input_0 = np.array([1, 2, 3, 4], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.array([5, 6, 7, 8], np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create abs op
        abs_op = Op.make_op("Abs", "abs", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, abs_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        # TODO: It cannot raise Error back to python currently
        # So we cannot test this case
        # net = Net.from_graph(graph, backend='cuda')

    def _random_abs_with_shapes(self, input_shape_0):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create abs op
        abs_op = Op.make_op("Abs", "abs", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, abs_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, output)

    def test_random_input_0(self):
        """when we use input shape as (2,3) result should be (2,3)"""
        input_shape_0 = (2, 3)
        expected_output_shape = (2, 3)

        input_0, output = self._random_abs_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, abs(input_0)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_1(self):
        """when we use input shape as (2,3,4,5) result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, output = self._random_abs_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, abs(input_0)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_2(self):
        """when we use very big input shape as (128,1024,1024) result should be (128,1024,1024)"""
        input_shape_0 = (128, 1024, 1024)
        expected_output_shape = (128, 1024, 1024)

        input_0, output = self._random_abs_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, abs(input_0)),
            msg=self._make_error_message([input_0], [output]),
        )


class OpDefaultLstmTest(unittest.TestCase):
    @staticmethod
    def _generate_random_inputs(
        seq_length, batch_size, hidden_size, input_size, num_directions
    ):
        """private util funciton which generate random inputs using spcific size"""
        X = np.random.random_sample((seq_length, batch_size, input_size)).astype(
            np.float32
        )
        W = np.random.random_sample(
            (num_directions, 4 * hidden_size, input_size)
        ).astype(np.float32)
        R = np.random.random_sample(
            (num_directions, 4 * hidden_size, hidden_size)
        ).astype(np.float32)
        Seq_len = np.random.randint(seq_length, size=(batch_size)).astype(np.int32)
        B = np.random.random_sample((num_directions, 8 * hidden_size)).astype(
            np.float32
        )
        return X, W, R, B, Seq_len

    @staticmethod
    def _generate_const_inputs(
        seq_length, batch_size, hidden_size, input_size, num_directions
    ):
        """private util funciton which generate random inputs using spcific size"""
        X = np.ones((seq_length, batch_size, input_size)).astype(np.float32)
        W = np.ones((num_directions, 4 * hidden_size, input_size)).astype(np.float32)
        R = np.ones((num_directions, 4 * hidden_size, hidden_size)).astype(np.float32)
        Seq_len = seq_length * np.ones((batch_size)).astype(np.int32)
        B = np.ones((num_directions, 8 * hidden_size)).astype(np.float32)
        return X, W, R, B, Seq_len

    @staticmethod
    def _fill_in_data(net, X, W, R, B=None, Seq_len=None):
        """private util function which fill in datas to lstm net"""
        inputs = net.get_input_binding()
        np.copyto(inputs["X"], X)
        np.copyto(inputs["W"], W)
        np.copyto(inputs["R"], R)

        if B is not None:
            np.copyto(inputs["B"], B)

        if Seq_len is not None:
            np.copyto(inputs["Seq_len"], Seq_len)

        return net

    @staticmethod
    def _net_lstm_with_shapes(
        seq_length,
        batch_size,
        hidden_size,
        input_size,
        direction="forward",
        seq_len=False,
        bias=True,
        attr=None,
        backend="default",
    ):
        """private util function which generate a lstm graph with specific size"""
        if direction == "forward":
            num_directions = 1
        elif direction == "reverse":
            num_directions = 1
        elif direction == "bidirectional":
            num_directions = 2

        op_inputs = ["X", "W", "R"]
        op_shapes = {
            "X": [seq_length, batch_size, input_size],
            "W": [num_directions, 4 * hidden_size, input_size],
            "R": [num_directions, 4 * hidden_size, hidden_size],
        }

        lstm_attr = {
            "hidden_size": hidden_size,
            "direction": direction,
        }
        if attr is not None:
            lstm_attr = {**lstm_attr, **attr}

        if bias:
            op_inputs.append("B")
            op_shapes["B"] = [num_directions, 8 * hidden_size]

        if seq_len:
            op_inputs.append("Seq_len")
            op_shapes["Seq_len"] = [batch_size]

        lstm_op = Op.make_op("LSTM", "lstm", op_inputs, ["Y", "Y_h", "Y_c"], lstm_attr)
        graph = Graph.make_graph("graph", {}, [lstm_op], op_inputs, ["Y", "Y_h", "Y_c"])
        graph.update_tensor_shape_with_preset(op_shapes)
        graph.update_topology()
        net = Net.from_graph(graph, backend=backend)

        return net

    def test_dummy(self):
        """simplest forward without sequence_lens
        seq_length: 10
        batch_size: 9
        hidden_size: 8
        input_size: 7
        X, W, R, B: random
        sequence_len: seq_length
        direction: forward
        """
        seq_length = 10
        batch_size = 9
        hidden_size = 8
        input_size = 7
        num_directions = 1

        X, W, R, B, _ = self._generate_random_inputs(
            seq_length, batch_size, hidden_size, input_size, num_directions
        )
        net = self._net_lstm_with_shapes(
            seq_length, batch_size, hidden_size, input_size, seq_len=True
        )
        Seq_len = seq_length * np.ones((batch_size,), dtype=np.int32)
        self._fill_in_data(net, X, W, R, B, Seq_len)
        net.forward()
        net_cuda = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            backend="cuda",
        )
        self._fill_in_data(net_cuda, X, W, R, B, Seq_len)
        net_cuda.forward()
        Y_default = net.get_binding("Y")
        cell_default = net.get_binding("Y_c")
        Y_cuda = net_cuda.get_binding("Y")
        cell_cuda = net_cuda.get_binding("Y_c")

        self.assertTrue(np.allclose(Y_cuda, Y_default))
        self.assertTrue(np.allclose(cell_cuda, cell_default))

    def test_bidirection(self):
        """bidirectional without sequence_lens
        seq_length: 10
        batch_size: 4
        hidden_size: 3
        input_size: 2
        X, W, R, B: random
        sequence_len: seq_length
        direction: bidirectional
        """
        seq_length = 10
        batch_size = 4
        hidden_size = 3
        input_size = 2
        direction = "bidirectional"
        num_directions = 2

        X, W, R, B, _ = self._generate_random_inputs(
            seq_length, batch_size, hidden_size, input_size, num_directions
        )
        Seq_len = seq_length * np.ones((batch_size,), dtype=np.int32)
        net = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            direction=direction,
        )
        self._fill_in_data(net, X, W, R, B, Seq_len)
        net.forward()
        net_cuda = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            direction=direction,
            backend="cuda",
        )
        self._fill_in_data(net_cuda, X, W, R, B, Seq_len)
        net_cuda.forward()
        Y_default = net.get_binding("Y")
        hidden_default = net.get_binding("Y_h")
        cell_default = net.get_binding("Y_c")

        Y_cuda = net_cuda.get_binding("Y")
        hidden_cuda = net_cuda.get_binding("Y_h")
        cell_cuda = net_cuda.get_binding("Y_c")

        self.assertTrue(np.allclose(Y_cuda, Y_default))
        self.assertTrue(np.allclose(hidden_cuda, hidden_default))
        self.assertTrue(np.allclose(cell_cuda, cell_default))

    def test_reverse(self):
        """reverse without sequence_lens
        seq_length: 10
        batch_size: 4
        hidden_size: 3
        input_size: 2
        X, W, R, B: random
        sequence_len: seq_length
        direction: reverse
        """
        seq_length = 10
        batch_size = 4
        hidden_size = 3
        input_size = 2
        direction = "reverse"
        num_directions = 1

        X, W, R, B, _ = self._generate_random_inputs(
            seq_length, batch_size, hidden_size, input_size, num_directions
        )
        Seq_len = seq_length * np.ones((batch_size,), dtype=np.int32)
        net = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            direction=direction,
        )
        self._fill_in_data(net, X, W, R, B, Seq_len)
        net.forward()

        net_cuda = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            direction=direction,
            backend="cuda",
        )
        self._fill_in_data(net_cuda, X, W, R, B, Seq_len)
        net_cuda.forward()

        Y_default = net.get_binding("Y")
        hidden_default = net.get_binding("Y_h")
        cell_default = net.get_binding("Y_c")

        Y_cuda = net_cuda.get_binding("Y")
        hidden_cuda = net_cuda.get_binding("Y_h")
        cell_cuda = net_cuda.get_binding("Y_c")

        self.assertTrue(np.allclose(Y_cuda, Y_default))
        self.assertTrue(np.allclose(hidden_cuda, hidden_default))
        self.assertTrue(np.allclose(cell_cuda, cell_default))

    def test_bidirectional_sequence_len(self):
        """bidirectional with sequence_len
        seq_length: 10
        batch_size: 4
        hidden_size: 3
        input_size: 2
        X, W, R, B: random
        sequence_len: random
        direction: bidirectional
        """
        seq_length = 10
        batch_size = 4
        hidden_size = 3
        input_size = 2
        direction = "bidirectional"
        num_directions = 2

        X, W, R, B, Seq_len = self._generate_random_inputs(
            seq_length, batch_size, hidden_size, input_size, num_directions
        )
        Seq_len = 3 * np.ones((batch_size,), dtype=np.int32)
        net = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            direction=direction,
        )
        self._fill_in_data(net, X, W, R, B, Seq_len)
        net.forward()

        net_cuda = self._net_lstm_with_shapes(
            seq_length,
            batch_size,
            hidden_size,
            input_size,
            seq_len=True,
            direction=direction,
            backend="cuda",
        )
        self._fill_in_data(net_cuda, X, W, R, B, Seq_len)
        net_cuda.forward()

        Y_default = net.get_binding("Y")
        hidden_default = net.get_binding("Y_h")
        cell_default = net.get_binding("Y_c")

        Y_cuda = net_cuda.get_binding("Y")
        hidden_cuda = net_cuda.get_binding("Y_h")
        cell_cuda = net_cuda.get_binding("Y_c")

        self.assertTrue(np.allclose(Y_cuda, Y_default))
        self.assertTrue(np.allclose(hidden_cuda, hidden_default))
        self.assertTrue(np.allclose(cell_cuda, cell_default))


class OpDefaultHsigmoidTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        """simplest shape = (4) situation"""
        # dummy test:
        # create input op
        input_0 = np.array([1, 0, -1, -2], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create Hsigmoid op
        hsigmoid_op = Op.make_op("Hsigmoid", "hsigmoid", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, hsigmoid_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = np.clip(input_0 * 0.2 + 0.5, 0, 1)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_alpha_change_pass(self):
        """simplest shape = (4) situation"""
        # alpha_change_pass test:
        # create input op
        input_0 = np.array([0.3, 0, -1, -2], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create op
        cons3 = Constant.make_constant("cons3", value=np.array(3).astype("float32"))
        addn = Op.make_op("Add", "addn", ["input_0", "cons3"], ["2"])
        clipn = Op.make_op(
            "Clip", "clipn", ["2"], ["3"], dict([("min", 0.0), ("max", 6.0)])
        )
        cons6 = Constant.make_constant("cons6", value=np.array(6).astype("float32"))
        divn = Op.make_op("Div", "divn", ["3", "cons6"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, cons3, addn, clipn, cons6, divn], [], ["output"]
        )
        graph.update_topology()
        from nart.utils.passes import ExtractHsigmoid

        ExtractHsigmoid().run(graph)
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = np.clip(input_0 + 3, 0, 6) / 6.0
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_dummy_with_attr(self):
        """simplest shape = (4) situation"""
        # dummy test:
        # create input op
        input_0 = np.array([1, 0, -1, -2], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create hsigmoid op
        alpha, beta = 0.3, 0.5
        hsigmoid_op = Op.make_op(
            "Hsigmoid",
            "hsigmoid",
            ["input_0"],
            ["output"],
            attributes={"alpha": alpha, "beta": beta},
        )
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, hsigmoid_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = np.clip(input_0 * alpha + beta, 0, 1)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_wrong_input_infer(self):
        """If Hsigmoid get binary input, we should raise error"""
        # create input op
        input_0 = np.array([1, 2, 3, 4], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.array([5, 6, 7, 8], np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create Hsigmoid op
        hsigmoid_op = Op.make_op("Hsigmoid", "hsigmoid", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, hsigmoid_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        # TODO: It cannot raise Error back to python currently
        # So we cannot test this case
        # net = Net.from_graph(graph)

    def _random_hsigmoid_with_shapes(self, input_shape_0):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create Hsigmoid op
        hsigmoid_op = Op.make_op("Hsigmoid", "hsigmoid", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, hsigmoid_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, output)

    def test_random_input_0(self):
        """when we use input shape as (2,3) result should be (2,3)"""
        input_shape_0 = (2, 3)
        expected_output_shape = (2, 3)

        input_0, output = self._random_hsigmoid_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, np.clip(input_0 * 0.2 + 0.5, 0, 1)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_1(self):
        """when we use input shape as (2,3,4,5) result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, output = self._random_hsigmoid_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, np.clip(input_0 * 0.2 + 0.5, 0, 1)),
            msg=self._make_error_message([input_0], [output]),
        )


class OpDefaultTransposeTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        """simplest (2, 3, 4, 5).transpose(1, 0, 3, 2) situation"""

        # dummy test:
        # create input op
        input_shape = (2, 3, 4, 5)
        transpose = (1, 0, 3, 2)

        transpose_attr = {
            "perm": transpose,
        }
        input_0 = np.random.random_sample(input_shape).astype(np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create add op
        transpose_op = Op.make_op(
            "Transpose", "transpose", ["input_0"], ["output"], transpose_attr
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, transpose_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, input_0.transpose(transpose)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_dummy_2(self):
        """simplest (2, 3, 4, 5).transpose(0, 1, 3, 2) situation"""

        # dummy test:
        # create input op
        input_shape = (2, 3, 4, 5)
        transpose = (0, 1, 3, 2)

        transpose_attr = {
            "perm": transpose,
        }
        input_0 = np.random.random_sample(input_shape).astype(np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create add op
        transpose_op = Op.make_op(
            "Transpose", "transpose", ["input_0"], ["output"], transpose_attr
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, transpose_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, input_0.transpose(transpose)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_shape_1(self):
        """(256, 64, 4, 4).transpose(1, 0, 2, 3) situation"""

        # dummy test:
        # create input op
        input_shape = (256, 64, 4, 4)
        transpose = (1, 0, 2, 3)

        transpose_attr = {
            "perm": transpose,
        }
        input_0 = np.random.random_sample(input_shape).astype(np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create add op
        transpose_op = Op.make_op(
            "Transpose", "transpose", ["input_0"], ["output"], transpose_attr
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, transpose_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, input_0.transpose(transpose)),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_shape_2(self):
        """(256, 64, 4, 4).transpose(0, 2, 1, 3) situation"""

        # dummy test:
        # create input op
        input_shape = (256, 64, 4, 4)
        transpose = (0, 2, 1, 3)

        transpose_attr = {
            "perm": transpose,
        }
        input_0 = np.random.random_sample(input_shape).astype(np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create add op
        transpose_op = Op.make_op(
            "Transpose", "transpose", ["input_0"], ["output"], transpose_attr
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, transpose_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, input_0.transpose(transpose)),
            msg=self._make_error_message([input_0], [output]),
        )


class OpDefaultPowTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        """simplest (4) + (4) situation"""

        # dummy test:
        # create input op
        input_0 = np.array([1, 2, 3], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.array([1, 2, 3], np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create pow op
        pow_op = Op.make_op("Pow", "pow", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, pow_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        self.assertTrue(
            np.allclose(output, input_0**input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_wrong_broadcast_shape(self):
        """[1,2] cannot broadcat to [1,4], so we should raise error"""
        # create input nodes with wrong broadcast shape
        input_0 = np.array([[1, 2]], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.array([[5, 6, 7, 8]], np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create pow op
        pow_op = Op.make_op("Pow", "pow", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, pow_op], [], ["output"]
        )
        graph.update_topology()
        # make model
        # make net and do forward
        with self.assertRaises(ValueError):
            Net.from_graph(graph)

    def _random_pow_with_shapes(self, input_shape_0, input_shape_1):
        """An utils function which generate random input and return output"""
        # create input nodes
        input_0 = (
            np.random.randint(size=input_shape_0, low=1, high=1024).astype(np.float32)
            / 256
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_1 = np.random.randn(*input_shape_1).astype(np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create pow op
        pow_op = Op.make_op("Pow", "pow", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, pow_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, input_1, output)

    def test_broadcast_input_shape_0(self):
        """when we use input shape as (2,3,4,5) + (5,) result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4)
        input_shape_1 = (4,)
        expected_output_shape = (2, 3, 4)

        input_0, input_1, output = self._random_pow_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0**input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_1(self):
        """when we use input shape as (4, 5) + (2,3,4,5) result should be (2,3,4,5)"""
        input_shape_0 = (4, 5)
        input_shape_1 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_pow_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0**input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_2(self):
        """when we use input shape as (1,4,5) + (2,3,1,1) result should be (2,3,4,5)"""
        input_shape_0 = (1, 4, 5)
        input_shape_1 = (2, 3, 1, 1)
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_pow_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0**input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_3(self):
        """when we use input shape as (3,4,5) + (2,1,1,1) result should be (2,3,4,5)"""
        input_shape_0 = (3, 4, 5)
        input_shape_1 = (2, 1, 1, 1)
        expected_output_shape = (2, 3, 4, 5)

        input_0, input_1, output = self._random_pow_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0**input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )

    def test_broadcast_input_shape_4(self):
        """when we use input shape as (1, ) and (1, 64, 4, 4) result should be (1, 64, 4, 4)"""
        input_shape_0 = (1,)
        input_shape_1 = (1, 64, 4, 4)
        expected_output_shape = (1, 64, 4, 4)

        input_0, input_1, output = self._random_pow_with_shapes(
            input_shape_0, input_shape_1
        )

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        self.assertTrue(
            np.allclose(output, input_0**input_1),
            msg=self._make_error_message([input_0, input_1], [output]),
        )


class OpDefaultSigmoidTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def test_dummy(self):
        """simplest shape = (4) situation"""
        # dummy test:
        # create input op
        input_0 = np.array([1, 0, -1, -2], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create Sigmoid op
        sigmoid_op = Op.make_op("Sigmoid", "sigmoid", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, sigmoid_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = 1 / (1 + np.exp(-input_0))
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def _random_sigmoid_with_shapes(self, input_shape_0):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create sigmoid op
        sigmoid_op = Op.make_op("Sigmoid", "sigmoid", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, sigmoid_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, output)

    def _sigmoid(self, x):
        x_ravel = x.ravel()  # 将numpy数组展平
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-x_ravel[index])))
            else:
                y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
        return np.array(y).reshape(x.shape)

    def test_random_input_0(self):
        """when we use input shape as (2,3) result should be (2,3)"""
        input_shape_0 = (2, 3)
        expected_output_shape = (2, 3)

        input_0, output = self._random_sigmoid_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        numpy_output = self._sigmoid(input_0)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_1(self):
        """when we use input shape as (2,3,4,5) result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, output = self._random_sigmoid_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        numpy_output = self._sigmoid(input_0)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )


class OpDefaultSoftmaxTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def _softmax(self, x, axis):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def test_dummy(self):
        """simplest shape = (4) situation"""
        # dummy test:
        # create input op
        input_0 = np.array([-1, 0, 1], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create Softmax op
        softmax_op = Op.make_op(
            "Softmax", "softmax", ["input_0"], ["output"], attributes={"axis": 0}
        )
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, softmax_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = self._softmax(input_0, 0)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def _random_softmax_with_shapes_and_axis(self, input_shape_0, axis):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create softmax op
        softmax_op = Op.make_op(
            "Softmax", "softmax", ["input_0"], ["output"], attributes={"axis": axis}
        )
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, softmax_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, output)

    def test_random_input_0(self):
        """when we use input shape as (2,3) and axis as 1 result should be (2,3)"""
        input_shape_0 = (2, 3)
        expected_output_shape = (2, 3)

        input_0, output = self._random_softmax_with_shapes_and_axis(input_shape_0, 1)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        numpy_output = self._softmax(input_0, 1)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_1(self):
        """when we use input shape as (2,3,4,5) and axis as 0 result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, output = self._random_softmax_with_shapes_and_axis(input_shape_0, 3)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )
        # softmax axis must be the last axis
        numpy_output = self._softmax(input_0, 3)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )


class OpDefaultConvTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        for n, i in enumerate(outputs):
            msg += "output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    # only support
    def _numpy_conv(self, inputs, filter, stride, dilation, _result):
        H, W = inputs.shape
        kH, kW = filter.shape

        result = np.zeros((_result.shape))
        kH_ = (kH - 1) * dilation[0] + 1
        kW_ = (kW - 1) * dilation[1] + 1

        use_input = np.zeros([kH, kW], np.float32)
        for r in range(0, result.shape[0]):
            for c in range(0, result.shape[1]):
                cur_input = inputs[
                    r * stride[0] : r * stride[0] + kH_,
                    c * stride[1] : c * stride[1] + kW_,
                ]
                for i in range(0, kH):
                    for j in range(0, kW):
                        use_input[i][j] = cur_input[i * dilation[0]][j * dilation[1]]

                cur_output = use_input * filter
                conv_sum = np.sum(cur_output)
                result[r, c] = conv_sum
        return result

    def _conv(
        self, inputs, filter, strides=[1, 1], padding_=[0, 0, 0, 0], dilation=[1, 1]
    ):
        inputs = np.pad(
            inputs,
            ((0, 0), (0, 0), (padding_[0], padding_[2]), (padding_[1], padding_[3])),
            "constant",
            constant_values=(0, 0),
        )

        N, C_in, H, W = inputs.shape
        C_out, C_in, kH, kW = filter.shape
        kH_ = (kH - 1) * dilation[0] + 1
        kW_ = (kW - 1) * dilation[1] + 1
        result = np.zeros(
            [
                N,
                C_out,
                int(np.floor((H - kH_) / strides[0]) + 1),
                int(np.floor((W - kW_) / strides[1]) + 1),
            ],
            np.float32,
        )

        for batch_size in range(N):
            for channel_out in range(C_out):
                for channel_in in range(C_in):
                    channel_data = inputs[batch_size][channel_in]
                    result[batch_size, channel_out, :, :] += self._numpy_conv(
                        channel_data,
                        filter[channel_out][channel_in],
                        strides,
                        dilation,
                        result[0][0],
                    )

        return result

    def test_dummy(self):
        """simplest shape = (1, 1, 5, 5) situation"""
        # dummy test:
        # create input op
        input_0 = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)

        input_1 = np.array(
            [
                [
                    [
                        [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ]
            ]
        ).astype(np.float32)
        input_op_1 = Constant.make_constant("input_1", input_1)

        # create Conv op
        conv_op = Op.make_op(
            "Conv",
            "conv",
            ["input_0", "input_1"],
            ["output"],
            attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, conv_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = self._conv(
            input_0, input_1, strides=[1, 1], padding_=[1, 1, 1, 1]
        )
        self.assertEqual(np.shape(output), np.shape(numpy_output), msg="Wrong shape!")
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_with_stride(self):
        """when we use input shape as (2, 3, 7 ,5) and kernel(3, 3, 3, 3)"""
        input_shape_0 = (2, 3, 7, 5)
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)

        input_shape_1 = (3, 3, 3, 3)
        input_1 = np.random.randint(size=input_shape_1, low=-10, high=10).astype(
            np.float32
        )
        input_op_1 = Constant.make_constant("input_1", input_1)

        conv_op = Op.make_op(
            "Conv",
            "conv",
            ["input_0", "input_1"],
            ["output"],
            attributes={
                "kernel_shape": [3, 3],
                "pads": [1, 1, 1, 1],
                "strides": [2, 2],
            },
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, conv_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()

        # get result and check
        output = net.get_binding("output")
        # print(output)
        numpy_output = self._conv(
            input_0, input_1, strides=[2, 2], padding_=[1, 1, 1, 1]
        )
        self.assertEqual(np.shape(output), np.shape(numpy_output), msg="Wrong shape!")
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_with_delation(self):
        """when we use input shape as (1, 3, 10 ,8) and kernel(3, 3, 3, 3)"""
        input_shape_0 = (1, 3, 10, 8)
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)

        input_shape_1 = (3, 3, 3, 3)
        input_1 = np.random.randint(size=input_shape_1, low=-10, high=10).astype(
            np.float32
        )
        input_op_1 = Constant.make_constant("input_1", input_1)

        conv_op = Op.make_op(
            "Conv",
            "conv",
            ["input_0", "input_1"],
            ["output"],
            attributes={
                "kernel_shape": [3, 3],
                "pads": [1, 1, 1, 1],
                "strides": [2, 2],
                "dilations": [2, 2],
            },
        )
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, conv_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()

        # get result and check
        output = net.get_binding("output")
        # print(output)
        numpy_output = self._conv(
            input_0, input_1, strides=[2, 2], padding_=[1, 1, 1, 1], dilation=[2, 2]
        )
        self.assertEqual(np.shape(output), np.shape(numpy_output), msg="Wrong shape!")
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )


class OpCudaErfTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs, true_outputs):
        msg = ""
        for n, i in enumerate(true_outputs):
            msg += "true_output_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"

        return msg

    def _test_base(self, input_const):
        """The common test function."""
        input_constant = Constant.make_constant("input", input_const)
        # create exp op
        erf = Op.make_op("Erf", "erf", ["input"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_constant, erf], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        from scipy import special

        true_output = special.erf(input_const)
        self.assertTrue(
            np.allclose(output, true_output),
            msg=self._make_error_message([input_const], [output], [true_output]),
        )

    def test_dummy(self):
        data = np.random.randn(2, 32, 8, 8).astype(np.float32)
        self._test_base(data)


class OpDefaultHswishTest(unittest.TestCase):
    def _make_error_message(self, inputs, outputs):
        """utils function which generate text including inputs and outputs"""
        msg = ""
        for n, i in enumerate(inputs):
            msg += "input_" + str(n) + ":\n"
            msg += str(i)
            msg += "\n"
        return msg

    def _hswish(self, x):
        return np.minimum((x + 3 > 0) * (x + 3), 6) * x / 6

    def test_dummy(self):
        # dummy test:
        # create input op
        # input_0 = np.array([1, 2, 3], np.float32)
        input_0 = np.array([1, 2, 3], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create pow op
        hswish_op = Op.make_op("Hswish", "hswish", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, hswish_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph)
        net.forward()
        # get result and check
        output = net.get_binding("output")
        num_output = self._hswish(input_0)
        self.assertTrue(
            np.allclose(output, num_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_pass(self):
        """simplest shape = (4) situation"""
        # pass test:
        # create input op
        input_0 = np.array([0.3, 0, -1, -2], np.float32)
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create op
        cons3 = Constant.make_constant("cons3", value=np.array(3).astype("float32"))
        addn = Op.make_op("Add", "addn", ["input_0", "cons3"], ["2"])
        clipn = Op.make_op(
            "Clip", "clipn", ["2"], ["3"], dict([("min", 0.0), ("max", 6.0)])
        )
        muln = Op.make_op("Mul", "muln", ["input_0", "3"], ["4"])
        cons6 = Constant.make_constant("cons6", value=np.array(6).astype("float32"))
        divn = Op.make_op("Div", "divn", ["4", "cons6"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph",
            {},
            [input_op_0, cons3, addn, clipn, muln, cons6, divn],
            [],
            ["output"],
        )
        graph.update_topology()
        from nart.utils.passes import ExtractHswish

        ExtractHswish().run(graph)
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph)
        net.forward()
        # get result and check
        output = net.get_binding("output")
        numpy_output = self._hswish(input_0)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def _random_hswish_with_shapes(self, input_shape_0):
        """A utils function which generate random input and return output"""
        # create input nodes
        input_0 = np.random.randint(size=input_shape_0, low=-1000, high=1000).astype(
            np.float32
        )
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create hswish op
        hswish_op = Op.make_op("Hswish", "hswish", ["input_0"], ["output"])
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, hswish_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph)
        net.forward()
        # get result and check
        output = net.get_binding("output")
        return (input_0, output)

    def test_random_input_0(self):
        """when we use input shape as (2,3) result should be (2,3)"""
        input_shape_0 = (2, 3)
        expected_output_shape = (2, 3)

        input_0, output = self._random_hswish_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        numpy_output = self._hswish(input_0)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )

    def test_random_input_1(self):
        """when we use input shape as (2,3,4,5) result should be (2,3,4,5)"""
        input_shape_0 = (2, 3, 4, 5)
        expected_output_shape = (2, 3, 4, 5)

        input_0, output = self._random_hswish_with_shapes(input_shape_0)

        self.assertEqual(
            np.shape(output), expected_output_shape, msg="Wrong broadcast shape!"
        )

        numpy_output = self._hswish(input_0)
        self.assertTrue(
            np.allclose(output, numpy_output),
            msg=self._make_error_message([input_0], [output]),
        )


class OpCudaMatmulTest(unittest.TestCase):
    def _make_error_message(self, pred, gt):
        """utils function which generate text including inputs and outputs"""
        msg = "Matmul test failed, cosine similarity = {0}, max abs diff = {1}".format(
            cosine_similarity(pred, gt), np.max(np.abs(pred - gt))
        )

        return msg

    def _test(self, input_0, input_1):
        """common test process."""
        # dummy test:
        # create input op
        input_op_0 = Constant.make_constant("input_0", input_0)
        input_op_1 = Constant.make_constant("input_1", input_1)
        # create Hsigmoid op
        hsigmoid_op = Op.make_op("MatMul", "matmul", ["input_0", "input_1"], ["output"])
        # make graph
        graph = Graph.make_graph(
            "graph", {}, [input_op_0, input_op_1, hsigmoid_op], [], ["output"]
        )
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, output_all=False, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")
        gt = np.matmul(input_0, input_1)
        self.assertTrue(
            np.allclose(output, gt, atol=1e-5), msg=self._make_error_message(output, gt)
        )

    def test_2D(self):
        """2D matmul"""
        M, K, N = 32, 128, 16
        input_0 = np.random.randn(M, K).astype(np.float32)
        input_1 = np.random.randn(K, N).astype(np.float32)
        self._test(input_0=input_0, input_1=input_1)

    def test_3D(self):
        """3D matmul"""
        M, K, N = 32, 128, 16
        input_0 = np.random.randn(8, M, K).astype(np.float32)
        input_1 = np.random.randn(1, K, N).astype(np.float32)
        self._test(input_0=input_0, input_1=input_1)

    def test_4D(self):
        """4D matmul"""
        M, K, N = 32, 128, 16
        input_0 = np.random.randn(4, 8, M, K).astype(np.float32)
        input_1 = np.random.randn(1, 8, K, N).astype(np.float32)
        self._test(input_0=input_0, input_1=input_1)

    def test_4D_broadcast(self):
        """4D matmul 2D, padding required."""
        M, K, N = 32, 128, 16
        input_0 = np.random.randn(4, 8, M, K).astype(np.float32)
        input_1 = np.random.randn(K, N).astype(np.float32)
        self._test(input_0=input_0, input_1=input_1)


class OpCudaEluTest(unittest.TestCase):
    def _make_error_message(self, pred, gt):
        """utils function which generate text including inputs and outputs"""
        msg = "Elu test failed, cosine similarity = {0}, max abs diff = {1}".format(
            cosine_similarity(pred, gt), np.max(np.abs(pred - gt))
        )

        return msg

    def _test(self, input_0, alpha):
        """common test process."""
        # dummy test:
        # create input op
        input_op_0 = Constant.make_constant("input_0", input_0)
        # create Hsigmoid op
        elu_op = Op.make_op("Elu", "elu", ["input_0"], ["output"], {"alpha": alpha})
        # make graph
        graph = Graph.make_graph("graph", {}, [input_op_0, elu_op], [], ["output"])
        graph.update_topology()
        # make net and do forward
        net = Net.from_graph(graph, backend="cuda")
        net.forward()
        # get result and check
        output = net.get_binding("output")

        gt = np.where(input_0 >= 0, input_0, alpha * (np.exp(input_0) - 1))
        self.assertTrue(
            np.allclose(output, gt, atol=1e-5), msg=self._make_error_message(output, gt)
        )

    def test_simple(self):
        """Simple Elu."""
        shape = [4, 128, 8, 16]
        alpha = 1.0
        input_0 = np.random.randn(*shape).astype(np.float32)
        self._test(input_0=input_0, alpha=alpha)

    def test_half_alpha(self):
        """Elu where alpha = 0.5"""
        shape = [4, 128, 8, 16]
        alpha = 0.5
        input_0 = np.random.randn(*shape).astype(np.float32)
        self._test(input_0=input_0, alpha=alpha)


if __name__ == "__main__":
    unittest.main()
