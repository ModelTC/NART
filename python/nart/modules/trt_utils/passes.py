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

from ...passes import Pass


class CollectExtraOutput(Pass):
    """Run a collection pass on graph to collect extra outputs needed to bypass the element-wise result error of tensorrt 7.0"""

    def __init__(self):
        self.extra_outputs = set()

    def run(self, graph):
        from ...ops.op import Relu, Conv, MaxPool, AveragePool, Add, Sub, Mul

        def check_and_collect(op1, op2):
            """Given two operands of a elementwise add node, detect if any output in **THE BRANCH OF SECOND OPERAND** should be marked as
            output of the network to avoid bug of tensorrt op fusion, and if any, they will be added into self.extra_outputs.

            Normally, this method has to be called twice with exchanged operands.
            """
            """ the graph structure is like:
                    A
                   / \
                  .   .
                  |   |
                  .   C
                  |   |
            (op1) B   D (op2)
                   \ /
                    E <+>
                if E is elementwise add, D is relu, C is conv, B and D has a common ancestor A, then output of D should be marked as output.
            """
            D = graph.get_tensor_producer(op2)[0]
            if not isinstance(D, Relu):
                # if D is not relu, no risk
                return
            C = graph.get_tensor_producer(D.input[0])[0]
            if not isinstance(C, Conv):
                # if C is not Conv, no risk
                return
            B = graph.get_tensor_producer(op1)[0]

            self.extra_outputs.add(op2)

        for node in list(graph.nodes):
            if isinstance(node, (Add,)):
                # suppose only add will cause error?
                check_and_collect(node.input[0], node.input[1])
                check_and_collect(node.input[1], node.input[0])
