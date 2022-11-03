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

from onnx import helper


class AnyAttrValue(object):
    def __init__(self, name):
        self._dummy_attr = helper.make_attribute("dummy", "any_attr")
        self._name = name

    @property
    def name(self):
        return self._name

    def __getattr__(self, key, default=None):
        if key in ["type", "i", "s"]:
            return getattr(self._dummy_attr, key)
        if key in self.__dict__:
            return self.__dict__[key]
        if default:
            return default
        raise AttributeError(f"No attribute named {key} in {self.__class__.__name__}")

    def __eq__(self, other):
        # print("AnyAttrValue.__eq__ called")
        return True

    def __repr__(self):
        return "AnyAttrValue"


from .graph import NodeBase


class CustomMatcher(NodeBase):
    """A class which can be used to do custom pattern matching.

    Args:
        pred (callable object): A callable predication object whose signature is `bool(Node)`,
            which checks whether the node is matched.
    """

    def __init__(self, pred, name, inputs, outputs):
        super().__init__(name)
        self.pred = pred
        self.input[:] = inputs
        self.output[:] = outputs

    def check_similarity(self, other):
        return self.pred(other)

    def match_input(self, name, graph):
        """Will be called when matching against an input.

        Args:
            name (str): The name of input to match against.
            graph (core.Graph): The graph.
        """
        return False


class ConstantMatcher(CustomMatcher):
    """A matcher which matches constants (Constant op and initializer).

    Args:
        output (str): The output name of node, also the node name.
        pshape (list<int>): If given, will be used to check the shape of constant.
        tolerant_shape (bool): If true, fuzzy matching will be done on the shape. that is, leading ones will be ignored.
    """

    def __init__(self, output, pshape=None, tolerant_shape=False, value=None):
        if tolerant_shape:
            assert pshape is not None
            pshape = ConstantMatcher._remove_leading_ones(pshape)
        self.pshape = pshape
        self.tolerant_shape = tolerant_shape
        self.pvalue = value

        def pred(node):
            if node.op_type != "Constant":
                return False
            tensor = node.attributes["value"].t
            return self._match(tensor)

        super().__init__(pred, output, [], [output])

    def match_input(self, name, graph):
        if name in graph.network_inputs:
            # not initializer, match fail.
            return False
        tensor = graph.initializer[name]
        return self._match(tensor)

    @staticmethod
    def _remove_leading_ones(lst):
        idx = 0
        while idx < len(lst) and lst[idx] == 1:
            idx += 1
        return lst[idx:]

    def _match(self, tensor):
        if self.pshape is not None:
            shape = list(tensor.dims)
            shape = (
                ConstantMatcher._remove_leading_ones(shape)
                if self.tolerant_shape
                else shape
            )
            if shape != self.pshape:
                return False
        if self.pvalue is not None:
            from onnx import numpy_helper
            import numpy as np

            value = numpy_helper.to_array(tensor)
            if not np.allclose(value.ravel(), self.pvalue.ravel()):
                return False
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}[input={self.input}, output={self.output}, value={self.pvalue}]"


class OpMatcher(CustomMatcher):
    """A matcher matches any node whose op_type is expected.
    The dfs search process will terminate at this matcher,
    that is, it's inputs will not be matched even if given,
    the input number will be ignored either.
    """

    def __init__(self, name, op_types, inputs, outputs):
        self.op_types = op_types

        def _pred(node):
            return node.op_type in op_types

        super(OpMatcher, self).__init__(_pred, name, inputs, outputs)

    def __repr__(self):
        return f"{self.__class__.__name__}[op_types={self.op_types}, input={self.input}, output={self.output}]"


def make_pattern(nodes, inputs, outputs, name="pattern"):
    from .graph import Graph

    pattern = Graph.make_graph(name, {}, nodes, inputs, outputs)
    pattern.update_topology()
    return pattern


# class MatchResult(object):
#     def __init__(self):
#         self._nodes = {}

#     def __iter__(self):
#         return self._nodes.values().__iter__()

#     def __getitem__(self, key):
#         return self._nodes[key]

#     def add(self, key, value):
#         self._nodes[key] = value
