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

""" Passes that are used to do graph simplification or optimization.
"""
from ..core.graph import Graph
from ..ops import Reshape, Op
from . import Pass, EliminateNodes


class RemoveIdentityOps(EliminateNodes):
    """Remove operators that actually does nothing."""

    def __init__(self):
        def is_identity(node: Op, graph: Graph):
            import math

            if node.op_type == "Pad":
                if "pads" in node.attributes:
                    return all(
                        math.isclose(x, 0) for x in node.get_attribute_value("pads")
                    )
                else:
                    pads = graph.get_const_tensor_as_array(
                        node.input[1], allow_none=True
                    )
                    return (pads is not None) and all(math.isclose(x, 0) for x in pads)
            return False

        super().__init__(is_identity)
