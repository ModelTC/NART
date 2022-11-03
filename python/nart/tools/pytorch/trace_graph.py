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

import torch

from torch.onnx.utils import _trace
from torch.onnx import OperatorExportTypes


def trace_graph(model, dummy_inputs, type_=OperatorExportTypes.ONNX):
    """Trace `forward` of the model with dummy inputs.

    Args:
        model: torch.nn.Module.
        dummy_inputs: tuple of torch.Tensor or torch.Tensor,
            dummy inputs for `model`.
        type_: torch._C._onnx.OperatorExportTypes.

    Returns:
        The traced graph.
    """
    if isinstance(dummy_inputs, torch.Tensor):
        dummy_inputs = (dummy_inputs,)

    if torch.__version__ > "1.3.1":

        class scope_name_workaround(object):
            """The logic of torch.nn.Module._slow_forward has changed since v1.4.0, and will not record scope name
            when calling torch.onnx.utils._trace.
            This class rollback to v1.3.1.
            """

            def __init__(self):
                self.backup = None

            def __enter__(self):
                def _tracing_name(self_, tracing_state):
                    if not tracing_state._traced_module_stack:
                        return None
                    module = tracing_state._traced_module_stack[-1]
                    for name, child in module.named_children():
                        if child is self_:
                            return name
                    return None

                def _slow_forward(self_, *input, **kwargs):
                    tracing_state = torch._C._get_tracing_state()
                    if not tracing_state or isinstance(
                        self_.forward, torch._C.ScriptMethod
                    ):
                        return self_.forward(*input, **kwargs)
                    if not hasattr(tracing_state, "_traced_module_stack"):
                        tracing_state._traced_module_stack = []
                    name = _tracing_name(self_, tracing_state)
                    if name:
                        tracing_state.push_scope("%s[%s]" % (self_._get_name(), name))
                    else:
                        tracing_state.push_scope(self_._get_name())
                    tracing_state._traced_module_stack.append(self_)
                    try:
                        result = self_.forward(*input, **kwargs)
                    finally:
                        tracing_state.pop_scope()
                        tracing_state._traced_module_stack.pop()
                    return result

                self.backup = torch.nn.Module._slow_forward
                setattr(torch.nn.Module, "_slow_forward", _slow_forward)

            def __exit__(self, type, value, tb):
                setattr(torch.nn.Module, "_slow_forward", self.backup)

        with scope_name_workaround():
            traced_graph = _trace(model, dummy_inputs, type_)
    else:
        traced_graph = _trace(model, dummy_inputs, type_)

    if not isinstance(traced_graph, torch._C.Graph):
        traced_graph = traced_graph.graph()
    return traced_graph


def _coloring(start, edges):
    """Visit all vertices."""
    q = list(start)
    c = set()
    while q:
        u = q.pop(0)
        c.add(u)
        for v in edges(u):
            if v not in c:
                q.append(v)
    return c


def gen_dot(g, path, n_model_inputs=1):
    """Utility function to dump traced graph into .dot file."""
    import networkx as nx

    G = nx.DiGraph()

    node_map = dict()

    def nodeName(u):
        name = "%s %s %s" % (str(u)[1:4], u.kind(), u.scopeName())
        if name not in node_map:
            node_map[name] = u
        else:
            assert node_map[name] == u
        return name

    tensor_map = dict()

    def tensorName(u):
        name = u.uniqueName()
        if name not in tensor_map:
            tensor_map[name] = u
        else:
            assert tensor_map[name] == u
        return name

    for u in g.nodes():
        for v in u.outputs():
            G.add_edge(nodeName(u), tensorName(v))

    for v in g.nodes():
        for u in v.inputs():
            G.add_edge(tensorName(u), nodeName(v))

    valid_inputs = frozenset(list(g.inputs())[:n_model_inputs])
    assert len(valid_inputs) == n_model_inputs
    valid_outputs = frozenset(g.outputs())

    def get_next(u):
        if isinstance(u, torch._C.Value):
            return (v.user for v in u.uses())
        elif isinstance(u, torch._C.Node):
            return u.outputs()
        assert False

    def get_previous(u):
        if isinstance(u, torch._C.Value):
            return [u.node()]
        elif isinstance(u, torch._C.Node):
            return u.inputs()
        assert False

    o_vis, r_vis = _coloring(valid_inputs, get_next), _coloring(
        valid_outputs, get_previous
    )

    assert all(map(lambda x: x in o_vis and x in r_vis, valid_inputs))
    assert all(map(lambda x: x in o_vis and x in r_vis, valid_outputs))

    for k, v in node_map.items():
        if v not in o_vis or v not in r_vis:
            G.remove_node(k)

    for k, v in tensor_map.items():
        if v not in o_vis or v not in r_vis:
            G.remove_node(k)

    from networkx.drawing.nx_agraph import write_dot

    write_dot(G, path)
