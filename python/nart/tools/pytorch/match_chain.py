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

"""
Nodes in traced graph will have methods named `scope_name` and `kind`.

The format of `scope_name` is like
`ResNet/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]`.

The format of `kind` depends on the type of tracing.
For example, when tracing the model with torch.onnx.OperatorExportTypes.RAW,
the format will be like `aten::_convolution`.

The function `refactor.trace_graph.gen_dot`
can dump the traced graph into .dot file.
You can use it for further observation.
"""

import re
import torch

from .trace_graph import trace_graph


# matches substrings like `[layer3]`
_scope_attr_re = re.compile(r"\[.*?\]")


def _scope_to_attrs(scope_name):
    return list(map(lambda x: x[1:-1], _scope_attr_re.findall(scope_name)))


def _scope_parent_module(model, scope_name):
    attrs = _scope_to_attrs(scope_name)
    if not attrs:
        return None
    if len(attrs) <= 1:
        return model
    return dict(model.named_modules())[".".join(attrs[:-1])]


def _scope_parent_module_name(scope_name):
    attrs = _scope_to_attrs(scope_name)
    if not attrs:
        return None
    if len(attrs) <= 1:
        return ""
    return ".".join(attrs[:-1])


def _scope_module_name(scope_name):
    return ".".join(_scope_to_attrs(scope_name))


def _scope_attr_name(scope_name):
    k = _scope_attr_re.findall(scope_name)
    if not k:
        return ""
    return k[-1][1:-1]


def _get_degree(g, inputs_num):
    """Get essential nodes and their in-degree of the traced graph."""
    nodes = frozenset(g.nodes())
    edges = []

    # initialize queue
    q = []
    vis = set()
    for i in list(g.inputs())[:inputs_num]:
        for u in i.uses():
            u = u.user

            if u not in nodes:
                continue

            if u in vis:
                continue
            vis.add(u)

            q.append(u)

    while q:
        u = q.pop(0)

        for i in u.outputs():
            for v in i.uses():
                v = v.user
                if v not in nodes:
                    continue

                edges.append((u, v))

                if v in vis:
                    continue
                vis.add(v)
                q.append(v)

    degree = {u: 0 for u in vis}
    for _, v in edges:
        degree[v] += 1

    return degree


def _match_chain(model, dummy_inputs, kinds):
    """Find nodes in the traced graph of `model` with kind of `kinds`.

    The Executions of `kinds` shall form a linear chain.
    """
    g = trace_graph(model, dummy_inputs)
    nodes = frozenset(g.nodes())
    degree = _get_degree(g, len(dummy_inputs))

    # initialize queue, which stores current vertix and the chain
    q = []
    vis = set()
    for i in list(g.inputs())[: len(dummy_inputs)]:
        for u in i.uses():
            u = u.user

            if u not in nodes:
                continue
            assert degree[u] == 0

            if u in vis:
                continue
            vis.add(u)

            q.append((u, [u]))

    while q:
        u, chain = q.pop(0)

        # we only need to find chains that match `kinds`
        if len(chain) > len(kinds):
            chain = chain[-len(kinds) :]

        if (
            _scope_parent_module(model, u.scopeName())
            and len(chain) == len(kinds)
            and all(map(lambda x, y: x.kind() == y, chain, kinds))
        ):
            # if `chain` matches `kinds`
            # yield parent info and attribute name
            yield _scope_parent_module_name(u.scopeName()), _scope_parent_module(
                model, u.scopeName()
            ), list(map(lambda x: _scope_attr_name(x.scopeName()), chain))

            # clear chain
            chain = []

        for i in u.outputs():
            for v in i.uses():
                v = v.user

                if v not in nodes:
                    continue

                degree[v] -= 1
                if degree[v] != 0:
                    continue

                if v in vis:
                    continue
                vis.add(v)

                if _scope_parent_module_name(
                    u.scopeName()
                ) == _scope_parent_module_name(v.scopeName()):
                    # still in the same parent module
                    q.append((v, chain + [v]))
                else:
                    # in different parent modules
                    q.append((v, [v]))


def _toposort(model, dummy_inputs):
    """Find all essential nodes in the traced graph."""
    g = trace_graph(model, dummy_inputs)
    nodes = frozenset(g.nodes())
    degree = _get_degree(g, len(dummy_inputs))

    # initialize queue
    q = []
    vis = set()
    for i in list(g.inputs())[: len(dummy_inputs)]:
        for u in i.uses():
            u = u.user

            if u not in nodes:
                continue
            assert degree[u] == 0

            if u in vis:
                continue
            vis.add(u)

            q.append(u)

    while q:
        u = q.pop(0)

        # yield its scopeName and its kind
        yield u.scopeName(), u.kind()

        for i in u.outputs():
            for v in i.uses():
                v = v.user
                if v not in nodes:
                    continue
                degree[v] -= 1
                if degree[v] != 0:
                    continue
                if v in vis or v not in nodes:
                    continue
                vis.add(v)
                q.append(v)


def match_all_chain(pattern_class, pattern_inputs, model, model_inputs):
    """Find all executions occurred in `model`
    that is the same with `pattern_class`'s.

    The traced graph of instances of pattern_class
    shall form a linear chain.

    Args:
        pattern_class: type of torch.nn.Module.
        pattern_inputs: tuple of torch.Tensor or torch.Tensor, dummy inputs for pattern.
        model: torch.nn.Module.
        model_inputs: tuple of torch.Tensor or torch.Tensor, dummy inputs for `model`.

    Returns:
        generator, yield the submodule and matched attributes
        the same with executions in `pattern_class`.
    """
    if isinstance(pattern_inputs, torch.Tensor):
        pattern_inputs = (pattern_inputs,)
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = (model_inputs,)

    # get the scope names and kinds of execution nodes in pattern
    pattern_model = pattern_class()
    pattern_model.eval()
    device = next(pattern_model.parameters(), torch.tensor([])).device
    pattern_inputs = tuple(
        map(
            lambda x: x.cuda(device) if device.type == "cuda" else x.cpu(),
            pattern_inputs,
        )
    )
    scope_names, kinds = zip(*_toposort(pattern_model, pattern_inputs))

    # find all submodules and matched attributes
    for submodule_name, submodule, attrs in _match_chain(model, model_inputs, kinds):

        # put corresponded children in `model` in an instance of `pattern_class`
        children = dict(submodule.named_children())
        pattern = pattern_class()
        for name, attr in zip(scope_names, attrs):
            pattern.add_module(_scope_module_name(name), children[attr])

        # yield the result
        yield submodule_name, submodule, pattern, attrs
