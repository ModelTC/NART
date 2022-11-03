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
from torch.nn import Module
import warnings


def _iter_filter(condition, allow_unknown=False, condition_msg=None, keyword=None):
    def _iter(obj):
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for var in obj:
                yield from _iter(var)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if keyword is None or key == keyword:
                    yield from _iter(value)
        elif allow_unknown:
            yield obj
        else:
            warnings.warn(
                "Skipping input - Auto nesting doesn't know how to process "
                "an input object of type "
                + torch.typename(obj)
                + (
                    ". Accepted types: "
                    + condition_msg
                    + ", or lists/tuples of them, or a dict where 'image' contains the tensor."
                    if condition_msg
                    else ""
                )
            )
            return

    return _iter


_unravel = _iter_filter(lambda x: isinstance(x, torch.Tensor), condition_msg="Tensors")
_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten


def _unique_state_dict(module, keep_vars=False):
    state_dict = module.state_dict(keep_vars=keep_vars)
    filtered_dict = type(state_dict)()
    seen_ids = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        filtered_dict[k] = v
    return filtered_dict


class LegacyTracedModule(Module):
    def __init__(self, inner):
        super(LegacyTracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner

    def forward(self, *args):
        dict_mode = type(args[0]) is dict
        if dict_mode:
            input_dict = args[0]
            args = (args[0]["image"],)
        in_vars, in_desc = _flatten(args)
        # NOTE: use full state, because we need it for BatchNorm export
        # This differs from the compiler path, which doesn't support it at the moment.
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        trace, all_trace_inputs = torch._C._tracer_enter(in_vars + module_state)
        torch.jit._tracing = True
        trace_inputs = _unflatten(all_trace_inputs[: len(in_vars)], in_desc)
        if dict_mode:
            input_dict["image"] = trace_inputs[0][0]
            out = self.inner(input_dict)
        else:
            out = self.inner(*trace_inputs)
        out = list(_unravel(out))
        out_vars, _ = _flatten(out)
        torch.jit._tracing = False
        torch._C._tracer_exit(out_vars)
        return trace, out
