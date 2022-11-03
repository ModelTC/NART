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

import itertools
import torch.autograd.function as function
from torch.autograd import Variable
from torch.nn import Module
import torch
import functools
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


_unravel = _iter_filter(
    lambda x: isinstance(x, torch.autograd.Variable), condition_msg="Variables"
)
_flatten = torch.jit._flatten
_unflatten = torch.jit._unflatten

# Functional version that assumes that all parameters are explicitly
# specified
def _raw_trace(nderivs=0):
    def raw_trace(f):
        # f takes two arguments, (in_vars, in_struct) (as determined
        # by _flatten); furthermore, it must be the case that in_vars
        # contains all Variable inputs (including parameters.)  It must
        # produce two outputs, (out_vars, out_struct) (also as determined
        # by _flatten).
        @functools.wraps(f)
        def wrapper(in_vars, in_struct=None):
            trace = torch._C._tracer_enter(in_vars, nderivs)
            out_vars, out_struct = f(in_vars, in_struct)
            torch._C._tracer_exit(out_vars)
            return trace, (out_vars, out_struct)

        return wrapper

    return raw_trace


class TracedModule(Module):
    def __init__(self, inner, nderivs=0):
        super(TracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner
        self.nderivs = nderivs

    def forward(self, *args, **kwargs):
        @_raw_trace(nderivs=self.nderivs)
        def traced_inner(in_vars, in_struct):
            out = self.inner(*args, **kwargs)
            out = list(_unravel(out))
            return _flatten(out)

        kw_items = list(kwargs.items())
        kw_items.sort()
        dict_input = isinstance(args[0], dict)
        if dict_input:
            in_vars, in_struct = _flatten(
                (args[0]["image"], tuple(kw_items)),
                self.state_dict(keep_vars=True).values(),
            )
            args[0]["image"] = args[0]["image"][0]
        else:
            in_vars, in_struct = _flatten(
                (args, tuple(kw_items)), self.state_dict(keep_vars=True).values()
            )
        torch.jit._tracing = True
        trace, (out_vars, out_struct) = traced_inner(in_vars, in_struct)
        torch.jit._tracing = False
        out, unmatched = _unflatten(out_vars, out_struct)
        assert len(unmatched) == 0
        return trace, out
