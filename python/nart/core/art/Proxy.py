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

from ... import _nart
from . import FakeOp
import numpy as np

import logging

logger = logging.getLogger("nart.core.art")

cuda_workspaces = None
default_workspace = None
quant_workspace = None


# class DeserializeConfig(_nart.art.DeserializeConfig):
#     def __init__(self, workspaces=None, input_mem_tp=None, backend='default') :
#         super(DeserializeConfig, self).__init__()
#         if not workspaces:
#             global default_workspace, quant_workspace
#             default_workspace = _nart.art.Workspace.new_default()
#             quant_workspace = _nart.art.Workspace.new_quant()
#             workspaces = [default_workspace, quant_workspace]
#         if not input_mem_tp:
#             input_mem_tp = default_workspace.mem_tp
#         cuda_enabled = hasattr(_nart.art.Workspace, "new_cuda")
#         if not cuda_enabled and backend == "cuda":
#             logger.warning("Requesting cuda workspace, but cuda is not built, fallback to default only")
#             backend = "default"

#         if backend == 'cuda':
#             global cuda_workspace
#             cuda_workspace = _nart.art.Workspace.new_cuda()
#             cuda_mem_tp = cuda_workspace.mem_tp
#             self.workspaces = [cuda_workspace] + list(workspaces)
#             self.input_mem_tp = cuda_mem_tp
#         else:
#             self.workspaces = list(workspaces)
#             self.input_mem_tp = input_mem_tp


class DeserializeConfig:
    def __init__(self) -> None:
        raise NotImplementedError("")


class ModuleProxy:
    class Constraint:
        __slot__ = ["name", "item", "dtype", "constraint", "default_value"]

        def __init__(self, name, item, dtype, constraint, default_value=None):
            self.name = name
            self.item = item
            self.dtype = dtype.lower()
            self.constraint = constraint
            if constraint == "optional":
                assert default_value is not None
                self.default_value = default_value

        def __repr__(self):
            if self.constraint == "optional":
                return f'{{name="{self.name}", dtype="{self.dtype}", constraint="{self.constraint}", default_value="{self.default_value}"}}'
            return f'{{name="{self.name}", dtype="{self.dtype}", constraint="{self.constraint}"}}'

    def __init__(self, module_name):
        self.true_obj = _nart.proxy("default")
        self.mp_name_tp_ = {}
        self.mp_name_constraint_ = {}
        for item in self.true_obj.op_tps:
            self.mp_name_tp_[item[1]] = item[0]
            self.mp_name_constraint_[item[1]] = {}
            for itm in self.true_obj.constraints_of(item[1]):
                self.mp_name_constraint_[item[1]][itm[0]] = ModuleProxy.Constraint(*itm)
        self.repeated_setting = def_setting_cls("repeate")
        self.single_setting = def_setting_cls("single")

        for item in self.mp_name_tp_:
            setattr(self, item, gen_class(self, item))

    @property
    def op_tps(self):
        return list(self.mp_name_tp_.keys())

    def constraints_of(self, key):
        return self.mp_name_constraint_[key] if key in self.mp_name_constraint_ else []


def def_setting_cls(rep):
    class setting:
        def __init__(self, name, dtype, v):
            if rep is True:
                self.data = np.array(v, dtype=dtype)
            else:
                if hasattr(v, "__len__"):
                    assert 1 == len(v)
                else:
                    v = [v]
                self.data = np.array(v, dtype=dtype)
            self.dtype = dtype

    setting.rep = rep
    return setting


def gen_class(mod, name):
    class _cls(FakeOp.FakeOpInterface):
        def __init__(self, inputs, params=None, layer_name=None, **k):
            super(_cls, self).__init__(inputs, params)
            self.settings = {}
            self.layer_name = layer_name
            for item in mod.mp_name_constraint_[name].values():
                attr_name = item.name.lower()[8:]
                if item.constraint == "required" and attr_name not in k:
                    err_str = f"{name} should set attr(s) "
                    err_str += str(
                        list(
                            map(
                                lambda x: x.name.lower()[8:],
                                filter(
                                    lambda x: x.constraint == "required",
                                    mod.mp_name_constraint_[name].values(),
                                ),
                            )
                        )
                    )
                    raise RuntimeError(err_str)
            for key, v in k.items():
                setattr(self, key, v)
            _nart.proxy.proxy_init(self)

        def op_tp_code(self):
            return mod.mp_name_tp_[name]

        def get_settings(self):
            res = []
            for k, v in self.settings.items():
                res.append(
                    (mod.mp_name_constraint_[name]["SETTING_" + k.upper()].item, v)
                )
            return res

        def __repr__(self):
            res = "FakeOp({self.__class__.__name__}):\n"
            for k, v in self.settings.items():
                res += f"\tsetting {k} -> {v}\n"
            res += "\tpass"
            return res

        @property
        def module(self):
            return mod

    def gen_attr_fun(attr_name, item):
        def attr_getter(self):
            if attr_name in self.settings:
                return self.settings[attr_name]
            if item.constraint == "optional":
                return item.default_value

        def attr_setter(self, value):
            if item.constraint != "repeated":
                self.settings[attr_name] = np.array([value], dtype=item.dtype)[0]
            else:
                self.settings[attr_name] = np.array(value, dtype=item.dtype)

        return attr_getter, attr_setter

    for item in mod.mp_name_constraint_[name].values():
        attr_name = item.name.lower()[8:]
        setattr(_cls, attr_name, property(*gen_attr_fun(attr_name, item)))
    _cls.__name__ = name
    return _cls
