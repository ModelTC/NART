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

MODULES_DICT = {
    "default": ["default"],
    "cuda": ["default", "cuda"],
    "tensorrt": ["default", "cuda", "tensorrt"],
}

DEFAULT_ENABLE = {"default", "cuda"}


class Net:
    def __init__(self, name=""):
        self._name = name
        self._parade = None
        self._inputs = None
        self._outputs = None

    @staticmethod
    def from_graph(g, output_all=True, backend="default"):
        """Construct a Net from core.Graph.

        Args:
            g (core.Graph): the graph
            output_all (bool): should all intermediate tensor be marked as output?
        """
        from ..modules import CaffeCUDAParser

        passes = CaffeCUDAParser.get_passes()
        for item in passes:
            item.run(g)
        g.update_topology()
        g.update_tensor_shape()
        CaffeCUDAParser.register_defaults()
        parade = CaffeCUDAParser(g).parse()
        if output_all:
            for n in g.nodes:
                for o in n.output:
                    parade.mark_as_output_by_name(o)
        return Net.from_parade(g.name, parade, backend)

    @staticmethod
    def from_parade(name, parade, backend="default"):
        import numpy
        import nart
        from nart.art import RuntimeContext, Parade, get_empty_io

        assert isinstance(parade, nart.FakeParade)
        ser_data = nart.serialize_v1(parade)
        net = Net(name)
        from nart.art import create_context_for_target

        net._ctx = create_context_for_target(backend)
        if net._ctx.loaded_workspaces == MODULES_DICT[backend]:
            # print("successfully load modules")
            pass
        else:
            if backend in DEFAULT_ENABLE:
                # print("modle will run on default module")
                pass
            else:
                raise RuntimeError(backend, "can not run on default")

        net._parade = Parade(ser_data, net._ctx)

        inputs, outputs = get_empty_io(net._parade)
        net._inputs = inputs
        net._outputs = outputs
        return net

    @property
    def name(self):
        return self._name

    def get_input_binding(self):
        if self._parade is None:
            raise RuntimeError("please set net")
        return self._inputs.copy()

    def get_output_binding(self):
        if self._parade is None:
            raise RuntimeError("please set net")
        return self._outputs.copy()

    def get_binding(self, name):
        if self._parade is None:
            raise RuntimeError("please set net")

        if name not in self._outputs:
            raise RuntimeError("can not get bingding")
        else:
            return self._outputs[name]

    def forward(self):
        if self._parade is None:
            raise RuntimeError("please set net")
        self._parade.run(self._inputs, self._outputs)
        return True

    def get_raw_tensor_data(self, tensor_name):
        """Get the raw pointer of a tensor."""
        return self._parade.get_raw_tensor_data(tensor_name)
