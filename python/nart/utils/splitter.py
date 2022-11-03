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

from ..core import Node
from ..core import Graph


class _op_helper(object):
    def __init__(self, op_node, name_to_input):
        self.op_node = op_node
        self._input_tensors_check_dict = {}
        for i in name_to_input[op_node.name].values():
            if i.is_empty():
                # print(f'initializing op with empty input')
                # empty input tensor do NOT require checking
                pass
            else:
                self._input_tensors_check_dict[i] = False
        self._module_type_code = 0

    def add_input(self, input_tensor):
        temp_dict = self._input_tensors_check_dict
        if input_tensor not in temp_dict:
            return False
        else:
            self._input_tensors_check_dict[input_tensor] = True
            return True

    def reset(self):
        for item in self._input_tensors_check_dict:
            self._input_tensors_check_dict[item] = False

    def satisfy(self):
        for item in self._input_tensors_check_dict:
            if not self._input_tensors_check_dict[item]:
                return False
        return True

    def set_module_type_code(self, code):
        self._module_type_code = code

    def get_module_type_code(self):
        return self._module_type_code

    def name(self):
        return self.op_node.name

    def get_op_node(self):
        return self.op_node

    def get_output_location(self):
        output_location = []
        for name in self.op_node.output:
            output_location.append(Splitter.tensor_location(self.op_node.name, name))
        return output_location


class Splitter(object):
    """split net.

    Attributes:
        net: origin core.graph.Net.
        op_type_dict: a dict point out node type.
        current_net_dict: name to node.
        name_to_input: {  "nodename1" : {"inputname1" : tensor_location1, "inputname2" : tensor_location1} }
    """

    class tensor_location:
        """tensor location.

        Attributes:
            node_name: node name.
            input_name: input name.

        """

        def __init__(self, node_name, input_name):
            self.node_name = node_name
            self.input_name = input_name

        def __eq__(self, other):
            return self.__dict__ == other.__dict__

        def __hash__(self):
            return hash(self.node_name + self.input_name)

        # Some node's input is empty.
        # In this case, that tensor is an empty tensor, and it's producer node name is empty, as well as input name
        def is_empty(self):
            if self.node_name == "" and self.input_name == "":
                return True
            return False

    def __init__(self, net, op_to_type_code_dict, breakpoints=[]):
        import copy

        self.net = copy.deepcopy(net)
        if not hasattr(op_to_type_code_dict, "__getitem__"):
            raise TypeError("op_type_dict should have __getitem__ function.")
        self.op_type_dict = op_to_type_code_dict
        self.current_net_dict = {}
        self.type_set = [0]
        self.current_subnet_list = []
        self.breakpoints = breakpoints
        self.name_to_input = {}

    def prepare(self):
        """generate name_to_input"""
        for n in self.net.nodes:
            self.name_to_input[n.name] = {}
            for i in n.input:
                if i in self.net.input:
                    self.name_to_input[n.name][i] = Splitter.tensor_location("input", i)
                elif i != "":
                    self.name_to_input[n.name][i] = Splitter.tensor_location(
                        self.net.get_tensor_producer(i)[0].name, i
                    )
                else:
                    self.name_to_input[n.name][i] = Splitter.tensor_location("", i)

        for node in self.net.nodes:
            self.current_net_dict[node.name] = _op_helper(node, self.name_to_input)
            # Copy a dict
            if node.name in self.op_type_dict:
                type_code = self.op_type_dict[node.name]
                self.current_net_dict[node.name].set_module_type_code(type_code)
                if type_code not in self.type_set:
                    self.type_set.append(type_code)

    @staticmethod
    def combine_set(arr_from, arr_to):
        """combine set.

        Args:
            arr_from: list, set from.
            arr_to: set to.

        """
        for item in arr_from:
            if item not in arr_to:
                arr_to.append(item)

    @staticmethod
    def validate_graph(graph):
        # check input graph
        # check no duplicated node name.
        from collections import Counter

        count = Counter([x.name for x in graph.nodes])
        if not all(freq == 1 for name, freq in count.items()):
            duplicated_node = {name: freq for name, freq in count.items() if freq > 1}
            raise ValueError(
                "splitter requires no node name duplication, the following nodes violates this requirement: \n"
                f"{duplicated_node}"
            )

    def run(self):
        """after call run, Splitter split the net to sub_nets."""
        self.validate_graph(self.net)
        self.prepare()
        input_location = []
        for name in self.net.input:
            input_location.append(Splitter.tensor_location("input", name))

        assert len(self.type_set) > 0
        cur_net_dict = self.current_net_dict
        cur_subnet_list = self.current_subnet_list
        cur_id = 0
        cur_subnet_info = Graph(f"{self.net.name}_subnet{cur_id}")

        cur_type_code = list(cur_net_dict.values())[0].get_module_type_code()

        weight_dict = {}
        for k, v in self.net.initializer.items():
            weight_dict[k] = v

        while len(cur_net_dict) > 0:
            current_net_dict_length_before = len(cur_net_dict)
            temp_cur_net_dict = cur_net_dict.copy()
            # on python 3.7+ and cpython implementation of python 3.6, dictionaries preserve insertion order.
            for op_name in cur_net_dict:
                op_helper = cur_net_dict[op_name]
                assert isinstance(op_helper, _op_helper)

                if op_helper.get_module_type_code() == cur_type_code:
                    if (op_helper.name() in self.breakpoints) and len(
                        cur_subnet_info.nodes
                    ) != 0:
                        # if current op is in breakpoints and it's not the first op in current subnet,
                        # break at here.
                        break
                    for input_tensor_loc in input_location:
                        op_helper.add_input(input_tensor_loc)

                    if op_helper.satisfy():
                        cur_subnet_info.add_node(op_helper.get_op_node())
                        for input_name in op_helper.get_op_node().input:
                            if input_name in weight_dict:
                                # weight
                                # cur_subnet_info.add_initializer(onnx.helper.make_tensor(input_name, onnx.TensorProto.FLOAT, weight_dict[input_name]))
                                cur_subnet_info.add_initializer(weight_dict[input_name])

                        temp_cur_net_dict.pop(op_name)
                        Splitter.combine_set(
                            op_helper.get_output_location(), input_location
                        )

            cur_net_dict = temp_cur_net_dict
            for op_name in cur_net_dict:
                op_helper = cur_net_dict[op_name]
                op_helper.reset()
            for i in cur_subnet_info.input:
                cur_subnet_info.set_tensor_shape(
                    i, self.net.get_tensor_shape(i), self.net.get_tensor_dtype(i)
                )
            cur_subnet_info.update_tensor_shape()
            if current_net_dict_length_before != len(cur_net_dict):
                cur_subnet_list.append((cur_subnet_info, cur_type_code))
                cur_id += 1
                cur_subnet_info = Graph(f"{self.net.name}_subnet{cur_id}")
            else:
                if len(cur_net_dict) > 0:
                    cur_type_code = list(cur_net_dict.values())[
                        0
                    ].get_module_type_code()

        self.current_subnet_list = cur_subnet_list
        self.add_net_outputs()

    def add_net_outputs(self):
        """add subnet's output after split"""
        input_names = []
        for i in range(len(self.current_subnet_list)):
            net, _ = self.current_subnet_list[i]
            input_names = list(net.input)
            for j in range(i):
                f_net, _ = self.current_subnet_list[j]
                for name in input_names:
                    producer = f_net.get_tensor_producer(name)
                    if producer and producer[0] != "input":
                        f_net.add_output(name)
            # add whole network output
            for name in self.net.output:
                producer = net.get_tensor_producer(name)
                if producer and producer[0] != "input":
                    net.add_output(name)
