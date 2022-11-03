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

import onnx
import logging
from ..core.art import Dtype

logger = logging.getLogger("nart.graph")


class NodeBase:
    """Base class owns inputs and outputs"""

    def __init__(self, name=""):
        self._input = []
        self._output = []
        self._name = name

    @property
    def name(self):
        """`str`: name"""
        return self._name

    @property
    def input(self):
        """`list of str`: contains node's inputs"""
        return self._input

    @property
    def output(self):
        """`list of str` contains node's outputs"""
        return self._output

    def add_input(self, inp):
        """add input by name

        Args:
            inp (str): input name
        """
        self._input.append(inp)

    def add_output(self, output):
        """add output by name

        Args:
            output (str): output name
        """
        if output not in self._output:
            self._output.append(output)

    def del_input(self, inp):
        """del input by name

        Args:
            inp (str): input name

        Returns:
            bool: return True if `self` had `inp` and deleted `inp` else False
        """
        if inp in self.input:
            self._input.remove(inp)
            return True
        return False

    def del_output(self, output):
        """del output by name

        Args:
            output: output name

        Returns:
            bool: return True if `self` had `output` and deleted `output` else False
        """
        if output in self.output:
            self._output.remove(output)
            return True
        return False

    def has_input(self, idx):
        """Does this node has the `idx`-th input.

        idx (int): the index of input.
        """
        return len(self.input) > idx and bool(self.input[idx])

    def has_output(self, idx):
        """Does this node has the `idx`-th output.

        idx (int): the index of output.
        """
        return len(self.output) > idx and bool(self.output[idx])


class Node(NodeBase):
    """Node class owns enough infomation for onnx.NodeProto"""

    def __init__(self):
        super(Node, self).__init__()
        self._attributes = {}
        self._op_type = ""
        self._domain = ""
        self._doc_string = ""
        self._owning_graph = None

    def get_attribute_value(self, name, default):
        """ """
        from onnx import helper

        if name in self.attributes:
            value = self.attributes[name]
            if isinstance(value, onnx.AttributeProto):
                return helper.get_attribute_value(value)
        return default

    def check_similarity(self, other):
        """ """
        if self.op_type != other.op_type:
            # return early when two node was different op type, since it's no safe to compare two different op.
            logger.debug(
                f"checking {self.name} vs {other.name}, op_type differs [False]"
            )
            return False
        logger.debug("checking " + str(self) + " and " + str(other))
        logger.debug("check op_type [True]")

        def get_attribute_with_default(node, name):
            """a helper function for getting attribute of a node.
            the attribute may not necessaryly be a onnx.AttributeProto, this functions handles this.
            """
            from onnx import helper

            if name in node.attributes:
                value = node.attributes[name]
                if isinstance(value, onnx.AttributeProto):
                    return helper.get_attribute_value(node.attributes[name])
                else:
                    return value
            assert (
                len(node.attr_dict[name]) > 1
            ), f"node `{node.name}`'s attribute {name} was not defined, which doesn't have default value"
            return node.attr_dict[name][1]

        logger.debug("checking " + str(self) + " and " + str(other))
        res = True
        res = res and len(self.input) == len(other.input)
        logger.debug(f"check input lens [{res}] ")
        # template must be in front, so that __eq__ method of pattern attribtues will be called.
        for attr in set(self.attributes.keys()) or set(other.attributes.keys()):
            a = get_attribute_with_default(self, attr)
            b = get_attribute_with_default(other, attr)
            res = res and a == b
            logger.debug(f"check attr `{attr}` [{res}]: {a} vs {b}")
        res = res and len(self.output) == len(other.output)
        logger.debug(f"check output lens [{res}]")
        return res

    @property
    def op_type(self):
        """`string`: identifier for op"""
        return self._op_type

    @property
    def attributes(self):
        """`dict<str, onnx.AttributeProto>`: contains all attributes"""
        return self._attributes

    def add_attribute(self, attr, force=False):
        """add attribute by attr

        Args:
            attr (onnx.AttributeProto): attribute to be added
            force (bool): overwrite attributes if attr.name has been added. Defaults to False

        Returns:
            bool: return True if added attr into self, else False
        """
        if attr.name not in self._attributes:
            self._attributes[attr.name] = attr
            return True
        elif force:
            self._attributes[attr.name] = attr
            return True
        return False

    def del_attribute(self, attr):
        """del attribute by attr

        Args:
            attr (onnx.AttributeProto): attribute to be deleted

        Returns:
            bool: return True if deleted attr from self, else False
        """
        if attr.name in self.attributes:
            self._attributes.pop(attr.name)
            return True
        return False

    @property
    def owning_graph(self):
        """`Graph`: graph which belong to."""
        return self._owning_graph

    @owning_graph.setter
    def owning_graph(self, graph):
        self._owning_graph = graph

    def replace_input_purely(self, ori_input, new_input):
        """replace a input with another one. ``Graph.update_topology()`` should be called after modification.

        Args:
            ori_input (str): original input name
            new_input (str): newer input name
        """
        for idx in range(len(self.input)):
            if self.input[idx] == ori_input:
                self.input[idx] = new_input

    def replace_input(self, ori_input, new_input):
        """replace a input with another one. Will update topology of owning graph if exists.

        Args:
            ori_input (str): original input name
            new_input (str): newer input name
        """
        ori_inputs = set(self.input)
        self.replace_input_purely(ori_input, new_input)
        if self.owning_graph is not None:
            self.owning_graph._update_topology_with_delta(
                self, ori_inputs, self.output, self.input, self.output
            )
        else:
            logger.warning(
                "calling replace_input on a Node whose owning graph is None, "
                "if this is intended, call `replace_input_purely` instead"
            )

    def replace_output_purely(self, origin_output, new_output):
        """replace a output with another one. ``Graph.update_topology()`` should be called after modification.

        Args:
            origin_output (str): original output name
            new_output (str): newer output name
        """
        for idx in range(len(self.output)):
            if self.output[idx] == origin_output:
                self.output[idx] = new_output

    def replace_output(self, origin_output, new_output):
        """replace a output with another one. Will update topology of owning graph if exists.

        Args:
            origin_output (str): original output name
            new_output (str): newer output name
        """
        ori_outputs = set(self.output)
        self.replace_output_purely(origin_output, new_output)
        if self.owning_graph is not None:
            self.owning_graph._update_topology_with_delta(
                self, self.input, ori_outputs, self.input, self.output
            )
        else:
            logger.warning(
                "calling replace_output on a Node whose owning graph is None, "
                "if this is intended, call `replace_output_purely` instead"
            )

    @staticmethod
    def from_onnx(node):
        """create Node from onnx.NodeProto with same name, op_type, input, ouptut and attributes

        Args:
            node (onnx.NodeProto): onnx NodeProto object
        Returns:
            Node:
        """
        res = Node.make_node(
            node.name,
            node.op_type,
            node.input,
            node.output,
            node.attribute,
            node.domain,
            node.doc_string,
        )
        return res

    @staticmethod
    def make_node(
        name, op_type, input, output, attributes=[], domain="", doc_string=""
    ):
        """create Node from with given name, op_type, input, ouptut and attributes

        Args:
            name (str): node name
            op_type (str): node type
            input (list of str): node inputs
            output (list of str): node outputs
            attributes (list of onnx.AttributeProto): node attributes. Defaults to empty list.
            domain (str): node domain. Defaults to empty str.
            doc_string (str): node domain. Defaults to empty str.
        Returns:
            Node:
        """
        node = Node()
        node._name = name
        node._op_type = op_type
        node._input = input
        node._output = output
        list(map(lambda x: node.add_attribute(x), attributes))
        node._domain = domain
        node._doc_string = doc_string
        return node

    def dump_to_onnx(self):
        """dump to onnx.NodeProto

        Returns:
            onnx.NodeProto: create onnx.NodeProto with self's members
        """
        node = onnx.NodeProto()
        node.name = self._name
        node.op_type = self.op_type
        node.domain = self._domain
        node.doc_string = self._doc_string
        node.input[:] = self.input
        node.output[:] = self.output
        node.attribute.extend([v for k, v in self.attributes.items()])
        return node

    def __repr__(self):
        return f"Node{'::'+self.op_type if self.op_type != '' else ''}[name={self._name}, input={self.input}, output={self.output}]"


class Graph(NodeBase):
    """Graph class owns all resources from onnx.GraphProto and provides basic operators as a graph should do"""

    def __init__(self, name=""):
        super(Graph, self).__init__()
        self._nodes = []
        self._name = name
        self._initializer = {}
        self._doc_string = ""
        self._tensor_shape = {}
        self._tensor_dtype = {}
        self._tensor_producer = {}
        self._tensor_consumer = {}

    def check_similarity(self, other):
        """check if match the template

        Args:
            other (Graph): graph to be check
        Returns:
            bool:
        """

        if len(self.network_inputs) != len(other.network_inputs):
            return False

        a2b = dict()
        b2a = dict()

        def dfs(lhs, rhs):
            """match the sub-dag rooted with lhs/rhs"""
            if lhs == "input" and rhs == "input":
                return True
            elif lhs == "input" or rhs == "input":
                # only one is input, don't match
                return False
            if lhs in a2b and rhs in b2a:
                # both lhs and rhs has already been matched.
                return a2b[lhs] == rhs and b2a[rhs] == lhs
            elif lhs in a2b or rhs in b2a:
                # only one has already been matched, meaning the two graph must not the equal
                return False

            # if not lhs.check_similarity(rhs):
            #     return False
            for inp_a, inp_b in zip(lhs.input, rhs.input):
                prod_a = self.get_tensor_producer(inp_a)
                prod_b = other.get_tensor_producer(inp_b)
                assert len(prod_a) == 1, "bad topology"
                assert len(prod_b) == 1, "bad topology"
                if not dfs(prod_a[0], prod_b[0]):
                    return False
            a2b[lhs] = rhs
            b2a[rhs] = lhs
            return True

        return all(
            dfs(self.get_tensor_producer(lhs)[0], other.get_tensor_producer(rhs)[0])
            for lhs, rhs in zip(self.output, other.output)
        )

    def match(self, pattern):
        """yield subgraph which matches `pattern`

        Args:
            pattern (Graph): pattern
        Returns:
            nodes (set<Node>):

        """
        from .match_utils import CustomMatcher, OpMatcher

        def dfs(sn, pn, rset):
            # print(sn, pn, rset)
            if pn.name in rset:
                # if patter node already matched, check sn is same node that matched ealier.
                return rset[pn.name] == sn
            if not pn.check_similarity(sn):
                return False
            rset[pn.name] = sn

            if isinstance(pn, OpMatcher):
                # as the definition of OpMatcher, dfs should terminate here.
                return True

            # backward search
            if len(sn.input) != len(pn.input):
                return False
            for (si, pi) in zip(sn.input, pn.input):
                sprod = self.get_tensor_producer(si)
                assert (
                    len(sprod) == 1
                ), "bad graph topology, maybe you forgot to call Graph.update_topology()?"
                spn = sprod[0]
                pprod = pattern.get_tensor_producer(pi)
                assert (
                    len(pprod) == 1
                ), "bad graph topology, maybe you forgot to call Graph.update_topology()?"
                ppn = pprod[0]

                if ppn == "input":
                    continue
                if spn != "input" and ppn != "input":
                    if not dfs(spn, ppn, rset):
                        return False
                elif isinstance(ppn, CustomMatcher):
                    # if pattern is an CustomMatcher
                    if ppn.match_input(si, self) == False:
                        return False
                else:
                    return False

            return True

        patn = pattern.nodes[-1]
        for n in list(self.nodes):
            if n.op_type != patn.op_type:
                continue
            ret = dict()
            if dfs(n, patn, ret):
                yield ret
            else:
                continue
        return

    @property
    def nodes(self):
        """`list of Node`: all nodes in this graph"""
        return self._nodes

    def add_node(self, node):
        """add node to this graph. Will update graph.input if node.input can't be
        founded in self.graph, and node.owning_graph will be set to self,
        and graph topology will be updated accordingly.

        Args:
            node (Node): node to be added
        """
        if not isinstance(node, NodeBase):
            raise TypeError(
                "node should be an instance of subclass of nart.core.NodeBase"
            )
        self.add_nodes([node])

    def add_nodes(self, nodes):
        """add nodes to this graph. Will update graph.input if node.input can't be
        founded in self.graph, and node.owning_graph will be set to self,
        and graph topology will be updated accordingly.

        Args:
            nodes (list<Node>): nodes to be added
        """
        for node in nodes:
            for i in node.input:
                prod = self.get_tensor_producer(i)
                if len(prod) == 0:
                    self.add_input(i)
            self._update_topology_with_delta(node, [], [], node.input, node.output)
            node.owning_graph = self
        self.nodes.extend(nodes)

    def add_node_purely(self, node):
        """Add node to this graph only.

        Graph topology will **NOT** be updated, so ``Graph.update_topology()`` should be called after adding.

        Args:
            node (Node): node to be added
        """
        self._nodes.append(node)
        node.owning_graph = self

    def add_nodes_purely(self, nodes):
        self._nodes.extend(nodes)
        for node in nodes:
            node.owning_graph = self

    def insert_node(self, node, pos=None):
        """insert node to specific position. Will update graph.input if node.input can't be
        founded in self.graph, and node.owning_graph will be set to itself,
        and graph topology will be updated accordingly.

        Args:
            node (Node): node to be inserted
            pos (int): target index to be inserted before, or `None` to indicate the tail.
        """
        if not isinstance(node, NodeBase):
            raise TypeError(
                "node should be an instance of subclass of nart.core.NodeBase"
            )
        self.insert_nodes([node], pos)

    def insert_nodes(self, nodes, pos=None):
        """insert nodes to specific position, the ``i``-th node in `nodes` will be inserted at ``pos+i`` after insertion.

        Will update graph.input if node.input can't be
        founded in self.graph, and node.owning_graph will be set to itself,
        and graph topology will be updated accordingly.

        Args:
            node (list<Node>): nodes to be inserted
            pos (int): target index to be inserted before, or `None` to indicate the tail.
        """
        for node in nodes:
            for i in node.input:
                prod = self.get_tensor_producer(i)
                if len(prod) == 0:
                    self.add_input(i)
            self._update_topology_with_delta(node, [], [], node.input, node.output)
            node.owning_graph = self
        pos = pos if pos is not None else len(self.nodes)
        self.nodes[pos:pos] = nodes

    def insert_node_purely(self, node, pos=None):
        """insert node to this graph only.

        Graph topology will **NOT** be updated, so ``Graph.update_topology()`` should be called after adding.

        Args:
            node (Node): node to be inserted
            pos (int): target index to be inserted before, or `None` to indicate the tail.
        """
        pos = pos if pos is not None else len(self.nodes)
        self._nodes.insert(pos, node)
        node.owning_graph = self

    def insert_nodes_purely(self, nodes, pos=None):
        """insert nodes to this graph only. The ``i``-th node in `nodes` will be inserted at ``pos+i`` after insertion.

        Graph topology will **NOT** be updated, so ``Graph.update_topology()`` should be called after adding.

        Args:
            node (lsit<Node>): node to be inserted
            pos (int): target index to be inserted before, or `None` to indicate the tail.
        """
        pos = pos if pos is not None else len(self.nodes)
        self._nodes[pos:pos] = nodes
        for node in nodes:
            node.owning_graph = self

    def _update_topology_with_delta(
        self, node, ori_inputs, ori_outputs, cur_inputs, cur_outputs
    ):
        """Update tensor producer/consumer relationship related to a modified node by input/output delta.

        Args:
            node (Node): The related node.
            ori_inputs (list<str>/set<str>): The original input list/set of the node. Should be empty for a newly added node.
            ori_outputs (list<str>/set<str>): The original output list/set of the node. Should be empty for a newly added node.
            cur_inputs (list<str>/set<str>): The current input list/set of the node. Should be empty for a deleted now.
            cur_outputs (list<str>/set<str>): The current output list/set of the node. Should be empty for a deleted now.
        """
        if not isinstance(node, Node) and node not in ["input", "output"]:
            logger.warning(
                "unexpected node value `{node}` when updating graph topology"
            )
        ori_inputs = set(ori_inputs)
        ori_outputs = set(ori_outputs)
        cur_inputs = set(cur_inputs)
        cur_outputs = set(cur_outputs)
        # the tensors not consumed by `node` anymore
        for tensor in ori_inputs - cur_inputs:
            assert tensor in self._tensor_consumer, f"bad graph topology"
            if node not in self._tensor_consumer[tensor]:
                logger.warning(
                    f"bad graph topology, {node.name} should be in {tensor}'s consumer list"
                )
            self._tensor_consumer[tensor].remove(node)
        # the tensors not produced by `node` anymore
        for tensor in ori_outputs - cur_outputs:
            assert tensor in self._tensor_producer, f"bad graph topology"
            if node not in self._tensor_producer[tensor]:
                logger.warning(
                    f"bad graph topology, {node.name} should be in {tensor}'s producer list"
                )
            self._tensor_producer[tensor].remove(node)
        # the tensors now consumed by `node` but not before
        for tensor in cur_inputs - ori_inputs:
            self._tensor_consumer.setdefault(tensor, [])
            if node in self._tensor_consumer[tensor]:
                logger.warning(
                    f"bad graph topology, {node.name} should no be in {tensor}'s consumer list"
                )
            self._tensor_consumer[tensor].append(node)
        # the tensors now produced by `node` but not before
        for tensor in cur_outputs - ori_outputs:
            self._tensor_producer.setdefault(tensor, [])
            if node in self._tensor_producer[tensor]:
                logger.warning(
                    f"bad graph topology, {node.name} should not be in {tensor}'s producer list'"
                )
            self._tensor_producer[tensor].append(node)

    def del_node(self, node):
        """delete node from self, graph topology will be updated accordingly.

        Args:
            node (Node): node to be deleted
        """
        self._nodes.remove(node)
        node.owning_graph = None
        self._update_topology_with_delta(node, node.input, node.output, [], [])

    def del_node_purely(self, node):
        """delete node from self, graph topology will **NOT** be updated.

        Args:
            node (Node): node to be deleted
        """
        self._nodes.remove(node)
        node.owning_graph = None

    @property
    def network_inputs(self):
        """`list<str>`: the network inputs (intializers excluded)"""
        return [x for x in self.input if x not in self.initializer]

    def add_input(self, inp):
        """add input by name, graph topology will be updated accordingly.

        Args:
            inp (str): input name
        """
        self._input.append(inp)
        self._add_producer(inp, "input")

    def add_output(self, output):
        """add output by name

        Args:
            output (str): output name
        """
        if output not in self._output:
            self._output.append(output)
        self._add_consumer(output, "output")

    def del_input(self, inp):
        """del input also initializer if needed by name

        Args:
            inp (str): input name

        Returns:
            bool: return True if `self` had `inp` and deleted `inp` else False
        """
        if inp in self.input:
            if inp in self.initializer:
                self.del_initializer(self.initializer[inp])
            else:
                self._input.remove(inp)
            if "input" in self._tensor_producer[inp]:
                self._tensor_producer[inp].remove("input")
            else:
                logger.warning("Graph.del_input called when topology not satisfied")
            return True
        return False

    def del_output(self, output):
        """del output by name

        Args:
            output: output name

        Returns:
            bool: return True if `self` had `output` and deleted `output` else False
        """
        if output in self.output:
            self._output.remove(output)
            if "output" in self._tensor_consumer[output]:
                self._tensor_consumer[output].remove("output")
            else:
                logger.warning("Graph.del_output called when topology not satisfied")
            return True
        return False

    @property
    def initializer(self):
        """`dict<str, onnx.TensorProto`: organize onnx.GraphProto.initializer as dict"""
        return self._initializer

    def add_initializer(self, init):
        """add initializer to this graph and update graph.input, if graph has any
        initializer with same name, it will be replaces directly

        Args:
            init (onnx.TensorProto): initializer to be added
        """
        assert init.name, "encountered an initilizer without name"
        if init.name not in self.input:
            self.add_input(init.name)
        self._initializer[init.name] = init

    def del_initializer(self, init):
        """delete initializer from this graph

        Args:
            init (onnx.TensorProto): initializer to be deleted
        """
        self.input.remove(init.name)
        del self._initializer[init.name]

    @property
    def tensor_shapes(self):
        """Get tensor shapes in this graph, modifying the returned dict will **NOT** affect self graph.

        Returns:
            dict<str, list<int>>: the tensor shapes.
        """
        return self._tensor_shape.copy()

    @property
    def tensor_dtypes(self):
        """Get tensor dtypes in this graph, modifying the returned dict will **NOT** affect this graph.

        Returns:
            dict<str, list<int>>: the tensor shapes.
        """
        return self._tensor_dtype.copy()

    def update_tensor_shape(self):
        """update all tensors' shape in this graph and store shape infomation in self.
        Performs as `infer_shape`
        """
        for k, v in self.initializer.items():
            # FIXME: support scalar.
            if len(v.dims) == 0:
                v.dims[:] = [1]
            self.set_tensor_shape(
                k, list(v.dims), Graph.get_nart_dtype_from_onnx(v.data_type)
            )
        for n in self.nodes:
            assert (
                n.owning_graph != None and n.owning_graph == self
            ), f"invalid node detected: {n}"
            n.infer_shape()

    def update_tensor_shape_with_preset(
        self, input_shape_dict, input_dtype_dict=dict()
    ):
        """update all tensors shpae in this graph with given input shape
        and store shape infomation in self. Performs as `infer_shape`
        """
        for k, v in input_shape_dict.items():
            dtype = input_dtype_dict.get(k, Dtype.Float32)
            self.set_tensor_shape(k, v, dtype=dtype)
        self.update_tensor_shape()

    def set_tensor_shape(self, tensor_name, shape, dtype=Dtype.Float32):
        """set tensor's shape and dtype with given shape and dtype

        Args:
            tensor_name (str): tensor's name
            shape (list): shape
        """
        self._tensor_dtype[tensor_name] = dtype
        self._tensor_shape[tensor_name] = [int(i) for i in shape]

    def get_tensor_shape(self, tensor_name):
        """acquire tensor's shape. Modifying the returned value will **NOT** affect the definition in self (graph).

        Args:
            tensor_name (str): tensor's name

        Returns:
            list: return shape of tensor_name if possible else empty list
        """
        return self._tensor_shape.get(tensor_name, []).copy()

    def get_tensor_dtype(self, tensor_name):
        """acquire tensor's dtype. Modifying the returned value will **NOT** affect the definition in self (graph).

        Args:
            tensor_name (str): tensor's name

        Returns:
            list: return dtype of tensor_name if possible else Dtype.Float32
        """
        return self._tensor_dtype.get(tensor_name, Dtype.Float32)

    def update_topology(self):
        """update topology info, renew tensor_producer and tensor_consumer"""
        # FIXME:
        #   lazy update, require graph modification info?
        self._tensor_producer.clear()
        self._tensor_consumer.clear()
        for i in self.input:
            self._tensor_producer[i] = ["input"]
        for o in self.output:
            self._tensor_consumer[o] = ["output"]

        for n in self.nodes:
            for i in n.input:
                if i == "":
                    continue
                if i not in self._tensor_consumer:
                    self._tensor_consumer[i] = []
                self._tensor_consumer[i].append(n)
            for i in n.output:
                if i not in self._tensor_producer:
                    self._tensor_producer[i] = []
                self._tensor_producer[i].append(n)

    def get_tensor_producer(self, tensor_name):
        """get tensor producer

        Args:
            tensor_name (str): tensor's name

        Returns:
            list: all nodes producing `tensor_name`, and 'input' means an input tensor
        """
        return list(self._tensor_producer.get(tensor_name, []))

    def get_tensor_consumer(self, tensor_name):
        """get tensor consumer

        Args:
            tensor_name: tensor's name

        Returns:
            list: all nodes using `tensor_name`, and 'outputs' means an output tensor
        """
        return list(self._tensor_consumer.get(tensor_name, []))

    def _add_producer(self, tensor_name, producer):
        self._tensor_producer.setdefault(tensor_name, [])
        if producer not in self._tensor_producer[tensor_name]:
            self._tensor_producer[tensor_name].append(producer)

    def _add_consumer(self, tensor_name, consumer):
        self._tensor_consumer.setdefault(tensor_name, [])
        if consumer not in self._tensor_consumer[tensor_name]:
            self._tensor_consumer[tensor_name].append(consumer)

    def __repr__(self):
        s = f"Graph({self._name})\n"
        s += "input:\n\t"
        s += "\n\t".join(self.input)
        s += "\n"
        s += "output:\n\t"
        s += "\n\t".join(self.output)
        s += "\n"
        s += "nodes:\n\t"
        s += "\n\t".join([str(n) for n in self.nodes])
        return s

    @staticmethod
    def from_onnx(graph):
        """create Graph from onnx.GraphProto

        Args:
            graph (onnx.GraphProto): onnx graph

        Returns:
            Graph: nart graph
        """
        res = Graph.make_graph(
            graph.name,
            {i.name: i for i in graph.initializer},
            [Node.from_onnx(n) for n in graph.node],
            {
                i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                for i in graph.input
            },
            {
                i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
                for i in graph.output
            },
            {
                i.name: Graph.get_nart_dtype_from_onnx(i.type.tensor_type.elem_type)
                for i in graph.input
            },
            {
                i.name: Graph.get_nart_dtype_from_onnx(i.type.tensor_type.elem_type)
                for i in graph.output
            },
            graph.doc_string,
        )
        return res

    @staticmethod
    def make_graph(
        name,
        initializer,
        nodes,
        input,
        output,
        input_dtype=dict(),
        output_dtype=dict(),
        doc_string="",
    ):
        """create Graph from given params

        Args:
            name (str): name
            initializer (dict<name, onnx.TensorProto>): initializer
            nodes (list): onnx.NodeProto or Node
            input (list): inputs
            output (list): output
            doc_string (str): doc string, defaults to empty str

        Returns:
            Graph: nart graph
        """
        graph = Graph()
        graph._name = name
        graph._initializer = initializer
        graph._nodes[:] = nodes
        graph._doc_string = doc_string

        for n in nodes:
            n.owning_graph = graph

        assert isinstance(input, (list, dict))
        assert isinstance(output, (list, dict))

        if isinstance(input, list):
            input.extend([x.name for x in initializer if x not in input])
            graph._input[:] = input
        else:
            # add initializers not included in input into input.
            for item in initializer.values():
                if item.name not in input:
                    shape = list(item.dims)
                    input[item.name] = shape

            graph._input[:] = list(input.keys())
            [
                graph.set_tensor_shape(t, s, input_dtype.get(t, Dtype.Float32))
                for t, s in input.items()
            ]
        if isinstance(output, list):
            graph._output[:] = output
        else:
            graph._output[:] = list(output.keys())
            [
                graph.set_tensor_shape(t, s, output_dtype.get(t, Dtype.Float32))
                for t, s in output.items()
            ]
        return graph

    def dump_to_onnx(self):
        """dump to onnx.GraphProto

        Returns:
            onnx.GraphProto: graph
        """
        graph = onnx.GraphProto()
        [graph.node.extend([x.dump_to_onnx()]) for x in self._nodes]
        graph.name = self._name
        graph.doc_string = self._doc_string
        graph.initializer.extend(list(self._initializer.values()))
        for ipt in self.input:
            if ipt in self._initializer:
                # if tensor is initializer, use the data type in initializer.
                dtype = self._initializer[ipt].data_type
            else:
                dtype = Graph.get_onnx_dtype_from_nart(self.get_tensor_dtype(ipt))
            graph.input.append(
                onnx.helper.make_tensor_value_info(
                    ipt, dtype, self.get_tensor_shape(ipt)
                )
            )
        [
            graph.output.extend(
                [
                    onnx.helper.make_tensor_value_info(
                        i,
                        Graph.get_onnx_dtype_from_nart(self.get_tensor_dtype(i)),
                        self.get_tensor_shape(i),
                    )
                ]
            )
            for i in self.output
        ]
        return graph

    @staticmethod
    def get_nart_dtype_from_onnx(onnx_dtype):
        """Get the fake tensor dtype of nart from tensor dtype of onnx

        Args:
            onnx.TensorProto.DataType: tensor dtype of onnx

        Returns:
            Dtype: fake tensor dtype of nart
        """
        maps = {
            onnx.TensorProto.FLOAT: Dtype.Float32,
            onnx.TensorProto.FLOAT16: Dtype.Float16,
            onnx.TensorProto.INT8: Dtype.Int8,
            onnx.TensorProto.UINT8: Dtype.Uint8,
            onnx.TensorProto.INT16: Dtype.Int16,
            onnx.TensorProto.UINT16: Dtype.Uint16,
            onnx.TensorProto.INT32: Dtype.Int32,
            onnx.TensorProto.UINT32: Dtype.Uint32,
            onnx.TensorProto.INT64: Dtype.Int64,
            onnx.TensorProto.UINT64: Dtype.Uint64,
        }
        return maps.get(onnx_dtype, Dtype.Float32)

    @staticmethod
    def get_onnx_dtype_from_nart(nart_dtype):
        """Get the tensor dtype of onnx from fake tensor dtype of nart

        Args:
            Dtype: fake tensor dtype of nart

        Returns:
            onnx.TensorProto.DataType: tensor dtype of onnx
        """
        maps = {
            Dtype.Float32: onnx.TensorProto.FLOAT,
            Dtype.Float16: onnx.TensorProto.FLOAT16,
            Dtype.Int8: onnx.TensorProto.INT8,
            Dtype.Uint8: onnx.TensorProto.UINT8,
            Dtype.Int16: onnx.TensorProto.INT16,
            Dtype.Uint16: onnx.TensorProto.UINT16,
            Dtype.Int32: onnx.TensorProto.INT32,
            Dtype.Uint32: onnx.TensorProto.UINT32,
            Dtype.Int64: onnx.TensorProto.INT64,
            Dtype.Uint64: onnx.TensorProto.UINT64,
        }
        return maps.get(nart_dtype, onnx.TensorProto.FLOAT)

    def get_const_tensor_as_array(self, name, allow_none=True):
        """Find a constant tensor with given name in graph, and convert it to numpy.array.

        Args:
            name (str): the desired tensor's name.
            allow_none (bool): if True, this method will return None when no such tensor found,
                otherwise, a exception will be raised under that condition.

        Returns:
            numpy.array: if a constant with the name found.
            None: otherwise.
        """

        def _find(graph):
            from ..ops.op import Constant

            if name in graph.initializer:
                from onnx import numpy_helper

                return numpy_helper.to_array(graph.initializer[name])
            producer = graph.get_tensor_producer(name)
            if len(producer) == 0:
                logger.warning(f"No tensor named {name} in graph {graph.name}")
                return None
            assert len(producer) == 1, "bad graph topology"
            producer = producer[0]
            if isinstance(producer, Constant):
                return producer.to_ndarray()
            try:
                # try to call the to_ndarray method of Op.
                return producer.to_ndarray()
            except Exception as ex:
                pass
            return None

        ret = _find(self)
        if not allow_none and ret is None:
            raise RuntimeError(
                f"No constant tensor named `{name}` found in graph `{self.name.name}`"
            )
        return ret

    def is_const_tensor(self, name):
        """Check whether a given tensor is constant.

        Args:
            name (str): the name of tensor to be checked.

        Returns:
            a boolean value to indicates whether the tensor is constant.
        """
        if name in self.initializer:
            return True
        producer = self.get_tensor_producer(name)
        if len(producer) == 0:
            logger.debug(f"No tensor named {name} in graph {self.name}")
            return False
        assert len(producer) == 1, "bad graph topology"
        producer = producer[0]
        if producer != "input" and producer.op_type == "Constant":
            return True
        return False


class Model(Graph):
    def __init__(self):
        super(Model, self).__init__()
        self._ir_version = 5
        self._producer_name = ""
        self._producer_version = ""
        self._domain = ""
        self._model_version = 1
        self._doc_string = ""
        self._graph = None
        self.opset = [("", 9)]

    @property
    def graph(self):
        return self._graph

    @staticmethod
    def from_onnx(model):
        """create model from onnx.ModelProto

        Args:
            model (onnx.ModelProto): onnx model

        Returns:
            Model: nart Model
        """
        res = Model.make_model(
            Graph.from_onnx(model.graph),
            model.ir_version,
            model.producer_name,
            model.producer_version,
            model.domain,
            model.model_version,
            model.doc_string,
        )
        return res

    @staticmethod
    def make_model(
        graph,
        ir_version=5,
        producer_name="",
        producer_version="",
        domain="",
        model_version=1,
        doc_string="",
        opset=None,
    ):
        model = Model()
        model._graph = graph
        model._ir_version = ir_version
        model._producer_name = producer_name
        model._producer_version = producer_version
        model._domain = domain
        model._model_version = model_version
        model._doc_string = doc_string
        if opset is None:
            from ..ops import Op

            # infer opset from graph.nodes
            # the maximum version required for each domain
            domain_versions = dict()
            for op in graph.nodes:
                assert isinstance(op, Op), "one of grpah.node is not Op"
                domain_versions.setdefault(op.domain, 1)
                domain_versions[op.domain] = max(domain_versions[op.domain], op.version)
            opset = [(domain, version) for domain, version in domain_versions.items()]
            logger.info("The detected opset is {0}".format(opset))
        model.opset = opset
        return model

    def dump_to_onnx(self):
        """dump to onnx.ModelProto

        Returns:
            onnx.ModelProto: model
        """
        model = onnx.ModelProto()
        model.graph.MergeFrom(self._graph.dump_to_onnx())
        model.ir_version = self._ir_version
        model.producer_name = self._producer_name
        model.producer_version = self._producer_version
        model.domain = self._domain
        model.model_version = self._model_version
        model.doc_string = self._doc_string
        for opset in self.opset:
            added_opset = model.opset_import.add()
            added_opset.domain = opset[0]
            added_opset.version = opset[1]
        return model
