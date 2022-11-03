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

from ...core import Node
from .caffe_to_onnx import CAFFE_TO_ONNX
from .. import read_net_def
from ...core.art import FakeParade
from ...proto import caffe_pb2

import warnings
import logging

logger = logging.getLogger("nart.utils.caffe_builder")
import sys
import tempfile
import os
import numpy as np

from onnx import GraphProto
from onnx import NodeProto
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper


def get_input_shape(net_def, input_shape_dict):
    temp_input_size = {k: None for k in net_def.input}
    if len(net_def.input_shape) != 0:
        for index in range(len(net_def.input)):
            temp_input_size[net_def.input[index]] = list(net_def.input_shape[index].dim)
    elif len(net_def.input_dim) != 0:
        for index in range(len(net_def.input)):
            temp_input_size[net_def.input[index]] = list(
                net_def.input_dim[index * 4 : index * 4 + 4]
            )
            # TODO: Support more type of net, current supported type: [input dim: 4]
    elif net_def.layer[0].type == "Input":
        for l in net_def.layer:
            if l.type == "Input":
                temp_input_size[l.top[0]] = list(l.input_param.shape[0].dim)
    else:
        raise TypeError("Unsupported network input format.")

    for k, v in input_shape_dict.items():
        if k not in temp_input_size:
            raise KeyError("'%s' is not an input of the net." % (k))
        temp_input_size[k] = list(v)
    for k, v in temp_input_size.items():
        if v is None:
            raise TypeError("shape of input '%s' is not specified." % (k))
    return temp_input_size


class CaffeEncoder:
    """change onnx to caffe netdef."""

    def __init__(self, graph):
        self.graph = graph
        self.caffe_net = caffe_pb2.NetParameter(name=graph.name)
        self._prepare = False
        self.index = 0

    def encode(self):
        """call encode to change the onnx net.

        Returns:
            caffe_pb2.NetParameter.
        """
        if self._prepare == True:
            pass
        else:
            self.parse_blobs()
            self.parse_input()
            self.parse_layers()
            self._prepare = True
        return self.caffe_net

    def parse_blobs(self):
        """parse blobs(weight)."""
        self.blobs = dict()
        for n, init in self.graph.initializer.items():
            self.blobs[n] = numpy_helper.to_array(init)

        self.blobshape = dict()
        for inp in self.graph.input:
            if inp not in self.blobs.keys():
                self.blobshape[inp] = self.graph.get_tensor_shape(inp)

    def parse_input(self):
        """parse input.

        input not include weight.
        """
        from ...proto import caffe_pb2

        inps = [inp for inp in self.graph.input if inp not in self.blobs.keys()]
        self.caffe_net.input.extend(inps)
        for inp in inps:
            inp_shape = caffe_pb2.BlobShape()
            inp_shape.dim.extend(self.graph.get_tensor_shape(inp))
            self.caffe_net.input_shape.extend([inp_shape])

        for inp in self.graph.initializer:
            self.blobshape[inp] = self.graph.get_tensor_shape(inp)

        for n in self.graph.nodes:
            for o in n.output:
                self.blobshape[o] = self.graph.get_tensor_shape(o)

    def parse_layers(self):
        """parse layers"""
        for node in self.graph.nodes:
            # layers = make_layers(node, index, self)
            layers = self.parse_layer(node)
            for layer in layers:
                self.caffe_net.layer.extend([layer.params])

    def parse_layer(self, node):
        self.index = self.index + 1
        from ...tools.pytorch.layer_utils import make_layers
        from .onnx_to_caffe import own_convert_dict
        from ...ops import GroupType

        if node.__class__.GROUP == GroupType.ONNX:
            # from onnx
            return make_layers(node.dump_to_onnx(), self.index, self)
        elif node.__class__.GROUP == GroupType.TOOLS:
            # from tools
            return make_layers(node.dump_to_onnx(), self.index, self)
        elif node.__class__.GROUP == GroupType.CASE:
            # from onnx
            layer = own_convert_dict[node.op_type](
                node.dump_to_onnx(), self.index, self
            )
            return [
                layer,
            ]
        else:
            print(node.__class__.GROUP)
            raise RuntimeError("unknown op group")


def de_inplace(netdef):
    """Remove inplace layer in netdef

    Returns:
        the modified netdef
    """
    # 每个blob被写入的总次数
    total_write = dict()
    for layer in netdef.layer:
        for top in layer.top:
            total_write.setdefault(top, 0)
            total_write[top] += 1
    # 一个blob名字已经被用作输出多少次了(写入次数)
    version = dict()
    for name in netdef.input:
        total_write[name] = 0
        version[name] = 0

    def new_name(name):
        if version[name] == total_write[name]:
            # this blob has already been writen into for the last time.
            return name
        else:
            return f"{name}_ver{version[name]}"

    for layer in netdef.layer:
        # first replace the input names
        for i in range(len(layer.bottom)):
            layer.bottom[i] = new_name(layer.bottom[i])
        # record output version
        for name in layer.top:
            version.setdefault(name, 0)
            version[name] += 1
        # replace output names
        for i in range(len(layer.top)):
            layer.top[i] = new_name(layer.top[i])
        pass
    return netdef


def canonicalize_name(name: str):
    import re

    p = re.compile("[^a-zA-Z0-9\._-]")
    name = p.sub("_", name)
    return name[:64]


class CaffeDecoder:
    """Convert caffe net to ``core.graph.Graph``"""

    def __init__(self):
        pass

    def decode(self, netdef, weight, config={}):
        """Convert caffe net to core.graph.Graph, which is a ONNX-like format.

        **NOTE**: if the caffe net have inplace operations, they will be converted to non-inplace operation,
        during the process some blobs' name will be changed.

        Args:
            netdef (caffe_pb2.NetParameter): caffe network definition.
            weight (caffe_pb2.NetParameter): caffe model which contains model weights.
            config (dict<str, list<int>>): convert configuration.
                ``config['input_shape']`` is the input shape configuration, which take precendence over the definition in netdef.

        Returns:
            core.graph.Graph: the Graph converted from caffe net.
        """
        netdef = de_inplace(netdef)

        # the input/output/nodes/initializer of the new graph
        initializer = {}
        nodes = []
        input = {}
        output = {}

        from ...proto import caffe_pb2

        assert isinstance(netdef, caffe_pb2.NetParameter)
        assert isinstance(weight, caffe_pb2.NetParameter)
        assert isinstance(config, dict)

        # get input_shape_dict
        if "input_shape" in config:
            input_shape_dict = config["input_shape"]
        else:
            input_shape_dict = {}
        input_shape_dict = get_input_shape(netdef, input_shape_dict)
        tensor_shape_by_name = {
            name: np.array(shape, dtype=np.int32)
            for name, shape in input_shape_dict.items()
        }

        # Merge netdef and weight
        import copy

        temp_proto = copy.deepcopy(netdef)
        for i in temp_proto.layer:
            for j in weight.layer:
                if i.name == j.name:
                    i.ClearField("blobs")
                    i.blobs.MergeFrom(j.blobs)
                    break
        netdef = temp_proto

        for name in tensor_shape_by_name:
            shape = tensor_shape_by_name[name].tolist()
            input_tensor_info = helper.make_tensor_value_info(
                name, TensorProto.FLOAT, shape
            )
            input[name] = tensor_shape_by_name[name]

        from ...tools.caffe.count import type2class

        for layer in netdef.layer:

            if layer.type == "Input":
                continue

            layer_input_shape = {}
            # find input_shape
            for i in layer.bottom:
                if i in tensor_shape_by_name:
                    layer_input_shape[i] = tensor_shape_by_name[i]
                else:
                    raise RuntimeError("can not get input tensor shape")

            # new node
            if layer.type not in CAFFE_TO_ONNX:
                logger.warn(f"unhandled layer type {layer.type} in caffe builder")
                to_node = CAFFE_TO_ONNX["Default"](layer, layer_input_shape)
            else:
                to_node = CAFFE_TO_ONNX[layer.type](layer, layer_input_shape)
            new_node = to_node.trans()
            # get param
            for i in to_node.weight:
                info = to_node.weight[i][0]
                shape = info.type.tensor_type.shape
                shape = [x.dim_value for x in shape.dim]
                input[i] = shape
                initializer[i] = to_node.weight[i][1]

            if isinstance(new_node, (list, tuple)):
                for node in new_node:
                    nodes.append(node)
            else:
                nodes.append(new_node)

            # infer shape on the caffe layer
            shape_inferer = type2class[layer.type](layer)
            # collect the bottom blobs' shape of current layer
            bottom_shapes = [tensor_shape_by_name[name] for name in layer.bottom]
            assert all(isinstance(x, np.ndarray) for x in bottom_shapes)
            # bottom_shapes = np.stack(bottom_shapes, axis=0)  # shape_inferer.infer need a numpy.array as parameter
            shape_inferer.infer(bottom_shapes)
            assert isinstance(shape_inferer.topShape, np.ndarray)
            # add the shape of top blobs of current layer
            tensor_shape_by_name.update(
                {
                    name: shape
                    for name, shape in zip(layer.top, list(shape_inferer.topShape))
                }
            )

        # convert type of shapes into lists. (the infered shape may be of type numpy.ndarray)
        for name, shape in tensor_shape_by_name.items():
            tensor_shape_by_name[name] = list(shape)

        # change numpy.int64 to int
        # helper.make_tensor_value_info will raise error if shape type is numpy.int64
        for shape in tensor_shape_by_name.values():
            for ind in range(len(shape)):
                shape[ind] = int(shape[ind])

        # add dag.output
        # collect outputs from the caffe netdef.
        out_candidates = set(input_shape_dict.keys())
        for layer in netdef.layer:
            for name in layer.bottom:
                if name in out_candidates:
                    out_candidates.remove(name)
            for name in layer.top:
                out_candidates.add(name)

        for name in out_candidates:
            output_tensor_info = helper.make_tensor_value_info(
                name, TensorProto.FLOAT, tensor_shape_by_name[name]
            )
            output[name] = tensor_shape_by_name[name]

        # create the new graph
        from nart.core.graph import Graph

        graph = Graph.make_graph(
            canonicalize_name(netdef.name), initializer, nodes, input, output
        )

        return graph

    def load(self, netdef_path, weight_path, config={}):
        """Load caffe net from file, and then call self.decode to convert it to Graph.

        Args:
            netdef_path (str): file path of caffe prototxt .
            weight_path (str): file path of caffe model.
            config (dict): convert configuration, see self.decode

        Returns:
            core.graph.Graph: the Graph converted from caffe net.
        """
        netdef = read_net_def(netdef_path)
        weight = caffe_pb2.NetParameter()
        with open(weight_path, "rb") as f:
            weight.ParseFromString(f.read())
        return self.decode(netdef, weight, config)
