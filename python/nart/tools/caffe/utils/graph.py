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

from nart.tools.proto import caffe_pb2
import google.protobuf.text_format


def readNetStructure(filePathA=None, filePathB=None):
    withBinFile = True
    net = caffe_pb2.NetParameter()
    if filePathA == None and filePathB == None:
        raise ValueError
    elif filePathA == None or filePathB == None:
        netPath = filePathB if filePathA == None else filePathA
        try:
            with open(netPath, "rb") as f:
                net.ParseFromString(f.read())
        except:
            try:
                with open(netPath) as f:
                    net.Clear()
                    google.protobuf.text_format.Merge(f.read(), net)
                    withBinFile = False
            except:
                raise ValueError
    else:
        netParam = caffe_pb2.NetParameter()
        try:
            with open(filePathA) as f:
                google.protobuf.text_format.Merge(f.read(), net)
            with open(filePathB, "rb") as f:
                netParam.ParseFromString(f.read())
        except:
            with open(filePathB) as f:
                net.Clear()
                google.protobuf.text_format.Merge(f.read(), net)
            with open(filePathA, "rb") as f:
                netParam.Clear()
                netParam.ParseFromString(f.read())
        for i in net.layer:
            for j in netParam.layer:
                if j.name == i.name:
                    i.blobs.MergeFrom(j.blobs)
    return net, withBinFile


class Node:
    def __init__(self, content=None, prev=None, succ=None):
        self.content = content
        self.prev = [] if prev is None else prev
        self.succ = [] if succ is None else succ
        return


class Graph:
    def __init__(self, root=None, leaf=None):
        self.root = [] if root is None else root
        self.leaf = [] if leaf is None else leaf
        return

    def root(self):
        return self.root

    def leaf(self):
        return self.leaf

    def nodes(self, filter_func=lambda x: True):
        visited = set()
        node_list = list(set(self.root))
        while len(node_list) > 0:
            node = node_list.pop(0)
            if node in visited or len(visited & set(node.prev)) != len(node.prev):
                continue
            visited.add(node)
            yield node
            for n in node.succ:
                if n not in visited:
                    node_list.append(n)

    def reverse_nodes(self, filter_func=lambda x: True):
        visited = set()
        node_list = list(set(self.leaf))
        while len(node_list) > 0:
            node = node_list.pop(0)
            if node in visited:
                continue
            visited.add(node)
            yield node
            for n in node.prev:
                if n not in visited:
                    node_list.append(n)

    def update(self):
        root_ = set()
        leaf_ = set()

        for node in self.nodes():
            if len(node.prev) == 0:
                root_.add(node)
            if len(node.succ) == 0:
                leaf_.add(node)
        self.root = list(root_)
        self.leaf = list(leaf_)


def gen_graph(proto):
    graph = Graph()
    for layer in proto.layer:
        node = Node(content=layer)
        for bot in node.content.bottom:
            prev = None
            for n in graph.leaf:
                if bot in n.content.top:
                    prev = n
            if prev != None and prev not in node.prev:
                node.prev.append(prev)
        graph.leaf.append(node)

    for n in graph.reverse_nodes():
        graph.root.append(n)
        for prev in n.prev:
            if n not in prev.succ:
                prev.succ.append(n)
    graph.update()
    return graph
