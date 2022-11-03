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

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from nart.modules import NetParser

from nart.modules import CaffeParser
from nart.modules import CaffeCUDAParser
from nart.core.art import Dtype
from nart import *
import nart.core.art
import argparse
import json
from netrt import mixnet_pb2
from caffe.proto import caffe_pb2
import tempfile


class net_parsers:
    def __init__(self, parsers, TryOutoutNames):
        self.parsers = parsers
        self.TryOutoutNames = list(TryOutoutNames)

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        for parser in self.parsers:
            into_parade = parser.parse(into_parade)
        into_parade.mark_as_output_by_name(self.TryOutoutNames)
        return into_parade


def gen_net_parsers(args):
    mixnet = mixnet_pb2.MixNetProto()
    with open(args.model, "rb") as f:
        try:
            mixnet.ParseFromString(f.read())
        except:
            print("not a mixnet")
    InputNames = list(mixnet.InputNames)
    OutputNames = list(mixnet.OutputNames)
    TryOutoutNames = list(mixnet.TryOutputNames)
    MaxBatch = mixnet.MaxBatch
    parser_list = []
    CaffeParser.register_defaults()
    # init config
    config = {"data_num": 1, "rand_input": True}
    config["max_batch_size"] = MaxBatch
    CaffeCUDAParser.register_defaults()
    NetParser.register_parser({"cuda": CaffeCUDAParser, "default": CaffeParser})

    if mixnet is None or len(mixnet.subnet) == 0:
        # try tensorrt to solve it
        print("try to parse by tensorrt parser")
        with open(args.model, "rb") as f:
            content = f.read()
        Trtnet_config = config.copy()
        tensorrt_parser = _TensorrtParser(None, None, content, Trtnet_config)
        parser_list.append(tensorrt_parser)
        return net_parsers(parser_list, TryOutoutNames)

    for subnet in mixnet.subnet:
        if subnet.subnet_type == 0:
            # TensorRTNet
            Trtnet_config = config.copy()
            tensorrt_parser = _TensorrtParser(None, None, subnet.content, Trtnet_config)

            parser_list.append(tensorrt_parser)

        elif subnet.subnet_type == 1:
            # CaffeNet
            content = subnet.content
            caffe_net = mixnet_pb2.CaffeNetProto()
            caffe_net.ParseFromString(subnet.content)
            assert MaxBatch == caffe_net.MaxBatch
            caffe_InputNames = caffe_net.InputNames
            caffe_OutputNames = caffe_net.OutputNames
            NetParameter = caffe_net.NetDef
            caffe_NetParameter = caffe_pb2.NetParameter()
            caffe_NetParameter.ParseFromString(NetParameter.SerializeToString())
            NetParameter = caffe_NetParameter
            # add MaxBatch to dims
            # no need to deal with input_shape (before_parse do the work)
            input_dim = []
            for index in range(len(caffe_net.Dims)):
                if index % 3 == 0:
                    input_dim.append(MaxBatch)
                input_dim.append(caffe_net.Dims[index])
            # NetParameter.input_dim.MergeFrom(input_dim)
            NetParameter.input_dim[:] = input_dim
            assert len(NetParameter.input_dim) == len(NetParameter.input) * 4
            # set net_config
            tmp_net_config = config.copy()
            tmp_net_config["default_net_type_token"] = "cuda"
            net_config = {"output_names": list(caffe_OutputNames)}
            net_config.update(tmp_net_config)

            import copy

            net_def = copy.deepcopy(NetParameter)
            for i in net_def.layer:
                i.ClearField("blobs")

            caffe_parser = NetParser(net_def, NetParameter, net_config)

            # add to the parser_list
            parser_list.append(caffe_parser)

        elif subnet.subnet_type == 2:
            # MixNet
            raise
        else:
            raise
    return net_parsers(parser_list, TryOutoutNames)


def gen_nart_config(args):
    res = {"workspaces": {"default": {}}, "input_workspace": "default"}
    res["workspaces"]["tensorrt"] = {}
    if args.type == "cuda":
        res["workspaces"]["cuda"] = {}
        res["input_workspace"] = "cuda"
    return res


class _TensorrtParser:
    def __init__(self, netdef, netweight, tensorrt_net, config={}):
        self.netdef = netdef
        self.netweight = netweight

        from netrt import _netrt

        self.tensorrt_net_string = tensorrt_net
        self.tensorrt_net = _netrt.deserialize(tensorrt_net)
        self.config = {"output_names": []}

        self.config.update(config)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.extend_outputs_ = self.config["output_names"]
        self.max_batch_size = self.config["max_batch_size"]

        for name in self.extend_outputs_:
            self.tensorrt_net.addOutput(name)
        assert self.tensorrt_net.prepareToRun()

    def __del__(self):
        self.temp_dir.cleanup()

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        into_parade_outputs = {}

        for t in into_parade.tensors:
            if t.dtype == "float32":
                into_parade_outputs[t.name] = t

        # name to tensor
        dct_name_tensor = self.parse_input()
        # dct_name_tensor.update(into_parade_outputs)
        dct_name_tensor.update(
            {
                x: into_parade_outputs[x]
                for x in dct_name_tensor
                if x in into_parade_outputs
            }
        )
        output_dict = self.parse_output()
        x = into_parade.append(
            Fakes.TENSORRTNet(
                list(map(lambda x: dct_name_tensor[x], dct_name_tensor)),
                self.tensorrt_net_string,
                list(map(lambda x: output_dict[x], output_dict)),
            )
        )
        # set name
        for t, name in zip(x, output_dict.keys()):
            t.name = name
        return into_parade

    def parse_input(self):
        # ret dict (name:tensor)
        input_dict = {}
        for k, v in self.tensorrt_net.getInputBinding().items():
            tensor = FakeTensor(Dtype.Float32, shape=(list(v.shape), 1, 0), name=k)
            tensor.name = k
            input_dict[k] = tensor
        return input_dict

    def parse_output(self):
        # ret dict (name:tensor)
        output_dict = {}
        for k, v in self.tensorrt_net.getOutputBinding().items():
            tensor = FakeTensor(Dtype.Float32, shape=(list(v.shape), 1, 0), name=k)
            tensor.name = k
            output_dict[k] = tensor
        return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="engine.bin", type=str)
    parser.add_argument("model", help="mixnet model", default="model.bin", type=str)
    parser.add_argument("-t", "--type", default="cuda", choices=["default", "cuda"])
    args = parser.parse_args()

    net_parsers = gen_net_parsers(args)

    parade = net_parsers.parse()
    ser_data = serialize_v1(parade)
    with open(args.output, "wb") as f:
        f.write(ser_data)

    # how to set config
    nart_config = gen_nart_config(args)
    with open(args.output + ".json", "wt") as f:
        json.dump(nart_config, f, indent=4)
