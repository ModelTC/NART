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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
import copy
import onnx
import multiprocessing
import argparse
import json
import yaml
import logging

import nart
from nart.modules import NetParser
from nart.modules import CaffeParser
from nart.modules import CaffeCUDAParser
from nart.modules import TensorrtParser
from nart.modules import CaffeQuantParser
from nart.modules import CaffeCUDAQuantParser

from nart.utils.caffe_utils import CaffeDecoder
from nart.utils.onnx_utils import OnnxDecoder
from nart.proto import caffe_pb2
from nart.utils import read_net_def
from nart.calibrator import ClbConfig
from nart.calibrator import calibrate
from nart.passes import (
    DeadCodeElimination,
    ConstantToInitializer,
    EliminateNodes,
    UnsqueezeOutputTo4D,
)
from nart.ops.op import Dropout, Cast, Identity

import logging

logger = logging.getLogger("nart.switch")

MODULE_DICT = {
    "quant": {"quant": CaffeQuantParser, "default": CaffeParser},
    "cuda_quant": {
        "cuda_quant": CaffeCUDAQuantParser,
        "cuda": CaffeCUDAParser,
        "default": CaffeParser,
    },
    "tensorrt": {
        "tensorrt": TensorrtParser,
        "cuda": CaffeCUDAParser,
        "default": CaffeParser,
    },
    "cuda": {"cuda": CaffeCUDAParser, "default": CaffeParser},
    "default": {"default": CaffeParser},
}

PARSER_DICT = {
    "default": CaffeParser,
    "cuda": CaffeCUDAParser,
    "tensorrt": TensorrtParser,
    "quant": CaffeQuantParser,
    "cuda_quant": CaffeCUDAQuantParser,
}

ALWAYS_4D_OUTPUT_TOKEN = "always_4D_output"


def gen_net_parser(args):
    config = {"data_num": 0, "rand_input": True}

    # update config
    if args.config:
        with open(args.config) as f:
            custom_config = yaml.load(f, Loader=yaml.FullLoader)
            config.update(custom_config)

    if args.max_batch_size > 0:
        config["max_batch_size"] = args.max_batch_size
    if "dump_info" not in config:
        config["dump_info"] = args.dump_raw_model
    config["output_engine"] = args.output
    # registe module
    NetParser.register_parser(MODULE_DICT[args.type])
    for parser in MODULE_DICT[args.type].values():
        parser.register_defaults()
    # update config
    if "default_net_type_token" not in config:
        config["default_net_type_token"] = list(MODULE_DICT[args.type].keys())[0]
    if args.type == "quant":
        q_config = ClbConfig.get_config()

        def get_prop(dct, key, dflt):
            if key in dct:
                return dct[key]
            return dflt

        if "quant" not in config:
            config["quant"] = {}
        q_config.use_gpu = get_prop(config["quant"], "use_gpu", False)
        q_config.device = get_prop(config["quant"], "device", 0)
        q_config.input_dir = get_prop(config["quant"], "input_dir", "")
        q_config.symmetric = get_prop(config["quant"], "symmetric", False)
        q_config.kl_divergence = get_prop(config["quant"], "kl_divergence", False)
        q_config.bit_width = get_prop(config["quant"], "bit_width", 8)
        q_config.output = get_prop(config["quant"], "output", "max_min.json")
        q_config.data_num = config["data_num"]

        if q_config.input_dir == "" or q_config.data_num == 0:
            quant_params = {}
        else:
            quant_params = calibrate(args.prototxt, args.model, q_config)
        config["quant"] = {"quant_param": quant_params}

        if args.verbose:
            with open(q_config.output, "wt") as f:
                json.dump(quant_params, f, indent=4)
    elif args.type == "cuda_quant":
        q_config = ClbConfig.get_config()

        def get_prop(dct, key, dflt):
            if key in dct:
                return dct[key]
            return dflt

        if "cuda_quant" not in config:
            config["cuda_quant"] = {}
        q_config.use_gpu = get_prop(config["cuda_quant"], "use_gpu", False)
        q_config.device = get_prop(config["cuda_quant"], "device", 0)
        q_config.input_dir = get_prop(config["cuda_quant"], "input_dir", "")
        q_config.symmetric = get_prop(config["cuda_quant"], "symmetric", False)
        q_config.kl_divergence = get_prop(config["cuda_quant"], "kl_divergence", False)
        q_config.bit_width = get_prop(config["cuda_quant"], "bit_width", 8)
        q_config.output = get_prop(config["cuda_quant"], "output", "max_min.json")
        q_config.data_num = config["data_num"]
        q_config.qparams_file = get_prop(config["cuda_quant"], "qparams_file", "")

        # todo: check this
        # if (q_config.input_dir == '' or q_config.data_num == 0):
        #     quant_params = {}
        # else:
        #     quant_params = calibrate(args.prototxt, args.model, q_config)
        quant_params = {}
        if q_config.qparams_file != "":
            with open(q_config.qparams_file, "r") as f:
                quant_params = yaml.load(f, Loader=yaml.FullLoader)
        config["cuda_quant"] = {"quant_param": quant_params}

        if args.verbose:
            with open(q_config.output, "wt") as f:
                json.dump(quant_params, f, indent=4)

    if not args.onnx:
        netdef = read_net_def(args.prototxt)
        netweight = copy.deepcopy(netdef)
        tmp_netweight = caffe_pb2.NetParameter()
        with open(args.model, "rb") as f:
            tmp_netweight.ParseFromString(f.read())
        for l in netweight.layer:
            for w in tmp_netweight.layer:
                if w.name == l.name:
                    l.blobs.MergeFrom(w.blobs)
                    break
        logger.info("received caffe model, building Graph...")
        graph = CaffeDecoder().decode(netdef, netweight, config)
    else:
        logger.info("received onnx model, building Graph...")
        graph = OnnxDecoder().decode(onnx.load(args.onnx))

    if "output_names" in config:
        # add extra outputs to graph.output
        extra_outputs = config["output_names"]
        assert all(
            isinstance(x, str) for x in extra_outputs
        ), "output names must be string"
        graph.output.extend([x for x in extra_outputs if x not in graph.output])
    if "outputs" in config:
        # override output completely.
        outputs = config["outputs"]
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        assert all(isinstance(x, str) for x in outputs), "output names must be string"
        graph.output[:] = outputs
    graph.update_topology()
    graph.update_tensor_shape()

    logger.info(f"\n{graph}")

    passes = []
    if config.get(ALWAYS_4D_OUTPUT_TOKEN, False) == True:
        passes.append(UnsqueezeOutputTo4D())

    # backend specific passes
    passes.extend(PARSER_DICT[args.type].get_passes())

    for item in passes:
        item.run(graph)
    graph.update_topology()
    graph.update_tensor_shape()

    logger.info(f"\nafter running backend specified passes: \n{graph}")
    return NetParser(graph, config)


def mark_output(parade, graph):
    # TODO: this function is buggy, when input model is onnx, the graph.output is not marked.
    # mark_output
    tensor_names = list(graph.output)
    parade.mark_as_output_by_name(tensor_names)


if __name__ == "__main__":
    # set multiprocessing start method as 'spawn', this is required to load another caffe_pb2 in subprocess
    # when protobuf api_implementation type is 'cpp'.
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", nargs="?", help="config.json", default=None)
    parser.add_argument(
        "-t",
        "--type",
        default="default",
        choices=["default", "cuda", "tensorrt", "quant", "cuda_quant"],
    )
    parser.add_argument("-o", "--output", default="engine.bin", type=str)
    parser.add_argument("-b", "--max_batch_size", default=-1, type=int)
    parser.add_argument("-d", "--dump_raw_model", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument(
        "--version", action="version", version=f"nart.switch v{nart.__version__}"
    )
    parser.add_argument("--prototxt", nargs="?", help="source net prototxt", type=str)
    parser.add_argument("--model", nargs="?", help="source net weight", type=str)
    parser.add_argument("--onnx", nargs="?", help="source onnx model", type=str)
    args = parser.parse_args()

    assert args.onnx or (
        args.prototxt and args.model
    ), "caffe model or onnx model required."

    if args.verbose:
        nart_logger = logging.getLogger("nart")
        nart_logger.setLevel(logging.DEBUG)
    logger.info(f"VERSION: nart v{nart.__version__}")
    net_parser = gen_net_parser(args)
    parade = net_parser.parse()
    mark_output(parade, net_parser.ori_net)

    logger.info(f"serializing to {args.output} ...")
    ser_data = parade.serialize()
    with open(args.output, "wb") as f:
        f.write(ser_data)

    logger.info(f'generating runtime-config to {args.output + ".json"} ...')
    nart_config = net_parser.gen_nart_config(args, MODULE_DICT)
    with open(args.output + ".json", "wt") as f:
        json.dump(nart_config, f, indent=4)
