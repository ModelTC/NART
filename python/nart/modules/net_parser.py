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

# -*- coding: utf-8 -*-
from ..core.art import FakeParade
from ..core.art import FakeTensor
from ..core.art import Dtype
from ..core import Net
import inspect
import numpy
import tempfile
import os
import warnings
from ..utils.caffe_utils import CaffeDecoder
from ..utils.onnx_utils import OnnxDecoder
from ..utils import Splitter
from ..utils import DataGenerator, PerFrameDataGenerator

from .default import CaffeParser
import logging

logger = logging.getLogger("nart.modules.netparser")

data_num_token = "data_num"
input_path_map_token = "input_path_map"
default_net_type_token = "default_net_type_token"
set_net_type_token = "set_net_type"


class NetParser:
    parser = {}

    @classmethod
    def register_parser(cls, parser_map):
        for name in parser_map:
            cls.parser[name] = parser_map[name]

    def __init__(self, graph, config={}):
        self.ori_net = graph
        self.config = {"max_batch_size": -1, data_num_token: 0}
        self.config.update(config)
        self.max_batch_size = self.config["max_batch_size"]
        self.temp_dir = tempfile.TemporaryDirectory()

        if data_num_token in self.config:
            self.data_num = self.config[data_num_token]
            assert self.data_num >= 0
        self.name_to_support_layer_list = {}
        self.input_data_map = {}
        self.id_to_name = {0: "default"}
        self.datagen_by_net_type = dict()
        self._used_workspace = []
        self._pre()

    def _pre(self):
        if self.data_num > 0 and input_path_map_token in self.config:
            input_path_map = self.config[input_path_map_token]
            for name in self.ori_net.input:
                if name in self.ori_net.initializer:
                    continue
                if name not in input_path_map:
                    logger.warning(
                        f"data for input `{name}` not provided, "
                        "which may cause errors when doing calibration"
                    )
                    continue
                output_path = input_path_map[name]
                self.input_data_map[name] = []
                for i in range(self.data_num):
                    self.input_data_map[name].append(f"{output_path}/{i}.bin")
        if default_net_type_token in self.config:
            default_type_name = self.config[default_net_type_token]
            self.id_to_name[0] = default_type_name
            _id = 1
            for name in NetParser.parser:
                if name != default_type_name:
                    self.id_to_name[_id] = name
                    _id += 1

        graph = self.ori_net
        # ====== fill missing shape configs ====== #
        original_shapes = {x: graph.get_tensor_shape(x) for x in graph.network_inputs}

        def set_defaults(config, cfg_key, defaults):
            original = config.setdefault(cfg_key, dict())
            for name, value in defaults.items():
                original.setdefault(name, value)

        if self.max_batch_size > 0:
            logger.info(
                "using max_batch_size config to set min/build/max shape of all input tensors"
            )
            # max_batch_size specified, use it to set missing shape configs.

            def apply_batch_size(shapes, batch_size):
                ret = dict()
                for name, shape in shapes.items():
                    shape = shape.copy()
                    shape[0] = batch_size
                    ret[name] = shape
                return ret

            set_defaults(
                self.config, "min_shapes", apply_batch_size(original_shapes, 1)
            )
            set_defaults(
                self.config,
                "max_shapes",
                apply_batch_size(original_shapes, self.max_batch_size),
            )
            set_defaults(
                self.config,
                "build_shapes",
                apply_batch_size(original_shapes, self.max_batch_size),
            )
        else:
            # no max_batch_size config given, fill missing shape configs with `original_shapes`
            set_defaults(self.config, "min_shapes", original_shapes)
            set_defaults(self.config, "max_shapes", original_shapes)
            set_defaults(self.config, "build_shapes", original_shapes)
        from copy import deepcopy

        # the default calibration shape is the shape in prototxt, with batch_size altered to 1.
        default_calib_shapes = deepcopy(original_shapes)
        for item in default_calib_shapes.values():
            item[0] = 1
        set_defaults(self.config, "calib_shapes", default_calib_shapes)
        # ====== validate shape configs: min<=build<=max ====== #

        def is_shape_less_equal(a, b):
            assert len(a) == len(
                b
            ), "shapes having different number of dimension cannot be compared"
            return all(x <= y for x, y in zip(a, b))

        for input in graph.network_inputs:
            min_shape = self.config["min_shapes"][input]
            build_shape = self.config["build_shapes"][input]
            if not is_shape_less_equal(min_shape, build_shape):
                raise RuntimeError(
                    f"the min/build shapes of input `{input}` are incompatible: {min_shape} vs {build_shape}"
                )
            max_shape = self.config["max_shapes"][input]
            if not is_shape_less_equal(build_shape, max_shape):
                raise RuntimeError(
                    f"the build/max shapes of input `{input}` are incompatible: {build_shape} vs {max_shape}"
                )
        # ====== infer typical shapes of all tensors in graph ====== #
        dtype_dict = graph.tensor_dtypes
        try:
            # try to infer calib_shapes for intermediate tensors, this may fail because the
            # default calib_shapes may be invalid.
            graph.update_tensor_shape_with_preset(
                self.config["calib_shapes"], dtype_dict
            )
            self.calib_shapes = deepcopy(graph.tensor_shapes)
        except:
            logger.warning("failed to infer calib_shapes")
            self.calib_shapes = dict()
        graph.update_tensor_shape_with_preset(self.config["min_shapes"], dtype_dict)
        self.min_shapes = graph.tensor_shapes
        graph.update_tensor_shape_with_preset(self.config["max_shapes"], dtype_dict)
        self.max_shapes = graph.tensor_shapes
        # finally, infer shape with config['build_shapes], because that's the default shape on which the model should be built.
        graph.update_tensor_shape_with_preset(self.config["build_shapes"], dtype_dict)
        self.build_shapes = graph.tensor_shapes

        set_net_type = {}
        if set_net_type_token in self.config:
            set_net_type = (
                self.config[set_net_type_token][graph.name].copy()
                if graph.name in self.config[set_net_type_token]
                else self.config[set_net_type_token].copy()
            )

        for node in self.ori_net.nodes:
            if node.name in set_net_type:
                if not NetParser.parser[set_net_type[node.name]].if_supported(node):
                    raise RuntimeError(
                        f"{set_net_type[node.name]} does not support [{node.name}]{node.op_type}"
                    )
            else:
                if not NetParser.parser[self.id_to_name[0]].if_supported(node):
                    # warnings.warn(f"{self.id_to_name[0]} does not support [{layer.name}]{layer.type}")
                    set_net_type[node.name] = self._search_support_backend(node)
                    pass
                else:
                    continue
            logger.info(
                "set op: %s to backend: %s" % (node.name, set_net_type[node.name])
            )

        for _id in self.id_to_name:
            name = self.id_to_name[_id]
            for layer in set_net_type:
                if set_net_type[layer] == name:
                    set_net_type[layer] = _id
        self.set_net_type = set_net_type

    def _search_support_backend(self, node):
        for name, parser in NetParser.parser.items():
            if parser.if_supported(node):
                return name
        raise RuntimeError(
            f"Unsupported op type: {node.op_type}(domain={node.domain}, version={node.version})"
        )

    def __del__(self):
        self.temp_dir.cleanup()

    def _new_data_gen(self, net_type):
        if self.data_num == 0:
            return None
        if net_type in self.datagen_by_net_type:
            return self.datagen_by_net_type[net_type]
        from copy import deepcopy

        graph = deepcopy(self.ori_net)
        graph.update_tensor_shape_with_preset(self.config["calib_shapes"])
        data_generator = PerFrameDataGenerator(graph)

        input_path_map = {}
        rand_input = False
        if net_type in self.config and "input_path" in self.config[net_type]:
            input_path_map = self.config[net_type]["input_path"]
        else:
            input_path_map = self.input_data_map

        if "rand_input" in self.config:
            rand_input = self.config["rand_input"]
        data_generator.set_config(
            input_path_map=input_path_map, data_num=self.data_num, rand_input=rand_input
        )
        data_generator.prepare()
        self.datagen_by_net_type[net_type] = data_generator
        return data_generator

    def parse(self, into_parade=None):
        if into_parade is None:
            into_parade = FakeParade()
        for t in into_parade.tensors:
            if t.dtype == "float32":
                pass

        graph = self.ori_net
        # graph should have been infer from the config[build_shapes], do a check to make sure
        assert all(
            graph.get_tensor_shape(x) == self.build_shapes[x] for x in self.build_shapes
        ), "the tensor shapes in graph mismatches with those in self.build_shapes"

        logger.info("check support info ...")
        splitter = Splitter(
            graph,
            self.set_net_type,
            self.config["set_breakpoints"] if "set_breakpoints" in self.config else [],
        )
        splitter.run()
        nets = splitter.current_subnet_list
        _id = 0

        logger.info(f"start dispatch graph, got {len(nets)} sub-graph(s)")

        for net, tp in nets:
            net_type_name = self.id_to_name[tp]
            if net_type_name not in self._used_workspace:
                self._used_workspace.append(net_type_name)
            net_config = {"output_names": []}
            if net_type_name in self.config:
                net_config.update(self.config[net_type_name])
            net_config["output_names"] = net_config["output_names"] + self.config.get(
                "output_names", []
            )
            net_config["output_names"] = list(
                set(net_config["output_names"] + net.output)
            )
            if "dtype" in self.config:
                net_config["dtype"] = self.config["dtype"]
            if self.max_batch_size > 0:
                net_config["max_batch_size"] = self.max_batch_size
            if net_type_name == "tensorrt" and "blob_range" in self.config:
                net_config["blob_range"] = self.config["blob_range"]
            if self.config.get("dump_info", False):
                net_config["dump_info"] = True
                net_config["dump_filename"] = os.path.join(
                    os.path.dirname(self.config.get("output_engine", "engine.bin")),
                    f"dump_model_{_id}.bin",
                )

            # security poliy: hidden passwd
            config_str = str(net_config).lower()
            if "passwd" in config_str or "password" in config_str:
                logger.info(f"dispatch `{net.name}` to `{net_type_name}`.")
            else:
                logger.info(
                    f"dispatch `{net.name}` to `{net_type_name}` with config: {net_config}"
                )
            logger.debug(f"\n{net}")
            into_parade = (
                NetParser.parser[net_type_name](
                    net,
                    net_config,
                    data_generator=self._new_data_gen(net_type_name),
                    net_id=_id,
                )
                .set_shape_range(self.min_shapes, self.max_shapes, self.calib_shapes)
                .parse(into_parade)
            )
            _id += 1
            logger.info(f"generate parade for `{net.name}` successfully")

        return into_parade

    # def parse_input(self):
    #     def apply_batch_size(shape):
    #         shape[0] = self.max_batch_size
    #         return shape
    #     res = {}
    #     for ipt in self.ori_net.input:
    #         sp = self.ori_net.get_tensor_shape(ipt)
    #         tensor = FakeTensor(dtype=Dtype.Float32, shape=apply_batch_size(list(sp.dim)), name=ipt)
    #         res[ipt] = tensor
    #     return res

    def gen_nart_config(self, args, MODULE_DICT):
        res = {"workspaces": {"default": {}}, "input_workspace": "default"}
        # input workspaces
        res["input_workspace"] = list(MODULE_DICT[args.type].values())[
            0
        ].gen_input_workspace()
        # workspaces
        for module_name, module in MODULE_DICT[args.type].items():
            # FIXME
            # generate engien.bin.json
            # if this module is used, or it is the main module, create workspace for it.
            flag = module_name in self._used_workspace or module_name == args.type
            if module_name == "x86":
                module_name = "mkldnn"
            if flag:
                res["workspaces"][module_name] = module.gen_workspace()

        return res
