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
from .trt_utils import ParserContext
from .trt_utils import CollectExtraOutput
from .trt_utils import tensorrt_version
from ..core.art import FakeParade
from ..core.art import FakeOp
from ..core.art import FakeTensor
from ..core.art import Fakes
from ..core.art import Dtype
import os
import tempfile
import logging
import numpy as np
import warnings
from .parser import Parser
from ..ops.op import OpDesc

from onnx import numpy_helper
import math

DEBUG_ENABLE_VERBOSE = False
DEBUG_DUMP_NETWORK_TOPO = False

LOGGER = logging.getLogger("nart.modules.tensorrt")


class Logger:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            import tensorrt.tensorrt as trt

            level = LOGGER.getEffectiveLevel()
            level_map = {
                logging.DEBUG: trt.Logger.VERBOSE,
                logging.INFO: trt.Logger.INFO,
                logging.WARNING: trt.Logger.WARNING,
                logging.ERROR: trt.Logger.ERROR,
                logging.FATAL: trt.Logger.INTERNAL_ERROR,
            }
            level = level_map[level]
            # disable VERBOSE if DEBUG_ENABLE_VERBOSE==False
            level = (
                trt.Logger.INFO
                if (level == trt.Logger.VERBOSE and not DEBUG_ENABLE_VERBOSE)
                else level
            )
            cls.instance = trt.Logger(level)
        return cls.instance


def enter_interative(locals):
    import code

    code.interact(banner="entering interative console", local=locals)


class TensorrtParserProxy(object):
    """Maybe a proxy class for real parser.

    Normally, we can simply set TensorrtParser variable to corresponding real parser when this module is imported,
    but switch requires that executing `import TensorrtParser` should not introduce tensorrt dependency,
    so we have to delay the detection of tensorrt version to when methods of TensorrtParser are called.
    This class is such a wrapper for this purpose.
    """

    def __init__(self, **kwargs):
        pass

    def get_class(self):
        """Select parser class according to tensorrt version."""
        trt_ver = tensorrt_version()
        if trt_ver.major == 7:
            if trt_ver.minor == 0:
                return TensorrtParser_7_0
            else:
                return TensorrtParser_7_1
        elif trt_ver.major == 8:
            return TensorrtParser_8_x
        else:
            raise NotImplementedError(
                f"Parser not implemented for tensorrt version `{trt_ver}`"
            )

    def __call__(self, *args, **kwargs):
        """Create real tensorrt parser according to tensorrt version."""
        return self.get_class()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.get_class(), name)


TensorrtParser = TensorrtParserProxy()


class ONNXNodeParser(object):
    """A base class for parser that parses each onnx node."""

    @classmethod
    def __init_subclass__(cls, layertype=None, is_abstract=False, *args, **kwargs):
        # the layertype, is_abstract argument has no use now, leave it to keep backward compatibility.
        super().__init_subclass__(*args, **kwargs)

    @classmethod
    def parse(cls, node, network, itensor_by_name):
        ctx = ParserContext.get_current()
        ctx.cur_node = node
        op, outputs = cls._parse_(node, network, itensor_by_name)
        for name, itensor in outputs.items():
            # print(name, itensor.shape)
            itensor.name = name
        # add output tensors to dict
        itensor_by_name.update(outputs)
        return op

    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        LOGGER.error(f"_parse_ method not implemented for {cls.__name__}")
        get_attr = cls.def_get_attr(node)
        enter_interative(locals())

    @staticmethod
    def def_get_attr(node):
        # attribute = {x.name: x for x in attribute}
        def get_attribute(name, default=None):
            value = node.get_attribute_value(name, default)
            # NOTE: temporary work around, the onnx node generated from caffe prototxt is not standard
            if isinstance(default, (list, tuple)) and len(value) < len(default):
                LOGGER.warning(
                    "non-standard attribute met at tensorrt-parser: value length is less than required"
                )
                # duplicate the value to meet requirement.
                value = list(value) * (len(default) // len(value))
            return value

        return get_attribute

    @staticmethod
    def get_const_array(name):
        """Get a constant tensor in current graph, casted to ndarray.
        The constant tensor can be an initializer, or output of Constant node.

        Returns:
            numpy.ndarray: The constant value if name IS a constant. Returns None if IT IS NOT.
        """
        ctx = ParserContext.get_current()
        graph = ctx.graph
        return graph.get_const_tensor_as_array(name)

    @staticmethod
    def get_itensor_by_name(name):
        """get a tensor with given name, converted to itensor. It may be a graph input, node output or an initializer.
        Constant op may be created from initializer.
        Returns:
            an tensorrt.itensor
        """
        ctx = ParserContext.get_current()
        if name in ctx.itensor_by_name:
            return ctx.itensor_by_name[name]
        array = ONNXNodeParser.get_const_array(name)
        if array is not None:
            from .trt_utils import to_const_tensor

            ret = to_const_tensor(array)
            ctx.itensor_by_name[name] = ret
            return ret
        else:
            raise RuntimeError(
                f'cannot get itensor with name "{name}", not output nor constant'
            )


class TensorrtParserBase(Parser):
    """A base class for tensorrt parser."""

    @classmethod
    def get_support_layer_list(cls):
        # return ONNXNodeParser.registered_parser.keys()
        raise NotImplementedError("supported op list not implemented")

    def __init__(self, net, config={}, data_generator=None, net_id=0):
        super().__init__(net, config, data_generator)
        # those are default configs
        self.config = {
            "output_names": [],
            "use_fp16": False,
            "use_int8_calibration": False,
            "max_workspace_size": 1 << 25,
            "dynamic_shape": True,
            "fp32_layers": [],
            "fallback_gpu_layers": [],
            "blob_range": None,
            # 'use_int8_dipoorlet': False,
            # 'dipoorlet_method': "minmax --profiling_bs 0",
            "use_dla": False,
            "dla_core_id": 0,
            "enable_verbose_logging": False,
            "tactic_sources": None,
        }

        if "use_int8" in config:
            # legacy config 'use_int8' given, check consistency
            if "use_int8_calibration" not in config:
                # update new config using legacy config
                config["use_int8_calibration"] = config["use_int8"]
                LOGGER.warning(
                    "The config key `use_int8` is deprecated, use `use_int8_calibration` instead"
                )
            else:
                # both given, check consistency
                assert config["use_int8_calibration"] == config["use_int8"], (
                    "Config `use_int8_calibration` and `use_int8` are both given but not consistent, "
                    f"{config['use_int8_calibration']} vs {config['use_int8']}"
                )
        if config.get("enable_verbose_logging", False):
            global DEBUG_ENABLE_VERBOSE
            DEBUG_ENABLE_VERBOSE = True

        # update with provided configs
        self.config.update(config)
        self.extend_outputs_ = self.config["output_names"]

        self.data_num = 1
        self.net = net
        self.data_gen = data_generator

        self.blob_range = self.config["blob_range"]

        if data_generator is None:
            LOGGER.info("data_generator is None")
            return
        self.data_num = self.data_gen.data_num

    @classmethod
    def get_passes(cls, backend_specialized=False):
        raise NotImplementedError(f"`get_passes` not implemented in {cls.__name__}.")

    def parse(self, into_parade=None):
        """Currently a parser for tensorrt 7.x"""
        # parse the net
        if into_parade is None:
            into_parade = FakeParade()

        # call backend specialized passes once more.
        for item in self.get_passes(backend_specialized=True):
            item.run(self.net)

        # mapping from input name to tensor
        input_tensor_dict = self.parse_input_dict()
        for tensor in into_parade.tensors:
            if tensor.name in input_tensor_dict:
                input_tensor_dict[tensor.name] = tensor
        # collect extra outputs
        self.collect_extra_outputs()

        dynamic_shape = self.config["dynamic_shape"]
        output_names = self.parse_output_names()

        tlogger = Logger.get_instance()
        import tensorrt.tensorrt as trt

        with trt.Builder(tlogger) as builder, builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network, builder.create_builder_config() as config, ParserContext(
            network, self.net
        ) as ctx:
            if (
                self.config["tactic_sources"]
                and tensorrt_version().major >= 7
                and tensorrt_version().minor >= 2
            ):
                # if tactic_sources was set
                tactic_sources = self.config["tactic_sources"]
                tactic_sources = [
                    getattr(trt.TacticSource, name)
                    for name in tactic_sources
                    if hasattr(trt.TacticSource, name)
                ]
                LOGGER.info("setting tactic sources to: {0}".format(tactic_sources))
                tactic_sources_mask = 0
                for source in tactic_sources:
                    tactic_sources_mask = tactic_sources_mask | (1 << int(source))
                config.set_tactic_sources(tactic_sources_mask)
            # the dict from tensor name to tensorrt.ITensor
            itensor_by_name = ParserContext.get_current().itensor_by_name
            for name in self.net.network_inputs:
                # TODO: no dtype information in onnx model??
                shape = self.get_determined_shape(name)
                if dynamic_shape:
                    # mark shape size on dynamic dimensions as -1 if dynamic shape enabled.
                    shape = [
                        x if not self.is_dynamic_shape_size(x) else -1 for x in shape
                    ]
                else:
                    # check whether all shape dimensions are determined.
                    if any(self.is_dynamic_shape_size(x) for x in shape):
                        raise RuntimeError(
                            f"dynamic_shape==False, but input tensor `{name}` has dynamic shape"
                        )
                if name in input_tensor_dict:
                    dtype = input_tensor_dict[name].dtype
                    from .trt_utils import dtype_from_art_dtype

                    dtype = dtype_from_art_dtype(dtype)
                else:
                    # actually this should not happen
                    dtype = trt.float32
                itensor_by_name[name] = network.add_input(
                    name=name, dtype=dtype, shape=shape
                )
            network.name = self.net.name

            LOGGER.info("begin parsing")

            if self.config["use_dla"]:
                dla_fallback_node_list = self.config["fallback_gpu_layers"]

            for node in self.net.nodes:
                cls = self.get_registered_op(node.op_type, node.domain, node.version)
                op = cls.parse(node, network, itensor_by_name)
                # DLA device FC layer or FC+Relu/LeakyRelu/PRelu layers are forced fallback to GPU
                if self.config["use_dla"]:
                    if node.op_type == "Gemm":
                        dla_fallback_node_list.append(node.name)
                    elif (
                        node.op_type == "Relu"
                        or node.op_type == "LeakyRelu"
                        or node.op_type == "PRelu"
                    ):
                        for node_ in self.net.nodes:
                            if (
                                node.input[0] == node_.output[0]
                                and node_.name in dla_fallback_node_list
                            ):
                                dla_fallback_node_list.append(node.name)
                                break

            LOGGER.info("parse finish")
            # mark subnet outputs
            for out_name in output_names:
                network.mark_output(itensor_by_name[out_name])

            # build configs

            if dynamic_shape:
                # optimization profile
                profile = builder.create_optimization_profile()
                for name in self.net.network_inputs:
                    min_shape = self.get_min_shape(name)
                    opt_shape = self.get_build_shape(name)
                    max_shape = self.get_max_shape(name)
                    profile.set_shape(name, min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)

            config.max_workspace_size = self.config["max_workspace_size"]
            # config.min_timing_iterations = 3
            config.flags |= (
                1 << int(trt.BuilderFlag.FP16) if self.config["use_fp16"] else 0
            )
            if (
                "use_strict_types" in self.config
                and network.name in self.config["use_strict_types"]
                and self.config["use_strict_types"][network.name]
            ):
                config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
                LOGGER.info("set net: %s to strict types." % network.name)

            if self.config["use_dla"]:
                # if builder.num_DLA_cores < 1:
                # raise RuntimeError(f'The platform is not support DLA')
                config.default_device_type = trt.DeviceType.DLA
                dla_core_id = self.config["dla_core_id"]
                # if dla_core_id  >= builder.num_DLA_cores:
                # raise RuntimeError(f'DLA core available is {builder.num_DLA_cores}, but set core id {dla_core_id}')
                config.DLA_core = dla_core_id
                # Enable layers marked to execute on GPU if layer cannot execute on DLA
                config.flags |= 1 << int(trt.BuilderFlag.GPU_FALLBACK)
                # DLA support is currently limited to networks running in either FP16 or INT8 mode
                if self.config["use_int8_calibration"]:
                    config.flags |= 1 << int(trt.BuilderFlag.INT8)
                else:
                    config.flags |= 1 << int(trt.BuilderFlag.FP16)
                self.set_fallback_gpu_layers(
                    network, tlogger, config, dla_fallback_node_list
                )

            self.set_calibrator(builder, config)

            # if self.config['use_int8_dipoorlet']:
            #     dct_name_tensor = self.net.network_inputs
            #     dipoorlet_range = self.calibrate_by_dipoorlet(
            #         self.net, dct_name_tensor, self.config['dipoorlet_method'])
            #     if self.blob_range is not None:
            #         # if given, use the manual tensor range in config.
            #         dipoorlet_range.update(self.blob_range)
            #     self.blob_range = dipoorlet_range

            # set dynamic range
            if self.blob_range is not None:
                self.set_dynamic_range(config, network, tlogger)

            self.set_fp32_layers(network, tlogger, config, self.config["fp32_layers"])

            LOGGER.info("start building")
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError(
                    f"fail to build tensorrt engine for subnet {self.net.name}"
                )
            LOGGER.info("build finished")
            LOGGER.info("start serializing")
            with engine:
                serialized_engine = engine.serialize()
            LOGGER.info("serialize finished")

            # the itensor's scope is inside the `with` stmt, so this method should be called inside `with` stmt.
            outputs = self.parse_output_dict(itensor_by_name)

        from io import BytesIO

        with BytesIO() as buf:
            buf.write(serialized_engine)
            serialized_engine = buf.getvalue()
            self.dump_file_if_required(serialized_engine)

        x = into_parade.append(
            Fakes.TENSORRTNet(
                list(input_tensor_dict.values()), serialized_engine, outputs
            )
        )
        # set name
        for t, name in zip(x, output_names):
            t.name = name
        return into_parade

    def collect_extra_outputs(self):
        """This method will be called before generating output tensor,
        derivate class can append outputs to self.net to bypass some bugs in certain tensorrt version.

        Returns: None
        """
        pass

    def set_calibrator(self, builder, builder_config):
        pass

    def process_input_data(self, path, data_gen, tensor, mangle_fn=None):
        if mangle_fn != None:
            full_path = os.path.join(path, mangle_fn(tensor))
        else:
            full_path = os.path.join(path, tensor)

        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        tensor_data = data_gen.fetch_data(tensor)
        for idx, v in enumerate(tensor_data):
            os.symlink(os.path.abspath(v), f"{full_path}/{idx}.bin")

    # def calibrate_by_dipoorlet(self, model=None, dct_name_tensor=None, method=None) -> dict:
    #     import onnx
    #     from nart.core import Model
    #     from copy import deepcopy
    #     model = deepcopy(model)
    #     model.update_tensor_shape_with_preset(self.calib_shapes)
    #     onnx_model = Model.dump_to_onnx(
    #         Model.make_model(model, producer_name='Dipoorlet'))
    #     del model
    #     self.tmp_dir = tempfile.TemporaryDirectory()
    #     onnx_path = os.path.join(self.tmp_dir.name, 'dipoorlet_model.onnx')
    #     onnx.save(onnx_model, os.path.join(self.tmp_dir.name, onnx_path))
    #     try:
    #         import dipoorlet
    #     except Exception as e:
    #         LOGGER.fatal(str(e))
    #         LOGGER.fatal('Can not import dipoorlet.')
    #         return None

    #     import runpy
    #     import sys
    #     import json

    #     if method == None or method == '':
    #         method = 'minmax'
    #     method = method.split()

    #     output_dir = 'dump'
    #     os.makedirs(output_dir, exist_ok=True)
    #     data_path = self.temp_dir.name + '/input_data/'
    #     for k in dct_name_tensor:
    #         self.process_input_data(data_path, self.data_gen, k)
    #     _save = sys.argv[:]
    #     sys.argv[1:] = ['-M', f'{onnx_path}', '-I', f'{data_path}', '-O',
    #                     f'{output_dir}', '-N', f'{self.data_gen.data_num}', '-D', 'trt', '-A', *method]
    #     LOGGER.info(f'run dipoorlet with {sys.argv[1:]}')
    #     m = runpy.run_module('dipoorlet', run_name='__main__')
    #     sys.argv[:] = _save

    #     max_min_path = os.path.join(output_dir, 'trt_clip_val.json')

    #     with open(max_min_path, 'r') as f:
    #         dipoorlet_range = json.load(f)
    #     return dipoorlet_range['blob_range']

    def set_dynamic_range(self, config, network, logger):
        import tensorrt.tensorrt as trt

        config.flags |= 1 << int(trt.BuilderFlag.INT8)
        # TODO: does STRICT_TYPES flag really needed?
        # config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
        # config.int8_calibrator = None
        blob_range = (
            self.blob_range[network.name]
            if network.name in self.blob_range
            else self.blob_range
        )
        for layer in network:
            if (
                layer.type != trt.LayerType.SHAPE
                and layer.type != trt.LayerType.CONSTANT
                and layer.type != trt.LayerType.CONCATENATION
                and layer.type != trt.LayerType.GATHER
            ):
                layer.precision = trt.DataType.INT8
            for i in range(layer.num_inputs):
                inp = layer.get_input(i)
                if inp is not None and inp.name in blob_range:
                    dmax = blob_range[inp.name]
                    if inp.dynamic_range is None:
                        inp.set_dynamic_range(-dmax, dmax)
                        logger.log(
                            trt.Logger.INFO,
                            f'set dynamic range of tensor "{inp.name}" to {dmax}.',
                        )
            for i in range(layer.num_outputs):
                output = layer.get_output(i)
                if output.name in blob_range:
                    dmax = blob_range[output.name]
                    if output.dynamic_range is None:
                        output.set_dynamic_range(-dmax, dmax)
                        logger.log(
                            trt.Logger.INFO,
                            f'set dynamic range of tensor "{output.name}" to {dmax}.',
                        )

    def set_fp32_layers(self, network, logger, config, names):
        import tensorrt.tensorrt as trt

        names = [x.split("::") for x in names]
        flag = False

        def match_any(name):
            """Is the layer named name a sub-layer of any layer in names.
            The match method: any one in names which is a prefix of name.
            """
            for item in names:
                if len(item) <= len(name) and name[: len(item)] == item:
                    return True
            return False

        for layer in network:
            if match_any(layer.name.split("::")) and layer.type not in (
                trt.LayerType.CONSTANT,
                trt.LayerType.SHAPE,
                trt.LayerType.CONCATENATION,
                trt.LayerType.GATHER,
                trt.LayerType.SHUFFLE,
            ):
                layer.precision = trt.DataType.FLOAT
                logger.log(trt.Logger.INFO, f"layer {layer.name} precision set to fp32")
                flag = True
        if flag:
            # use STRICT_TYPES flag to force tensorrt obey the precision requirement for fp32 layers.
            config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)

    def set_fallback_gpu_layers(self, network, logger, config, names):
        import tensorrt.tensorrt as trt

        names = [x.split("::") for x in names]

        def match_any(name):
            """Is the layer named name a sub-layer of any layer in names.
            The match method: any one in names which is a prefix of name.
            """
            for item in names:
                if len(item) <= len(name) and name[: len(item)] == item:
                    return True
            return False

        for layer in network:
            if match_any(layer.name.split("::")) or layer.type in (
                trt.LayerType.CONSTANT,
                trt.LayerType.SHAPE,
                trt.LayerType.GATHER,
                trt.LayerType.SHUFFLE,
                trt.LayerType.SOFTMAX,
                trt.LayerType.CONCATENATION,
            ):
                config.set_device_type(layer, trt.DeviceType.GPU)
                logger.log(
                    trt.Logger.INFO, f"force layer {layer.name} fall back to gpu"
                )

    def parse_input_dict(self):
        """parse input dict

        Returns:
            a dict, name to FakeTensor.

        """
        res = {}
        for t in self.net.network_inputs:
            sp = self.net.get_tensor_shape(t)
            dtype = self.net.get_tensor_dtype(t)
            LOGGER.info("set input `%s` to dtype: %s" % (t, dtype))

            tensor = FakeTensor(dtype=dtype, shape=(sp, 1, 0), name=t)
            res[t] = tensor
        return res

    def parse_output_dict(self, itensor_by_name):
        from .trt_utils import art_dtype_from_dtype

        output_names = self.parse_output_names()
        outputs = [
            FakeTensor(
                dtype=art_dtype_from_dtype(itensor_by_name[x].dtype),
                shape=(self.net.get_tensor_shape(x), 1, 0),
                name=x,
            )
            for x in output_names
        ]
        return outputs

    @classmethod
    def register_defaults(cls):
        from .trt_utils import init_parsers

        init_parsers()

    @classmethod
    def gen_workspace(cls):
        return {}

    @classmethod
    def gen_input_workspace(cls):
        return "tensorrt"


def _repr_edge(blob):
    return blob.name + " " + str(blob.shape)


def trt_network_to_dot_graph(network):
    import graphviz

    dot = graphviz.Digraph(comment="Network")

    # add nodes (layers)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        dot.node(layer.name)

    # add nodes (inputs)
    for i in range(network.num_inputs):
        dot.node(network.get_input(i).name)

    # add nodes (outputs)
    for i in range(network.num_outputs):
        dot.node(network.get_output(i).name)

    # add layer->layer edges
    for a in range(network.num_layers):
        layer_a = network.get_layer(a)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for i in range(layer_a.num_outputs):
                output_i = layer_a.get_output(i)

                for j in range(layer_b.num_inputs):
                    input_j = layer_b.get_input(j)

                    if output_i == input_j:
                        dot.edge(layer_a.name, layer_b.name, label=_repr_edge(input_j))

    # add input->layer edges
    for i in range(network.num_inputs):
        input_i = network.get_input(i)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for j in range(layer_b.num_inputs):
                input_j = layer_b.get_input(j)

                if input_i == input_j:
                    dot.edge(input_i.name, layer_b.name, label=_repr_edge(input_j))

    # add layer->output edges
    for i in range(network.num_outputs):
        input_i = network.get_output(i)

        for b in range(network.num_layers):
            layer_b = network.get_layer(b)

            for j in range(layer_b.num_outputs):
                input_j = layer_b.get_output(j)

                if input_i == input_j:
                    dot.edge(layer_b.name, input_i.name, label=_repr_edge(input_j))

    return dot


class TensorrtParser_7_0(TensorrtParserBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_passes(cls, backend_specialized=False):
        from .. import passes

        ret = [
            passes.RemoveCertainOp(["Dropout"]),
            passes.ConstantToInitializer(),
            passes.ReplaceRoundByFloor(),
            passes.SplitReduceL2ToGeneralOps(),
            passes.ConvertDualInputScale(),
            passes.SplitCaffeScale(),
            passes.SplitCaffeExp(),
            passes.SumToAdd(),
            passes.FuseFloorDiv(),
            passes.ConvertDynamicUpsample(),
            passes.ConvFuser(),
            passes.RemoveIdentityOps(),
            passes.ConstantToInitializer(),
            passes.DeadCodeElimination(),
        ]
        if backend_specialized:
            # the following passes will be executed on sub-models after splitting
            # that will be executed on tensorrt for certain.
            ret.extend(
                [
                    passes.InsertUnsqueezeForBroadcastOp("MatMul"),
                ]
            )
        return ret

    def collect_extra_outputs(self):
        collect_pass = CollectExtraOutput()
        collect_pass.run(self.net)
        extra_outputs = collect_pass.extra_outputs
        if len(extra_outputs) != 0:
            LOGGER.warning(
                f"adding {extra_outputs} to output to bypass bug in tensorrt"
            )
            self.extend_outputs_.extend(extra_outputs)

    def set_calibrator(self, builder, builder_config):
        if self.config["use_int8_calibration"]:
            raise NotImplementedError("calibration not supported on tensorrt 7.0")
            # raise NotImplementedError("calibration not supported on tensorrt 7.0, use dipoorlet instead: "
            #                           "http://spring.sensetime.com/docs/dipoorlet/index.html")


class TensorrtParser_7_1(TensorrtParserBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_passes(cls, backend_specialized=False):
        from .. import passes

        ret = [
            passes.RemoveCertainOp(["Dropout"]),
            passes.ConstantToInitializer(),
            passes.ReplaceRoundByFloor(),
            passes.SplitReduceL2ToGeneralOps(),
            passes.ConvertDualInputScale(),
            passes.SplitCaffeScale(),
            passes.SplitCaffeExp(),
            passes.SumToAdd(),
            passes.FuseFloorDiv(),
            passes.ConvertGlobalPoolToReduce(),
            passes.RemoveIdentityOps(),
            passes.ConstantToInitializer(),
            passes.SimplifyReisze(),
            passes.SimplifyExpand(),
            passes.DeadCodeElimination(),
        ]
        if backend_specialized:
            # the following passes will be executed on sub-models after splitting
            # that will be executed on tensorrt for certain.
            ret.extend(
                [
                    passes.InsertUnsqueezeForBroadcastOp("MatMul"),
                ]
            )
        return ret

    def set_calibrator(self, builder, builder_config):
        from tensorrt import tensorrt as trt
        from .trt_utils import create_calibrator

        dynamic_shape = self.config["dynamic_shape"]

        if self.config["use_int8_calibration"]:
            assert (
                self.data_gen is not None
            ), "data_generator is None, can not do calibration"
            builder_config.flags |= 1 << int(trt.BuilderFlag.INT8)

            calib_shapes = dict()
            for name in self.net.network_inputs:
                calib_shapes[name] = self.calib_shapes[name]
            if dynamic_shape:
                # do calibration in dynamic shape mode, need to set calibration profile
                profile = builder.create_optimization_profile()

                for name in self.net.network_inputs:
                    shape = self.calib_shapes[name]
                    profile.set_shape(name, shape, shape, shape)
                builder_config.set_calibration_profile(profile)

            # create calibrator if INT8 enabled and explicit dynamic range is not set.
            def _input_fp_dict_getter():
                # used to delay evaluation of input_fp_dict (input file path dict).
                input_fp_dict = dict()
                self.data_gen.pre_fetch(self.net.network_inputs)
                for name in self.net.network_inputs:
                    input_fp_dict[name] = self.data_gen.fetch_data(name)
                return input_fp_dict

            calibrator = create_calibrator(
                self.data_num,
                _input_fp_dict_getter,
                calib_shapes,
                net_name=self.net.name,
            )
            builder_config.int8_calibrator = calibrator


class TensorrtParser_8_x(TensorrtParser_7_1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
