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

from abc import ABC, abstractmethod
from onnx import numpy_helper, helper
import logging
import numpy as np

logger = logging.getLogger("nart.alter.caffe")


class Alter(object):
    def __init__(self):
        pass


# _caffe_resitry = {}
from ...ops import OpSetRegistry


class CaffeAlterContext(object):
    _current = None

    def __init__(self, extra_parsers=None, caffe_pb2=None):
        """Args:
        extra_parsers (OpSetRegistry): Parsers that is to be added to current context.
        """
        cls = self.__class__
        self._old = None
        from copy import copy

        self._caffe_registry = (
            copy(cls._current.caffe_registry)
            if cls._current is not None
            else OpSetRegistry()
        )
        if extra_parsers is not None:
            assert isinstance(
                extra_parsers, OpSetRegistry
            ), "The extra_pasers should be OpSetRegistry"
            self._caffe_registry.update(extra_parsers)

        if caffe_pb2 is not None:
            self._caffe_pb2 = caffe_pb2
        else:
            from ...proto import nart_caffe_pb2

            self._caffe_pb2 = nart_caffe_pb2

    @property
    def caffe_registry(self) -> OpSetRegistry:
        return self._caffe_registry

    @property
    def caffe_pb2(self):
        return self._caffe_pb2

    def __enter__(self):
        cls = self.__class__
        self._old = cls._current
        cls._current = self
        return self

    def __exit__(self, type, value, trace):
        cls = self.__class__
        cls._current = self._old


# create the default context
_default_context = CaffeAlterContext()

CaffeAlterContext._current = _default_context


def current_context() -> CaffeAlterContext:
    return CaffeAlterContext._current


def register_caffe_layer(op_name, cls, domain=None, version=None):
    # global _caffe_resitry
    ctx = current_context()
    if not issubclass(cls, Layer):
        raise NotImplementedError
    if op_name in ctx.caffe_registry:
        pass
    ctx.caffe_registry.insert(cls, op_type=op_name, domain=domain, version=version)


def initializer_to_blob(init, caffe_pb2):
    arr = numpy_helper.to_array(init)
    blob = caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.astype(float).flat)
    return blob


def array_to_blob(arr, caffe_pb2):
    blob = caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.astype(float).flat)
    return blob


def get_attr_with_default(node, attr_name, default_value=None):
    return node.get_attribute_value(attr_name, default_value)


class CaffeAlter(Alter):
    def __init__(self, graph, caffe_pb2=None, use_input_layer=False):
        """Convert core.Graph to caffe model.

        Args:
            graph (core.Graph): the original graph.
            caffe_pb2 (NetParameter): the target caffemodel protobuf message.
            use_input_layer (bool): whether to use Input layers to represent inputs.
        """
        self._pre_parse_hook = None
        self._post_parse_hook = None
        self.graph = graph
        self.caffe_pb2 = caffe_pb2
        self.use_input_layer = use_input_layer
        if not self.caffe_pb2:
            # use the caffe_pb2 defined in context
            self.caffe_pb2 = current_context().caffe_pb2
        self.caffe_net = self.caffe_pb2.NetParameter()
        self.caffe_net.name = graph.name

    def parse_input(self):
        """parse input.

        input not include weight.
        """
        if not self.use_input_layer:
            self.add_inputs()
        else:
            self.add_input_layers()

    def add_inputs(self):
        inps = [inp for inp in self.graph.input if inp not in self.graph.initializer]
        self.caffe_net.input.extend(inps)
        for inp in inps:
            inp_shape = self.caffe_pb2.BlobShape()
            inp_shape.dim.extend(self.graph.get_tensor_shape(inp))
            self.caffe_net.input_shape.extend([inp_shape])

    def add_input_layers(self):
        for ipt in self.graph.network_inputs:
            ipt_layer = self.caffe_net.layer.add()
            ipt_layer.name = ipt
            ipt_layer.type = "Input"
            ipt_layer.top[:] = [ipt]
            inp_shape = ipt_layer.input_param.shape.add()
            inp_shape.dim[:] = self.graph.get_tensor_shape(ipt)

    def pre_parse(self):
        if self._pre_parse_hook:
            self._pre_parse_hook(self.graph, self.caffe_net)

    def register_pre_parse_hook(self, func):
        self._pre_parse_hook = func

    def parse(self):
        self.pre_parse()
        self.parse_input()
        ctx = current_context()
        for node in self.graph.nodes:
            parser_cls = None
            parser_cls = ctx.caffe_registry.find(
                node.op_type, node.domain, node.version
            )
            if parser_cls is None:
                logger.error(
                    f"No CaffeAlter parser registered for Op "
                    f"`{node.op_type} (domain={node.domain}, version={node.version})`"
                )
                parser_cls = ctx.caffe_registry.find("Unknown")
            layer = parser_cls(node).parse(self.caffe_pb2)
            layer = self.post_parse_layer(node, layer)
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            logger.info(
                f"alter node `{node.op_type}` to layer(s): {', '.join([x.type for x in layer])}"
            )
            self.caffe_net.layer.extend(layer)
        self.post_parse()
        self.validate(self.caffe_net)
        return self.caffe_net

    def post_parse(self):
        if self._post_parse_hook:
            self._post_parse_hook(self.graph, self.caffe_net)

    def register_post_parse_hook(self, func):
        self._post_parse_hook = func

    def post_parse_layer(self, node, layers):
        return layers

    def validate(self, netdef):
        """check the validity of produced caffe model."""
        produced = set(self.caffe_net.input)
        # check all bottoms are produced by one layer or is input.
        for layer in self.caffe_net.layer:
            for bottom in layer.bottom:
                if bottom not in produced:
                    if self.graph.is_const_tensor(bottom):
                        logger.error(
                            f"layer `{layer.name}` has a bottom named {bottom}, which is a constant tensor"
                        )
                    else:
                        logger.error(
                            f"layer `{layer.name}` has a bottom named {bottom}, which is not produced by anybody"
                        )
            for top in layer.top:
                produced.add(top)


class Layer(ABC):
    def __init__(self, node):
        self.node = node

    @abstractmethod
    def layer_type(self):
        raise NotImplementedError

    def layer_name(self):
        if self.node.name != "":
            return self.node.name
        return self.node.type + "_" + str(self.node.owning_graph.nodes.index(self.node))

    @classmethod
    def __init_subclass__(
        cls, is_register=False, op_type=None, domain="", version=9, *args, **kwargs
    ):
        """Args:
        is_register (bool): If true, this class will be registered to current CaffeAlterContext
                            when defined.
        op_type (str): The corresponding op_type
        domain (str): The corresponding operator set domain, default="".
        version (int): The corresponding operator set version, default=9.
        """
        super().__init_subclass__(*args, **kwargs)
        # Bind the domain & version to class, otherwise, this information will be lost if is_register is False.
        cls._domain = domain
        cls._version = version
        if op_type is None:
            op_type = cls.__name__
        cls._op_type = op_type
        if not is_register:
            return
        ctx = current_context()
        if op_type not in ctx.caffe_registry:
            ctx.caffe_registry.insert(cls, op_type=op_type)
        else:
            logger.warning(f"{op_type} has been overrided by {cls}")

    @abstractmethod
    def parse(self, caffe_pb2):
        raise NotImplementedError


class NonParamLayer(Layer):
    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def layer_type(self):
        raise NotImplementedError

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = list(self.node.input)
        layer.top[:] = list(self.node.output)
        return layer


class BatchNormalization(Layer, is_register=True):
    def layer_type(self):
        return "BN"

    def parse(self, caffe_pb2):
        from onnx import helper

        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        layer.bn_param.moving_average = True
        layer.bn_param.var_eps = get_attr_with_default(self.node, "epsilon", 1e-5)

        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[1]], caffe_pb2
                )
            ]
        )
        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[2]], caffe_pb2
                )
            ]
        )
        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[3]], caffe_pb2
                )
            ]
        )
        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[4]], caffe_pb2
                )
            ]
        )

        channel = len(layer.blobs[3].data)
        for i in range(4):
            layer.blobs[i].shape.ClearField("dim")
            layer.blobs[i].shape.dim.extend([1, channel, 1, 1])
        return layer


class Clip(Layer, is_register=True):
    def layer_type(self):
        return "Clip"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        if len(self.node.input) == 1:
            # Clip-6
            min_val = get_attr_with_default(self.node, "min")
            max_val = get_attr_with_default(self.node, "max")
        else:
            # Clip-10
            min_const = initializer_to_blob(
                self.node.owning_graph.initializer[self.node.input[1]], caffe_pb2
            )
            max_const = initializer_to_blob(
                self.node.owning_graph.initializer[self.node.input[2]], caffe_pb2
            )
            if hasattr(min_const, "item"):
                min_val = min_const.item(0)
                max_val = max_const.item(0)
            else:
                assert hasattr(
                    min_const, "data"
                ), "min_const should have `item` or `data` attribute."
                min_val = min_const.data[0]
                max_val = max_const.data[0]

        if min_val == 0.0 and max_val == 6.0:
            # ReLU6
            layer.type = "ReLU6"
        else:
            # normal clip
            layer.clip_param.min = min_val
            layer.clip_param.max = max_val

        return layer


class Concat(Layer, is_register=True):
    def layer_type(self):
        return "Concat"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        layer.concat_param.axis = get_attr_with_default(self.node, "axis", 1)
        return layer


class Conv(Layer, is_register=True):
    def layer_type(self):
        dilations = get_attr_with_default(self.node, "dilations", [1, 1])
        has_hole = any(x != 1 for x in dilations)
        return "Convolution" if not has_hole else "HoleConvolution"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        layer.convolution_param.group = get_attr_with_default(self.node, "group", 1)

        kh, kw = get_attr_with_default(
            self.node,
            "kernel_shape",
            self.node.owning_graph.get_tensor_shape(self.node.input[1])[-2:],
        )
        kc = self.node.owning_graph.get_tensor_shape(self.node.input[1])[0]
        phs, pws, phe, pwe = get_attr_with_default(self.node, "pads", [0, 0, 0, 0])
        assert phs == phe and pws == pwe
        sh, sw = get_attr_with_default(self.node, "strides", [1, 1])
        dh, dw = get_attr_with_default(self.node, "dilations", [1, 1])

        layer.convolution_param.num_output = kc
        layer.convolution_param.kernel_h = kh
        layer.convolution_param.kernel_w = kw
        layer.convolution_param.pad_h = phs
        layer.convolution_param.pad_w = pws
        layer.convolution_param.stride_h = sh
        layer.convolution_param.stride_w = sw
        if dh != 1 or dw != 1:
            layer.convolution_param.hole_h = dh
            layer.convolution_param.hole_w = dw

        layer.convolution_param.bias_term = False
        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[1]], caffe_pb2
                )
            ]
        )
        if len(self.node.input) == 3:
            layer.convolution_param.bias_term = True
            layer.blobs.extend(
                [
                    initializer_to_blob(
                        self.node.owning_graph.initializer[self.node.input[2]],
                        caffe_pb2,
                    )
                ]
            )
        return layer


class Corr(Layer, is_register=True):
    def layer_type(self):
        return "Correlation"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[::-1]
        layer.top[:] = [self.node.output[0]]

        layer.correlation_param.groups = get_attr_with_default(self.node, "groups", 1)

        return layer


class Correlation1D(Layer, is_register=True):
    def layer_type(self):
        return "Correlation1D"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        layer.correlation_param.max_displacement = self.node.get_attribute_value(
            "max_displacement"
        )
        layer.correlation_param.kernel_size = self.node.get_attribute_value(
            "kernel_size"
        )
        layer.correlation_param.pad = get_attr_with_default(self.node, "pad", 0)
        layer.correlation_param.single_direction = get_attr_with_default(
            self.node, "single_direction", 0
        )

        return layer


class ConvTranspose(Layer, is_register=True):
    def layer_type(self):
        return "Deconvolution"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        group = get_attr_with_default(self.node, "group", 1)
        kh, kw = get_attr_with_default(
            self.node,
            "kernel_shape",
            self.node.owning_graph.get_tensor_shape(self.node.input[1])[-2:],
        )
        kc = self.node.owning_graph.get_tensor_shape(self.node.input[1])[1]
        phs, pws, phe, pwe = get_attr_with_default(self.node, "pads", [0, 0, 0, 0])
        assert phs == phe and pws == pwe
        sh, sw = get_attr_with_default(self.node, "strides", [1, 1])
        dh, dw = get_attr_with_default(self.node, "dilations", [1, 1])
        layer.convolution_param.group = group
        layer.convolution_param.num_output = kc * group
        layer.convolution_param.kernel_h = kh
        layer.convolution_param.kernel_w = kw
        layer.convolution_param.pad_h = phs
        layer.convolution_param.pad_w = pws
        layer.convolution_param.stride_h = sh
        layer.convolution_param.stride_w = sw
        if dh != 1:
            layer.convolution_param.hole_h = dh
        if dw != 1:
            layer.convolution_param.hole_w = dw

        layer.convolution_param.bias_term = False
        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[1]], caffe_pb2
                )
            ]
        )
        if len(self.node.input) == 3:
            layer.convolution_param.bias_term = True
            layer.blobs.extend(
                [
                    initializer_to_blob(
                        self.node.owning_graph.initializer[self.node.input[2]],
                        caffe_pb2,
                    )
                ]
            )
        return layer


class Eltwise(Layer, is_register=True):
    def layer_type(self):
        return "Eltwise"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
        layer.eltwise_param.coeff[:] = get_attr_with_default(
            self.node, "coeff", [1.0] * len(self.node.input)
        )
        return layer


class Upsample(Layer, is_register=True):
    def layer_type(self):
        mode = get_attr_with_default(self.node, "mode")
        if mode == "nearest":
            return "NNInterp"
        elif mode == "bilinear" or mode == "linear":
            return "Interp"
        else:
            logger.fatal(f"Unsupported Upsample node with mode={mode}")

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        if len(self.node.input) == 1:
            # compatible with nart.tools.pytorch
            oh = self.node.get_attribute_value("height")
            ow = self.node.get_attribute_value("width")
        else:
            scales = numpy_helper.to_array(
                self.node.owning_graph.initializer[self.node.input[1]]
            )
            import math

            input_shape = self.node.owning_graph.get_tensor_shape(self.node.input[0])
            oh = math.floor(np.float32(np.float32(scales[2]) * input_shape[2]))
            ow = math.floor(np.float32(np.float32(scales[3]) * input_shape[3]))
        mode = get_attr_with_default(self.node, "mode")

        if mode == "nearest":
            layer.nninterp_param.height = oh
            layer.nninterp_param.width = ow
        elif mode == "bilinear" or mode == "linear":
            layer.interp_param.height = oh
            layer.interp_param.width = ow
        return layer


class Gemm(Layer, is_register=True):
    def layer_type(self):
        return "InnerProduct"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        transpose_layer = None
        if 0 != get_attr_with_default(self.node, "transA", 0):
            sp = self.node.owning_graph.get_tensor_shape(self.node.input[0])
            if len(sp) == 2:
                transpose_layer = caffe_pb2.LayerParameter(
                    type="Transpose", name=self.layer_name() + "_transposed"
                )
                trans_tensor = self.node.input[0] + "_trans"
                transpose_layer.bottom[:] = [self.node.input[0]]
                transpose_layer.top[:] = [trans_tensor]

                # modify FC's input
                layer.bottom[:] = [trans_tensor]

                transpose_layer.transpose_param.dim[:] = [1, 0]
            else:
                logger.fatal("can not support transposed gemm with non-2D input.")

        layer.blobs.extend(
            [
                initializer_to_blob(
                    self.node.owning_graph.initializer[self.node.input[1]], caffe_pb2
                )
            ]
        )
        weight = numpy_helper.to_array(
            self.node.owning_graph.initializer[self.node.input[1]]
        )
        if 0 == get_attr_with_default(self.node, "transB", 0):
            # if transB==0, B will not be transposed before matrix multiply, this requires B to be transposed in caffe InnerProduct layer.
            weight = np.transpose(weight, [1, 0])
            layer.blobs[0].shape.dim[:] = reversed(layer.blobs[0].shape.dim[:])
            layer.blobs[0].data[:] = weight.astype("float32").flat
        layer.inner_product_param.num_output = weight.shape[0]

        if len(self.node.input) == 3:
            layer.blobs.extend(
                [
                    initializer_to_blob(
                        self.node.owning_graph.initializer[self.node.input[2]],
                        caffe_pb2,
                    )
                ]
            )
            layer.inner_product_param.bias_term = True
        else:
            layer.inner_product_param.bias_term = False

        if transpose_layer is not None:
            return [transpose_layer, layer]
        return layer


class _Pool(Layer, is_register=False):
    def layer_type(self):
        return "Pooling"

    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        kh, kw = get_attr_with_default(node, "kernel_shape")
        pads = get_attr_with_default(node, "pads", [0, 0, 0, 0])
        if len(pads) == 2:
            pads = [pads[0], pads[1], pads[0], pads[1]]
        phs, pws, phe, pwe = pads
        assert phs == phe and pws == pwe
        sh, sw = get_attr_with_default(node, "strides", [1, 1])
        ceil_mode = (
            get_attr_with_default(node, "ceil_mode", 0)
            if ("ceil_mode" in node.attr_dict)
            else 0
        )

        if ceil_mode == 0:
            # ceil_mode is True by default in caffe
            pool_param.ceil_mode = False
        pool_param.kernel_h = kh
        pool_param.kernel_w = kw
        pool_param.pad_h = phs
        pool_param.pad_w = pws
        pool_param.stride_h = sh
        pool_param.stride_w = sw

        return pool_param

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]
        self.parse_pooling_param(self.node, layer.pooling_param, caffe_pb2)
        return layer


class MaxPool(_Pool, is_register=True):
    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        super().parse_pooling_param(node, pool_param, caffe_pb2)
        pool_param.pool = caffe_pb2.PoolingParameter.MAX
        return pool_param


class MaxPool_10(_Pool, is_register=True, op_type="MaxPool", version=10):
    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        super().parse_pooling_param(node, pool_param, caffe_pb2)
        pool_param.pool = caffe_pb2.PoolingParameter.MAX
        return pool_param


class AveragePool(_Pool, is_register=True):
    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        super().parse_pooling_param(node, pool_param, caffe_pb2)
        pool_param.pool = caffe_pb2.PoolingParameter.AVE
        if node.get_attribute_value("count_include_pad", 0) == 0:
            logger.warn(
                f"node `{node.name}`'s `count_include_pad` attribute set to 0, which caffe cannot support"
            )
        return pool_param


class AveragePool_10(_Pool, is_register=True, op_type="AveragePool", version=10):
    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        super().parse_pooling_param(node, pool_param, caffe_pb2)
        pool_param.pool = caffe_pb2.PoolingParameter.AVE
        if node.get_attribute_value("count_include_pad", 0) == 0:
            logger.warn(
                f"node `{node.name}`'s `count_include_pad` attribute set to 0, which caffe cannot support"
            )
        return pool_param


class GlobalAveragePool(_Pool, is_register=True):
    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        pool_param.pool = caffe_pb2.PoolingParameter.AVE
        pool_param.global_pooling = True
        return pool_param


class GlobalMaxPool(_Pool, is_register=True):
    def parse_pooling_param(self, node, pool_param, caffe_pb2):
        pool_param.pool = caffe_pb2.PoolingParameter.MAX
        pool_param.global_pooling = True
        return pool_param


class ArgMax(NonParamLayer, is_register=True):
    def layer_type(self):
        return "ArgMax"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.argmax_param.axis = get_attr_with_default(self.node, "axis")

        keepdims = get_attr_with_default(self.node, "keepdims")
        if keepdims != 1:
            logger.warn(
                f"node `{self.node.name}`'s `keepdims` attribute set to 0, which caffe cannot support"
            )
        return layer


class PRelu(Layer, is_register=True):
    def layer_type(self):
        return "PReLU"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        slope = self.node.owning_graph.initializer[self.node.input[1]]
        slope = numpy_helper.to_array(slope).ravel()

        layer.blobs.extend([array_to_blob(slope, caffe_pb2)])
        if len(layer.blobs[0].data) == 1:
            layer.prelu_param.channel_shared = True
            layer.blobs[0].shape.dim.pop()
        return layer


class Relu(NonParamLayer, is_register=True):
    def layer_type(self):
        return "ReLU"


class LeakyRelu(NonParamLayer, is_register=True):
    def layer_type(self):
        return "ReLU"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.relu_param.negative_slope = self.node.get_attribute_value("alpha", 0.01)
        return layer


class ReLU3d(NonParamLayer, is_register=True):
    def layer_type(self):
        return "ReLU3d"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        return layer


class Reshape(Layer, is_register=True):
    def layer_type(self):
        return "Reshape"

    def parse_reshape_param(self, node, reshape_param, caffe_pb2):
        if "shape" in node.attributes:
            shape = get_attr_with_default(node, "shape")
        if "dims" in node.attributes:
            shape = get_attr_with_default(node, "dims")
        if len(node.input) >= 2:
            shape = numpy_helper.to_array(node.owning_graph.initializer[node.input[1]])
        reshape_param.shape.dim.extend(shape)
        return reshape_param

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]
        self.parse_reshape_param(self.node, layer.reshape_param, caffe_pb2)
        return layer

    @staticmethod
    def make_reshape(layer_name, bottom, top, shape, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type="Reshape", name=layer_name)
        layer.bottom[:] = [bottom]
        layer.top[:] = [top]
        layer.reshape_param.shape.dim[:] = list(shape)
        return layer


class Flatten(Reshape, is_register=True):
    def parse_reshape_param(self, node, reshape_param, caffe_pb2):
        axis = node.get_attribute_value("axis", 1)
        if axis == 0:
            reshape_param.shape.dim.extend([-1])
        elif axis == 1:
            reshape_param.shape.dim.extend([0, -1])
        else:
            logger.warn(
                "converting Flatten op whose axis > 1, which MAY lead to incorrect caffe model"
            )
            from functools import reduce
            from operator import mul

            graph = node.owning_graph
            input_shape = graph.get_tensor_shape(node.input[0])
            tail_dims = reduce(mul, input_shape[axis:], 1)
            reshape_param.shape.dim.extend([-1, tail_dims])

        return reshape_param


class Squeeze(Reshape, is_register=True):
    def parse_reshape_param(self, node, reshape_param, caffe_pb2):
        shape = node.owning_graph.get_tensor_shape(node.input[0])
        axes = np.array(get_attr_with_default(node, "axes"))

        axes[axes < 0] += len(shape)
        axes = sorted(axes, reverse=True)
        for axis in axes:
            if shape[axis] != 1:
                logger.warnings("`squeeze` on non-1 dim will be ignored")
            else:
                shape.pop(axis)
        reshape_param.shape.dim.extend([*[0] * axes[-1], *shape[axes[-1] :]])
        return reshape_param


class Unsqueeze(Reshape, is_register=True):
    def parse_reshape_param(self, node, reshape_param, caffe_pb2):
        shape = node.owning_graph.get_tensor_shape(node.input[0])
        axes = get_attr_with_default(node, "axes")
        for axis in axes:
            shape.insert(axis, 1)
        if all([x != -1 for x in shape]):
            shape[0] = -1
        reshape_param.shape.dim.extend(shape)
        return reshape_param


class Sigmoid(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Sigmoid"


class SliceBase(Layer, is_register=False):
    def layer_type(self):
        return "Slice"

    def get_slice_info(self):
        raise NotImplementedError(
            "derivate class should override `SliceBase.get_slice_info` method"
        )

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        starts, ends, axes = self.get_slice_info()
        assert len(axes) == 1

        shape = self.node.owning_graph.get_tensor_shape(self.node.input[0])

        starts = starts[0]
        ends = ends[0]
        axes = axes[0]

        starts = starts if starts >= 0 else shape[axes] + starts
        ends = ends if ends >= 0 else shape[axes] + ends

        slice_points = []
        if starts > 0:
            layer.top.insert(0, layer.name + "_0")
            slice_points.append(starts)
        if ends < shape[axes]:
            layer.top.append(layer.name + "_2")
            slice_points.append(ends)

        layer.slice_param.axis = axes
        layer.slice_param.slice_point.extend(slice_points)
        return layer


class Slice_1(SliceBase, op_type="Slice", is_register=True, version=1):
    def get_slice_info(self):
        starts = get_attr_with_default(self.node, "starts")
        ends = get_attr_with_default(self.node, "ends")
        axes = get_attr_with_default(self.node, "axes")
        return starts, ends, axes


class Slice_10(SliceBase, op_type="Slice", is_register=True, version=10):
    def get_slice_info(self):
        graph = self.node.owning_graph
        starts = graph.get_const_tensor_as_array(self.node.input[1], False)
        ends = graph.get_const_tensor_as_array(self.node.input[2], False)
        axes = graph.get_const_tensor_as_array(self.node.input[3], False)
        return starts, ends, axes


class ShuffleChannel(NonParamLayer, is_register=True):
    def layer_type(self):
        return "ShuffleChannel"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.shuffle_channel_param.group = get_attr_with_default(self.node, "group", 1)
        return layer


class Tanh(NonParamLayer, is_register=True):
    def layer_type(self):
        return "TanH"


class Softmax(Layer, is_register=True):
    def layer_type(self):
        return "Softmax"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]
        axis = get_attr_with_default(self.node, "axis", 1)
        # do a check on axis
        graph = self.node.owning_graph
        input_shape = graph.get_tensor_shape(self.node.input[0])
        nb_dims = len(input_shape)
        if not (axis == -1 or axis == nb_dims - 1) and any(
            x != 1 for x in input_shape[axis + 1 :]
        ):
            # if axis is not the last axis, and axes after it are not all 1s, then this may be incorrect.
            logger.fatal(
                f"When converting ONNX Softmax to caffe Softmax, the `axis`={axis}, which is not the last axis. "
                "This may result to incorrect caffe model"
            )
        layer.softmax_param.axis = axis
        return layer


class Softmax_13(Layer, is_register=True, op_type="Softmax", version=13):
    def layer_type(self):
        return "Softmax"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]
        layer.softmax_param.axis = get_attr_with_default(self.node, "axis", 1)
        return layer


class Sub(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Eltwise"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
        layer.eltwise_param.coeff[:] = [1.0, -1.0]
        return layer


class CaffeSoftmax(Layer, is_register=True):
    def layer_type(self):
        return "Softmax"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]
        layer.softmax_param.axis = get_attr_with_default(self.node, "axis", 1)
        return layer


class Unknown(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Unknown"


class Transpose(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Transpose"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        perm = self.node.get_attribute_value("perm")
        layer.transpose_param.dim.extend(perm)
        return layer


class Mul(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Eltwise"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        assert len(self.node.input) == 2
        lhs_shape = self.node.owning_graph.get_tensor_shape(self.node.input[0])
        rhs_shape = self.node.owning_graph.get_tensor_shape(self.node.input[1])

        if lhs_shape == rhs_shape:
            layer.type = "Eltwise"
            layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.PROD
            return layer

        elif (
            len(lhs_shape) == len(rhs_shape)
            and lhs_shape[1] == rhs_shape[1]
            and (
                all(i == 1 for i in rhs_shape[2:]) or all(i == 1 for i in lhs_shape[2:])
            )
        ):
            layer.type = "Scale"
            layer.scale_param.bias_term = False
            layer.scale_param.axis = 0

            if all(i == 1 for i in rhs_shape[2:]):
                lhs = self.node.input[0]
                rhs = self.node.input[1]
                reshaped_rhs = rhs + "_flatten"
            else:
                lhs = self.node.input[1]
                rhs = self.node.input[0]
                reshaped_rhs = rhs + "_flatten"

            reshape_layer = caffe_pb2.LayerParameter(
                type="Reshape", name=self.layer_name() + "_flat"
            )
            reshape_layer.bottom[:] = [rhs]
            reshape_layer.top[:] = [reshaped_rhs]
            reshape_layer.reshape_param.shape.dim[:] = [0, -1]

            layer.bottom[:] = [lhs, reshaped_rhs]
            return [reshape_layer, layer]

        else:
            layer.type = "Mul"
            return layer

        return layer


class Max(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Eltwise"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.MAX
        return layer


class _reduce(Layer, is_register=False):
    """Base class of converting Reduction Ops."""

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(name=self.node.name, type=self.layer_type())
        layer.bottom.append(self.node.input[0])
        layer.top.append(self.node.output[0])
        param = layer.reduction_param
        param.coeff = 1.0
        # set the axis parameter according to axes attribute.
        axes = get_attr_with_default(self.node, "axes")
        graph = self.node.owning_graph
        nb_dims = len(graph.get_tensor_shape(self.node.input[0]))
        axes = [x % nb_dims for x in axes]
        axis = min(axes)
        # in order to convert to caffe Reduction layer, the `axes` must be continous tail axes.
        if set(axes) != set(range(axis, nb_dims)):
            logger.fatal(
                f"Convert incompatible ONNX OP {self.node.op_type} to caffe Reduction, "
                f"the axes attribute are {axes}, while rank of input is {nb_dims}. "
                f"This may result to incorrect caffe model"
            )
        param.axis = axis
        param.operation = self.get_operation(caffe_pb2)

        keep_dims = self.node.get_attribute_value("keepdims")
        if keep_dims:
            logger.warning(
                f"Attribute 'keep_dims = True' of {self.__class__.__name__} is ignored"
            )

        return layer

    def layer_type(self):
        return "Reduction"

    @abstractmethod
    def get_operation(self, caffe_pb2):
        raise RuntimeError("Derivate class should override this method")


class ReduceReduceL1(_reduce, is_register=True):
    def get_operation(self, caffe_pb2):
        return caffe_pb2.ReductionParameter.ASUM


class ReduceSum(_reduce, is_register=True):
    def get_operation(self, caffe_pb2):
        return caffe_pb2.ReductionParameter.SUM


class ReduceSumSquare(_reduce, is_register=True):
    def get_operation(self, caffe_pb2):
        return caffe_pb2.ReductionParameter.SUMSQ


class ReduceMean(_reduce, is_register=True):
    def get_operation(self, caffe_pb2):
        return caffe_pb2.ReductionParameter.MEAN


class GroupNorm(Layer, is_register=True):
    def layer_type(self):
        return "GroupNorm"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = list(self.node.output)

        layer.group_norm_param.group_num = self.node.get_attribute_value(
            "num_groups", 1
        )
        layer.group_norm_param.eps = self.node.get_attribute_value("eps", 1e-5)

        if len(self.node.input) > 1:
            # with affine
            scale = caffe_pb2.LayerParameter(
                type="Scale", name=self.layer_name() + "_scale"
            )
            layer.top[:] = [self.node.input[0] + "_scale"]
            scale.bottom[:] = layer.top[:]
            scale.top[:] = list(self.node.output)
            scale.blobs.extend(
                [
                    initializer_to_blob(
                        self.node.owning_graph.initializer[self.node.input[1]],
                        caffe_pb2,
                    )
                ]
            )

            if len(self.node.input) > 2:
                scale.scale_param.bias_term = True
                scale.blobs.extend(
                    [
                        initializer_to_blob(
                            self.node.owning_graph.initializer[self.node.input[2]],
                            caffe_pb2,
                        )
                    ]
                )
            return [layer, scale]
        return layer


class HeatMap2Coord(Layer, is_register=True):
    def layer_type(self):
        return "HeatMap2Coord"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = list(self.node.output)

        param = layer.heatmap_param
        param.coord_h = get_attr_with_default(self.node, "coord_h", 0)
        param.coord_w = get_attr_with_default(self.node, "coord_w", 0)
        param.coord_reposition = get_attr_with_default(
            self.node, "coord_reposition", False
        )
        return layer


class Add(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Eltwise"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        layer.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
        layer.eltwise_param.coeff[:] = [1.0, 1.0]
        return layer


class CaffeBatchNorm(Layer, is_register=True):
    def layer_type(self):
        return "BatchNorm"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        batch_norm_param = layer.batch_norm_param
        batch_norm_param.use_global_stats = True
        batch_norm_param.eps = get_attr_with_default(self.node, "eps", 1e-5)

        graph = self.node.owning_graph
        for idx in range(1, 4):
            weight = graph.get_const_tensor_as_array(self.node.input[idx])
            layer.blobs.extend([array_to_blob(weight, caffe_pb2)])

        return layer


class CaffeScale(Layer, is_register=True):
    def layer_type(self):
        return "Scale"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = [self.node.output[0]]

        scale_param = layer.scale_param
        scale_param.axis = get_attr_with_default(self.node, "axis", 1)

        graph = self.node.owning_graph

        if self.node.input[1] in graph.initializer:
            scale = graph.get_const_tensor_as_array(self.node.input[1])
            layer.blobs.extend([array_to_blob(scale, caffe_pb2)])
            scale_param.num_axes = len(scale.shape)
        else:
            # dual input scale, which represents broadcast multiply
            layer.bottom.append(self.node.input[1])

        if len(self.node.input) >= 3:
            scale_param.bias_term = True
            bias = graph.get_const_tensor_as_array(self.node.input[2])
            layer.blobs.extend([array_to_blob(bias, caffe_pb2)])
        else:
            scale_param.bias_term = False

        return layer


class DynamicUpsample(Upsample, is_register=True):
    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = self.node.output[:]
        return layer


class PODRoiAlign(Layer, is_register=True):
    def layer_type(self):
        return "PODROIAlignPooling"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        pool_param = layer.podroi_align_pooling_param
        pool_param.pooled_h = get_attr_with_default(self.node, "pooled_height", 0)
        pool_param.pooled_w = get_attr_with_default(self.node, "pooled_width", 0)
        pool_param.sample_num = get_attr_with_default(self.node, "sample_num", 1)
        pool_param.spatial_scale = get_attr_with_default(self.node, "spatial_scale", 1)

        return layer


class RoiAlign(Layer, is_register=True):
    def layer_type(self):
        return "ROIAlignPooling"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        pool_param = layer.roi_align_pooling_param
        pool_param.pooled_h = get_attr_with_default(self.node, "pooled_height", 0)
        pool_param.pooled_w = get_attr_with_default(self.node, "pooled_width", 0)
        pool_param.sample_num = get_attr_with_default(self.node, "sample_num", 1)
        pool_param.spatial_scale = get_attr_with_default(self.node, "spatial_scale", 1)

        return layer


class MaxRoiPool(Layer, is_register=True):
    def layer_type(self):
        return "ROIPooling"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output

        pool_param = layer.roi_pooling_param
        pooled_shape = self.node.get_attribute_value("pooled_shape")
        pool_param.pooled_h = pooled_shape[0]
        pool_param.pooled_w = pooled_shape[1]
        pool_param.spatial_scale = self.node.get_attribute_value("spatial_scale")

        return layer


class PSRoiPool(Layer, is_register=True):
    def layer_type(self):
        return "PSROIPooling"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = self.node.output[0]

        param = layer.psroi_pooling_param
        param.output_dim = self.node.get_attribute_value("output_dim")
        param.group_size = self.node.get_attribute_value("group_size")
        param.spatial_scale = self.node.get_attribute_value("spatial_scale", 1.0)

        return layer


class RoiPool(Layer, is_register=True):
    def layer_type(self):
        return "ROIPooling"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = self.node.output[0]

        param = layer.roi_pooling_param
        param.pooled_h = self.node.get_attribute_value("pooled_height")
        param.pooled_w = self.node.get_attribute_value("pooled_width")
        param.spatial_scale = self.node.get_attribute_value("spatial_scale", 1.0)

        return layer


class Split(Layer, is_register=True):
    def layer_type(self):
        return "Slice"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output
        param = layer.slice_param
        param.axis = self.node.get_attribute_value("axis")
        splits = self.node.get_attribute_value("split")
        # accumulate split sizes to get slice points
        from itertools import accumulate

        slice_points = list(accumulate(splits))
        slice_points = slice_points[:-1]
        param.slice_point[:] = slice_points
        return layer


class MeanVarianceNormalization(Layer, is_register=True):
    def layer_type(self):
        return "MVN"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output
        param = layer.mvn_param
        # MeanVarianceNormalization op always normalize variance.
        param.normalize_variance = True
        node = self.node
        graph = node.owning_graph
        ndim = len(graph.get_tensor_shape(node.input[0]))
        axes = set(node.get_attribute_value("axes"))

        def set_equal(a, b):
            a = set(a)
            b = set(b)
            return len(a.difference(b)) == 0 and len(b.difference(a)) == 0

        if set_equal(axes, range(1, ndim)):
            param.across_channels = True
        elif set_equal(axes, range(2, ndim)):
            param.across_channels = False
        else:
            logger.error(
                f"can not convert MeanVarianceNormalization op whose inputs is {ndim}D, and axes={axes} to caffe layer"
            )
        return layer


class TopK(Layer, is_register=True):
    def layer_type(self):
        return "ArgMax"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        node = self.node
        graph = node.owning_graph
        value_used = len(graph.get_tensor_consumer(node.output[0])) != 0
        if value_used:
            # TopK's output is [Values, Indices], while ArgMax's output is (argmax, maxval).
            layer.top[:] = [self.node.output[1], self.node.output[0]]
        else:
            layer.top[:] = [self.node.output[1]]
        param = layer.argmax_param
        param.out_max_val = value_used
        param.top_k = self.node.get_attribute_value("k")
        param.axis = self.node.get_attribute_value("axis")

        return layer


class CaffePower(Layer, is_register=True):
    def layer_type(self):
        return "Power"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output
        node = self.node
        param = layer.power_param
        for attr in ["power", "scale", "shift"]:
            setattr(param, attr, node.get_attribute_value(attr))
        return layer


class Pow(Layer, is_register=True):
    def layer_type(self):
        return "Power"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = [self.node.input[0]]
        layer.top[:] = self.node.output
        node = self.node
        param = layer.power_param
        for attr in ["power", "scale", "shift"]:
            setattr(param, attr, node.get_attribute_value(attr))
        return layer


class CaffeThreshold(Layer, is_register=True):
    def layer_type(self):
        return "Threshold"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output
        node = self.node
        layer.threshold_param.threshold = node.get_attribute_value("threshold")
        return layer


class Parameter(Layer, is_register=True):
    def layer_type(self):
        return "Parameter"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output
        node = self.node
        graph = node.owning_graph
        param = layer.parameter_param
        weight = numpy_helper.to_array(node.get_attribute_value("value"))
        param.batch = node.get_attribute_value("batch")
        param.channel = node.get_attribute_value("channel")
        param.height = node.get_attribute_value("height")
        param.width = node.get_attribute_value("width")
        param.n = node.get_attribute_value("n")
        param.m = node.get_attribute_value("m")
        layer.blobs.extend([array_to_blob(weight, caffe_pb2)])
        return layer


class Abs(Layer, is_register=True):
    def layer_type(self):
        return "Abs"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input
        layer.top[:] = self.node.output
        return layer


class Reciprocal(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Reciprocal"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        return layer


class Warp(NonParamLayer, is_register=True):
    def layer_type(self):
        return "Warp"

    def parse(self, caffe_pb2):
        layer = super().parse(caffe_pb2)
        layer.bottom[:] = self.node.input[:]
        layer.top[:] = [self.node.output[0]]

        return layer


class ConvexUpsample(Layer, is_register=True):
    def layer_type(self):
        return "ConvexUpsample"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:2]
        layer.top[:] = [self.node.output[0]]

        layer.convexupsample_param.scale = self.node.get_attribute_value("scale")

        return layer


class CostVolume(Layer, is_register=True):
    def layer_type(self):
        return "CostVolume"

    def parse(self, caffe_pb2):
        layer = caffe_pb2.LayerParameter(type=self.layer_type(), name=self.layer_name())
        layer.bottom[:] = self.node.input[:2]
        layer.top[:] = [self.node.output[0]]

        layer.costvolume_param.max_disp = self.node.get_attribute_value("max_disp")
        layer.costvolume_param.index_num = self.node.get_attribute_value("index_num")
        layer.costvolume_param.bias_term = self.node.get_attribute_value("bias_term")

        # process weight, bias, index
        for i in self.node.input[2:]:
            w = self.node.owning_graph.get_const_tensor_as_array(i)
            layer.blobs.extend([array_to_blob(w, caffe_pb2)])
        return layer
