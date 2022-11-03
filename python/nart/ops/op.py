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

from enum import IntEnum
from enum import Enum
import warnings
import copy
import onnx
from onnx.onnx_pb import TensorProto, AttributeProto
from onnx import NodeProto
from onnx import helper
from onnx import numpy_helper
from ..core.graph import Node
from ..core import Context
from ..core.art import Dtype
import math
import numpy as np

from functools import reduce
from operator import mul, add

# add an op (frontend:caffe,ir:onnx,backend:default)
# step 1  add op def in op.py, should add attr_dict and infer_shape func
# step 2  add a convert class in caffe_to_onnx
# step 3 optional, you may implement a function to merge some nodes
# step 4 finish make layer in tools or in caffe to onnx
# step 5 add convert class in default.py

import logging

logger = logging.getLogger("nart.op")


class GroupType(Enum):
    ONNX = 1
    TOOLS = 2
    CASE = 3


class OpDesc:
    class AttrType(Enum):
        UNDEFINED = 0
        FLOAT = 1
        INT = 2
        STRING = 3
        TENSOR = 4
        GRAPH = 5
        SPARSE_TENSOR = 11

        FLOATS = 6
        INTS = 7
        STRINGS = 8
        TENSORS = 9
        GRAPHS = 10
        SPARSE_TENSORS = 12

    def attr(tp, default=None):
        class _attr_wrapper:
            """attribute decorator
            Usage:
                class Conv(OpDesc):
                    ...
                    @OpDesc.attr(tp=OpDesc.INTS, default=3)
                    def kernel_size(self):
                        return self.kernel_size > 0

                    @OpDesc.attr(tp=OpDesc.INT)
                    def num_output(self):
                        return self.num_output > 0

            """

            def __init__(self, func):
                self.func = func

            def __set_name__(self, owner, name):
                """ """

                import inspect
                import ast
                import astor
                import textwrap

                code = textwrap.dedent(inspect.getsource(self.func))
                mod = ast.parse(code)

                assert isinstance(mod.body[0], ast.FunctionDef)
                target_mod = []
                for m in mod.body[0].body:
                    if isinstance(m, ast.Return):
                        target_mod.append(ast.Expr(m.value))
                    else:
                        target_mod.append(m)
                target_mod = ast.Module(target_mod)
                src = astor.to_source(target_mod)
                if not hasattr(owner, "_desc"):
                    setattr(owner, "_desc", {})
                owner._desc[name] = (tp, src, default)

                # add args to original function and bind it to self.new_func
                if len(mod.body[0].args.args) == 0:
                    mod.body[0].args.args.append(ast.arg("cls", None))
                    mod.body[0].args.args.append(ast.arg(name, None))
                new_method_name = (
                    owner.__module__.replace(".", "_")
                    + "_"
                    + owner.__name__
                    + "_"
                    + name
                )
                mod.body[0].name = new_method_name
                mod.body[0].decorator_list = []
                m = compile(astor.to_source(mod), "op_desc", "exec")
                exec(m)
                exec(f"self.new_func = {new_method_name}")

                @classmethod
                def wrapper(cls, val):
                    return self.new_func(cls, val)

                setattr(owner, name, wrapper)

        return _attr_wrapper

    @classmethod
    def __init_subclass__(
        cls,
        parser=None,
        op_type: str = None,
        domain: str = "",
        version: int = 9,
        **kwargs,
    ):
        cls._domain = domain
        cls._version = version
        if op_type is None:
            op_type = cls.__name__
        parser.register_op(op_type, cls)
        if not hasattr(cls, "_desc"):
            cls._desc = {}
        super().__init_subclass__(**kwargs)

    @classmethod
    def support(cls, op):
        attributes = op.attributes
        check = True
        for k, v in attributes.items():
            if k in cls._desc:
                val = helper.get_attribute_value(v)
                if v.type == AttributeProto.STRING:
                    val = val.decode()
                elif v.type == AttributeProto.STRINGS:
                    val = [x.decode() for x in val]
                check = check and getattr(cls, k)(val)
        return check

    @classmethod
    def docstring(cls):
        return cls._desc


# the delimiter to seperate node name parts, to indicate a node is sub-node of some original node.
# for example, during SumToAdd pass, several add nodes may have to be created from sum node (suppose its name is N),
# the add nodes's name should be like N::add0, N::add1 and so on (when DELIM IS '::').
DELIM = "::"


class DataType(IntEnum):
    UNDEFINED = 0
    # Basic types.
    FLOAT = 1  # float
    UINT8 = 2  # uint8_t
    INT8 = 3  # int8_t
    UINT16 = 4  # uint16_t
    INT16 = 5  # int16_t
    INT32 = 6  # int32_t
    INT64 = 7  # int64_t
    STRING = 8  # string
    BOOL = 9  # bool

    #  IEEE754 half-precision floating-point format (16 bits wide).
    #  This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10

    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14  # complex with float32 real and imaginary components
    COMPLEX128 = 15  # complex with float64 real and imaginary components

    # Non-IEEE floating-point format based on IEEE754 single-precision
    # floating-point number truncated to 16 bits.
    # This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16

    #  Future extensions go here.


class Op(Node):
    """base class of onnx op def.

    `ONNX_OP`: onnx op def dict.

    `attr_dict`: attribute definition.

    """

    GROUP = GroupType.ONNX

    # automatically register op in ONNX_OP
    # the key is op_type, the value is list of op classes.
    ONNX_OP = {}

    # val type
    NO_CHECK = 0
    INT = 1
    FLOAT = 2
    STRING = 3
    LIST_OF_INTS = 4
    LIST_OF_FLOATS = 5
    LIST_OF_STRINGS = 6
    TENSOR = 7
    BOOL = 8

    # op attr dict
    attr_dict = {}
    # {
    #    'attr_name_1' : (Node.INT, default val),
    #    'attr_name_2' : (Node.LIST_OF_STRINGS),
    #    'attr_name_3' : (Node.FLOAT)
    # }

    _op_type = "Op"

    @classmethod
    def __init_subclass__(
        cls,
        op_type=None,
        is_abstract=False,
        domain: str = "",
        version: int = 9,
        **kwargs,
    ):
        """register onnx op def in ONNX_OP.

        Args:
            op_type: str.
        """
        if not is_abstract:
            if op_type is None:
                op_type = cls.__name__
            Op.ONNX_OP.setdefault(op_type, list())
            Op.ONNX_OP[op_type].append(cls)
        cls._domain = domain
        cls._version = version
        cls._op_type = op_type
        super().__init_subclass__(**kwargs)

    def __init__(self, name):
        super(Op, self).__init__()

        self._name = name
        self._op_type = self.__class__._op_type

        # self.input_and_weight = [] #存放inputname and weightname
        # self.output = [] #存放output_name

        # self.attr_dict = {}

    def satisfy(self, insure_type_mark=False):
        """Check that the current information whether satisfies the definition.

        Args:
            insure_type_make: bool,check the attr type if True.

        """
        if self.__class__.__name__ == "Op":
            # 基类不需要检查
            return True
        # 是否需要检查类型
        if insure_type_mark == True:
            if not self.insure_type():
                return False
        support_dict = {}
        for k, v in self.__class__.attr_dict.items():
            if len(v) > 1:
                support_dict[k] = True
            else:
                support_dict[k] = False
        for k in self.attributes:
            if k in support_dict:
                support_dict[k] = True
            else:
                warnings.warn(
                    f"op {self.__class__.__name__} can not support attribute {k}"
                )
        for k, val in support_dict.items():
            if val is False:
                return False
        return True

    def insure_type(self):
        """check attr type."""
        for k in self.attr_dict:
            val = helper.get_attribute_value(k)
            k = k.name
            if k in self.__class__.attr_dict:
                val_type = self.__class__.attr_dict[k]
                if val_type == Op.NO_CHECK:
                    continue
                elif val_type == Op.INT:
                    if not isinstance(val, int):
                        return False
                elif val_type == Op.FLOAT:
                    if not isinstance(val, float):
                        return False
                elif val_type == Op.STRING:
                    if not isinstance(val, str):
                        return False
                elif val_type == Op.LIST_OF_INTS:
                    if not isinstance(val, list):
                        return False
                    for sub_val in val:
                        if not isinstance(sub_val, int):
                            return False
                elif val_type == Op.LIST_OF_FLOATS:
                    if not isinstance(val, list):
                        return False
                    for sub_val in val:
                        if not isinstance(sub_val, float):
                            return False
                elif val_type == Op.LIST_OF_STRINGS:
                    if not isinstance(val, list):
                        return False
                    for sub_val in val:
                        if not isinstance(sub_val, float):
                            return False
                elif val_type == Op.TENSOR:
                    if not isinstance(val, TensorProto):
                        return False
                elif val_type == Op.BOOL:
                    if not isinstance(val, bool):
                        return False
                else:
                    return False
        return True

    def infer_shape(self):
        """default implementation: pass"""
        raise NotImplementedError(
            f"{self.__class__.__name__}.infer_shape was not implemented"
        )

    @staticmethod
    def from_onnx_node(node, opsets=[("", 9)]):
        """generate onnx op def

        Args:
            node: onnx.NodeProto
        """
        # Try find the first imported opset in which this op is implemented.
        # print(opsets)

        domain, version = "", 9
        for _domain, _version in opsets:
            for item in Op.ONNX_OP.get(node.op_type, list()):
                if item._domain == _domain and item._version <= _version:
                    domain, version = _domain, _version
                    break
        # NOTE: A hardcode patch for non-standard Max/AveragePool.
        # Old version of nart export non-standard Pool operators, this is historical reason.
        if node.op_type in ["MaxPool", "AveragePool"] and version < 10:
            ceil_mode = 0
            for attr in node.attribute:
                if attr.name == "ceil_mode":
                    ceil_mode = attr.i
            if ceil_mode == 1:
                logger.warning(
                    "non-standard {0}-{1} operator encountered, using {0}-10 instead".format(
                        node.op_type, version
                    )
                )
                version = 10

        onnx_op = Op.gen_op(
            node.op_type,
            node.name if node.name != "" else node.op_type + "_" + node.output[0],
            domain=domain,
            version=version,
        )
        [onnx_op.add_attribute(attr) for attr in node.attribute]
        [onnx_op.add_input(i) for i in node.input]
        [onnx_op.add_output(o) for o in node.output]
        return onnx_op

    @staticmethod
    def gen_op(op_type, name, domain="", version=9):
        """generate onnx op def

        Args:
            op_type: str.
            name: str.
        """
        if op_type in Op.ONNX_OP:
            # find the class corresponds to given domain and version
            cls = None
            for item in Op.ONNX_OP[op_type]:
                if (
                    domain == item._domain
                    and version >= item._version
                    and (cls is None or item._version > cls._version)
                ):
                    # find the op class whose domain is same as given domain, and whose version is the highest
                    # among those lower than given version, that is, `other._version < cls._version <= version`
                    cls = item
            if cls is not None:
                return cls(name)
            logger.warning(
                f"No corresponding version for op `{op_type} (domain={domain}, version={version})`"
            )

        logger.warning(f'unhandled op_type "{op_type}" in Op.gen_op')
        return Op(name)

    def get_attribute_value(self, name, default=None):
        """Get attribute value by name.
        If not specified, default is returned if provided, otherwise KeyError is raised.
        """
        if name not in self.attr_dict:
            logger.warning(
                f"try getting attribute `{name}` of node {self.name} [{self.op_type}], "
                f"whose attr_dict does not have this attribute."
            )

        if name not in self.attributes:
            if default is not None:
                return default
            if name in self.attr_dict and len(self.attr_dict[name]) >= 2:
                # using default value in op attr_dict
                return self.attr_dict[name][1]
            else:
                raise KeyError(f'no attribute named "{name}" in Node "{self.name}"')
        else:
            value = helper.get_attribute_value(self.attributes[name])
            # convert from bytes to str
            value = value.decode("utf-8") if isinstance(value, bytes) else value
            return value

    @staticmethod
    def make_op(op_type, name, inputs, outputs, attributes=None, domain="", version=9):
        """Make a onnx op node.

        Args:
            op_type (str): Op type.
            name (str): Op name.
            inputs (list<str>): Node input names.
            outputs (list<str>): Node output name.
            attribtues (dict<str, any>): Dict of attribtues.
        """
        ret = Op.gen_op(op_type, name, domain=domain, version=version)
        ret.input[:] = inputs
        ret.output[:] = outputs
        if attributes:
            for name, value in attributes.items():
                if isinstance(value, onnx.AttributeProto):
                    attr = value
                else:
                    try:
                        attr = helper.make_attribute(name, value)
                    except (ValueError, TypeError):
                        logger.info(
                            f"{value} is not valid onnx attribute type, using original object instead of creating onnx.AttributeProto"
                        )
                        attr = value
                assert ret.add_attribute(attr)
        return ret

    @property
    def domain(self):
        """Get the domain of the operator set being identified.
        The empty string ("") or absence of this field implies the operator set
        that is defined as part of the ONNX specification.
        """
        return self.__class__._domain

    @property
    def version(self):
        """Get the version of the operator set being identified."""
        return self.__class__._version


class PassThroughOp(Op, is_abstract=True):
    """A type of Op which does a pass through computation,
    so all outputs' shape is same as the shape of first (also the only) input.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def infer_shape(self):
        """Pass through shape infer, all outputs' shape set to the same as the first"""
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        for output in self.output:
            graph.set_tensor_shape(output, shape, dtype)


class BroadcastOp(Op, is_abstract=True):
    """A type of Op which does supports **multidirectional (i.e., Numpy-style) broadcasting**."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def infer_shape(self):
        """Broadcast shape infer."""
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        for tensor in self.input[1:]:
            shape = BroadcastOp.broadcast_shape(shape, graph.get_tensor_shape(tensor))
        for output in self.output:
            graph.set_tensor_shape(output, shape, dtype)

    @staticmethod
    def broadcast_shape(a, b):
        """Infer the output tensor shape after broadcast operation on two operands.

        Args:
            a (list<int>): shape of first operand.
            b (list<int>): shape of second operand.

        Returns:
            list<int>: the output shape after broadcasting.
        """
        ret = []
        out_dim = max(len(a), len(b))
        # fill ones in the front of a/b to align them.
        a = [1] * (out_dim - len(a)) + a
        b = [1] * (out_dim - len(b)) + b
        for x, y in zip(a, b):
            if x == y:
                ret.append(x)
            elif x == 1:
                ret.append(y)
            elif y == 1:
                ret.append(x)
            else:
                raise ValueError(f"cannot broadcast the two shapes: {a} vs {b}")
        return ret


class Conv(Op):

    GROUP = GroupType.ONNX

    def infer_shape(self):
        attributes = self.attributes
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        weight_shape = self.owning_graph.get_tensor_shape(self.input[1])
        assert len(input_shape) == len(
            weight_shape
        ), "input_shape and weight_shape must match in Conv"
        if len(input_shape) == 4:
            n, c, h, w = input_shape
            assert self.get_attribute_value("auto_pad") == "NOTSET"
            pads = self.get_attribute_value("pads", [0, 0, 0, 0])
            pre_pad = pads[0:2]
            post_pad = pads[2:4]
            kernel_shape = self.get_attribute_value("kernel_shape", [])
            if kernel_shape:
                kernel_h, kernel_w = kernel_shape
            else:
                # infer kernel shape from weight shape
                kernel_h, kernel_w = weight_shape[2], weight_shape[3]
            stride_h, stride_w = self.get_attribute_value("strides", [1, 1])

            dilations = self.get_attribute_value("dilations", [1, 1])
            dilation_h, dilation_w = dilations

            kernel_h_eff = kernel_h + (kernel_h - 1) * (dilation_h - 1)
            kernel_w_eff = kernel_w + (kernel_w - 1) * (dilation_w - 1)
            h = math.floor((h + pre_pad[0] + post_pad[0] - kernel_h_eff) / stride_h + 1)
            w = math.floor((w + pre_pad[1] + post_pad[1] - kernel_w_eff) / stride_w + 1)

            c = weight_shape[0]

            self.owning_graph.set_tensor_shape(self.output[0], [n, c, h, w])
        else:
            output_shape = input_shape
            assert self.get_attribute_value("auto_pad") == "NOTSET"
            pads = self.get_attribute_value(
                "pads", [0 for i in range(2 * (len(input_shape) - 2))]
            )
            pre_pad = pads[0 : len(pads) // 2]
            post_pad = pads[len(pads) // 2 :]
            kernel_shape = self.get_attribute_value("kernel_shape", [])
            if kernel_shape:
                pass
            else:
                kernel_shape = weight_shape[2:]
            stride = self.get_attribute_value(
                "strides", [1 for i in range(len(input_shape) - 2)]
            )

            dilations = self.get_attribute_value(
                "dilations", [1 for i in range(len(input_shape) - 2)]
            )
            kernel_eff = [0 for i in range(len(kernel_shape))]
            for i in range(len(kernel_shape)):
                kernel_eff[i] = kernel_shape[i] + (kernel_shape[i] - 1) * (
                    dilations[i] - 1
                )
            for i in range(2, len(output_shape)):
                output_shape[i] = math.floor(
                    (
                        output_shape[i]
                        + pre_pad[i - 2]
                        + post_pad[i - 2]
                        - kernel_eff[i - 2]
                    )
                    / stride[i - 2]
                    + 1
                )
            output_shape[1] = weight_shape[0]
            self.owning_graph.set_tensor_shape(self.output[0], output_shape)

    attr_dict = {
        "dilations": (Op.LIST_OF_STRINGS,),
        "group": (Op.INT, 1),
        "kernel_shape": (Op.LIST_OF_INTS,),
        "pads": (Op.LIST_OF_INTS,),
        "strides": (Op.LIST_OF_INTS,),
        "auto_pad": (Op.STRING, "NOTSET"),
    }


class ConvTranspose(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        "dilations": (Op.LIST_OF_STRINGS,),
        "group": (Op.INT, 1),
        "kernel_shape": (Op.LIST_OF_INTS,),
        "pads": (Op.LIST_OF_INTS,),
        "strides": (Op.LIST_OF_INTS,),
        "auto_pad": (Op.STRING, "NOTSET"),
        "output_padding": (Op.LIST_OF_INTS, 0),
    }

    def infer_shape(self):
        attributes = self.attributes
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        assert len(input_shape) == 4, "Can only support 2-D {self.__class__.__name__}"
        n, c, h, w = input_shape

        pads = self.get_attribute_value("pads", [0, 0, 0, 0])
        pre_pad = pads[0:2]
        post_pad = pads[2:4]
        kernel_h, kernel_w = self.get_attribute_value("kernel_shape")
        stride_h, stride_w = self.get_attribute_value("strides", [1, 1])

        h = stride_h * (h - 1) + kernel_h - (pre_pad[0] + post_pad[0])
        w = stride_w * (w - 1) + kernel_w - (pre_pad[1] + post_pad[1])

        weight_shape = self.owning_graph.get_tensor_shape(self.input[1])
        c = weight_shape[1] * self.get_attribute_value("group", 1)
        self.owning_graph.set_tensor_shape(self.output[0], [n, c, h, w])


class BatchNormalization(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        "epsilon": (Op.FLOAT, 1e-5),
        "momentum": (Op.FLOAT, 0.9),
        "training_mode": (Op.BOOL, False),
    }


class InstanceNormalization(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {"epsilon": (Op.FLOAT, 1e-5)}


class MatMul(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    def infer_shape(self):
        shape1 = self.owning_graph.get_tensor_shape(self.input[0])
        shape2 = self.owning_graph.get_tensor_shape(self.input[1])
        assert len(shape1) >= 2 and len(shape2) >= 2, "1D matmul not handled"
        batch = BroadcastOp.broadcast_shape(shape1[:-2], shape2[:-2])
        # no transpose, so layout is determined.
        M, K = shape1[-2:]
        K1, N = shape2[-2:]
        if K != K1:
            logger.error(
                f"The reduce axis (K) of matmul's input shapes are not consistent: {shape1} vs {shape2}"
            )
        output_shape = batch + [M, N]
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


""" Family of binary ops """


class Max(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Min(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Mul(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    @staticmethod
    def make_mul(name, x, y, output):
        return Op.make_op("Mul", name, [x, y], [output])


class Div(BroadcastOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Add(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    @staticmethod
    def make_add(name, x, y, output):
        return Op.make_op("Add", name, [x, y], [output])


class Sub(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Greater(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class ArgMax(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        "axis": (Op.INT, 0),
        "keepdims": (Op.INT, 1),
        "select_last_index": (Op.INT, 0),
    }

    def infer_shape(self):
        """ArgMax shape infer."""
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])

        keepdims = self.get_attribute_value("keepdims", 1)
        axis = self.get_attribute_value("axis")

        out_shape = list(shape)
        if keepdims == 1:
            out_shape[axis] = 1
        else:
            del out_shape[axis]

        # if output shape is empty, it actually should be [1].
        out_shape = out_shape if len(out_shape) != 0 else [1]
        self.owning_graph.set_tensor_shape(self.output[0], out_shape)


class ArgMin(ArgMax):
    pass


class PoolBase(Op, is_abstract=True):
    def __init_subclass__(
        cls,
        op_type=None,
        is_abstract=False,
        domain: str = "",
        version: int = 9,
        **kwargs,
    ):
        return super().__init_subclass__(
            op_type, is_abstract, domain, version, **kwargs
        )

    def infer_shape(self):
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        nb_spatial = len(input_shape) - 2
        pads = self.get_attribute_value("pads", [0] * (nb_spatial * 2))
        pre_pad = pads[0:nb_spatial]
        post_pad = pads[nb_spatial:]
        kernel_shape = self.get_attribute_value("kernel_shape")
        strides = self.get_attribute_value("strides", [1] * nb_spatial)
        if "ceil_mode" not in self.attr_dict and "ceil_mode" in self.attributes:
            logger.error(
                "non-standard {0} operator, ceil_mode is not defined before opset-10".format(
                    self.op_type
                )
            )
        ceil_mode = "ceil_mode" in self.attr_dict and self.get_attribute_value(
            "ceil_mode", 0
        )

        input_spatial_shape = input_shape[2:]
        if ceil_mode:
            spatial_dims = [
                math.ceil(
                    (
                        input_spatial_shape[k]
                        + pre_pad[k]
                        + post_pad[k]
                        - kernel_shape[k]
                    )
                    / strides[k]
                    + 1
                )
                for k in range(nb_spatial)
            ]
        else:
            spatial_dims = [
                math.floor(
                    (
                        input_spatial_shape[k]
                        + pre_pad[k]
                        + post_pad[k]
                        - kernel_shape[k]
                    )
                    / strides[k]
                    + 1
                )
                for k in range(nb_spatial)
            ]

        # ensure the last pooling starts strictly inside the image
        if any(pads):
            for k in range(nb_spatial):
                if (spatial_dims[k] - 1) * strides[k] >= input_spatial_shape[
                    k
                ] + pre_pad[k]:
                    spatial_dims[k] -= 1

        self.owning_graph.set_tensor_shape(
            self.output[0], input_shape[:2] + spatial_dims
        )


class AveragePool_9(PoolBase, op_type="AveragePool", version=9):
    GROUP = GroupType.ONNX

    attr_dict = {
        "auto_pad": (Op.STRING, "NOTSET"),
        "kernel_shape": (Op.LIST_OF_INTS,),
        "pads": (Op.LIST_OF_INTS,),
        "strides": (Op.LIST_OF_INTS,),
        "count_include_pad": (Op.INT, 0),
    }


class AveragePool_10(PoolBase, op_type="AveragePool", version=10):
    GROUP = GroupType.ONNX

    attr_dict = {
        "auto_pad": (Op.STRING, "NOTSET"),
        "ceil_mode": (Op.INT, 0),
        "kernel_shape": (Op.LIST_OF_INTS,),
        "pads": (Op.LIST_OF_INTS,),
        "strides": (Op.LIST_OF_INTS,),
        "dilations": (Op.LIST_OF_INTS, (1, 1)),
        "count_include_pad": (Op.INT, 0),
    }

    @staticmethod
    def make_average_pool(
        name,
        input,
        output,
        kernel_shape,
        pads,
        strides,
        ceil_mode=0,
        auto_pad="NOTSET",
        count_include_pad=0,
    ):
        node = Op.gen_op("AveragePool", name)
        node.add_input(input)
        node.add_output(output)
        attrs = {
            "kernel_shape": kernel_shape,
            "pads": pads,
            "strides": strides,
            "ceil_mode": ceil_mode,
            "auto_pad": auto_pad,
            "count_include_pad": count_include_pad,
        }
        for name, value in attrs.items():
            node.add_attribute(helper.make_attribute(name, value))
        return node


class QuantDequant(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        "q_min": (Op.LIST_OF_INTS, [-128]),
        "q_max": (Op.LIST_OF_INTS, [127]),
        "scale": (Op.LIST_OF_FLOATS, [1.0]),
    }

    def infer_shape(self):
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        self.owning_graph.set_tensor_shape(self.output[0], input_shape)
        return input_shape

    @staticmethod
    def make_quant_dequant(name, input, output, q_min, q_max, scale):
        node = Op.gen_op("QuantDequant", name)
        node.add_input(input)
        node.add_output(output)
        attrs = {"q_min": q_min, "q_max": q_max, "scale": scale}
        for name, value in attrs.items():
            node.add_attribute(helper.make_attribute(name, value))
        return node


class MaxPool(PoolBase):
    attr_dict = {
        "auto_pad": (Op.STRING, "NOTSET"),
        "kernel_shape": (Op.LIST_OF_INTS,),
        "pads": (Op.LIST_OF_INTS,),
        "strides": (Op.LIST_OF_INTS,),
    }

    @staticmethod
    def make_max_pool(
        name, input, output, kernel_shape, pads, strides, ceil_mode=0, auto_pad="NOTSET"
    ):
        node = Op.gen_op("MaxPool", name)
        node.add_input(input)
        node.add_output(output)
        attrs = {
            "kernel_shape": kernel_shape,
            "pads": pads,
            "strides": strides,
            "ceil_mode": ceil_mode,
            "auto_pad": auto_pad,
        }
        for name, value in attrs.items():
            node.add_attribute(helper.make_attribute(name, value))
        return node


class MaxPool_10(PoolBase, op_type="MaxPool", version=10):
    attr_dict = {
        "auto_pad": (Op.STRING, "NOTSET"),
        "ceil_mode": (Op.INT, 0),
        "kernel_shape": (Op.LIST_OF_INTS,),
        "pads": (Op.LIST_OF_INTS,),
        "strides": (Op.LIST_OF_INTS,),
        "dilations": (Op.LIST_OF_INTS, (1, 1)),
    }


class Relu(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class ReLU3d(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {"channel_shared": (Op.BOOL, False), "negative_slope": (Op.FLOAT, 0.0)}


class PRelu(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Erf(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Identity(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Sqrt(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Log(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Exp(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Tanh(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Sigmoid(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Squeeze(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"axes": (Op.LIST_OF_INTS,)}

    def infer_shape(self):
        graph = self.owning_graph
        ipt_shape = graph.get_tensor_shape(self.input[0])
        top_shape = list(graph.get_tensor_shape(self.input[0]))
        dtype = graph.get_tensor_dtype(self.input[0])
        axes = [idx for idx, axis in enumerate(ipt_shape) if axis == 1]
        axes = self.get_attribute_value("axes", axes)
        mask = [True] * len(top_shape)
        for axis in axes:
            mask[axis] = False
        for idx in range(len(mask), 0, -1):
            if not mask[idx - 1]:
                assert top_shape[idx - 1] == 1, "?"
                del top_shape[idx - 1]

        self.owning_graph.set_tensor_shape(self.output[0], top_shape, dtype)


class Concat(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"axis": (Op.INT,)}

    def infer_shape(self):
        graph = self.owning_graph
        attributes = self.attributes

        top_shape = list(self.owning_graph.get_tensor_shape(self.input[0]))
        dtype = self.owning_graph.get_tensor_dtype(self.input[0])
        axis = attributes["axis"].i
        axis = axis % len(top_shape)
        top_shape[axis] = 0

        for bottom in self.input:
            top_shape[axis] += self.owning_graph.get_tensor_shape(bottom)[axis]
        ndim = len(top_shape)
        for ipt in self.input:
            ipt_shape = self.owning_graph.get_tensor_shape(ipt)
            if any(t != axis and ipt_shape[t] != top_shape[t] for t in range(ndim)):
                ipt0 = self.input[0]
                ipt0_shape = self.owning_graph.get_tensor_shape(ipt0)
                logger.error(
                    "Concat layer's inputs should have same shape except on the concat axis"
                    f"but layer `{self.name}` is: {ipt0_shape} ({ipt0}) vs {ipt_shape} ({ipt})"
                )

        self.owning_graph.set_tensor_shape(self.output[0], top_shape, dtype)

    def to_ndarray(self):
        def _get_const_array(node, _input):
            if node == "input":
                ret = numpy_helper.to_array(self.owning_graph.initializer[_input])
            else:
                ret = node.to_ndarray()
            return ret

        try:
            input_values = []
            for _input in self.input:
                input_node = self.owning_graph.get_tensor_producer(_input)[-1]
                input_values.append(_get_const_array(input_node, _input))
            return np.concatenate(input_values).astype(input_values[0].dtype)
        except:
            raise NotImplementedError(f"to_ndarray not implemented for {self.name}.")


class Reshape(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    def infer_shape(self):
        graph = self.owning_graph
        from functools import reduce

        bottom_shape = list(self.owning_graph.get_tensor_shape(self.input[0]))
        dtype = self.owning_graph.get_tensor_dtype(self.input[0])

        if len(self.input) > 1:
            shape = graph.get_const_tensor_as_array(self.input[1])
            top_shape = list(shape)
        else:
            top_shape = helper.get_attribute_value(self.attributes["shape"])
            # raise RuntimeError(f'output shape not found in node[`{self.name}`]\'s input')

        infer_idx = None
        for idx in range(len(top_shape)):
            dim = top_shape[idx]
            if dim == -1:
                infer_idx = idx
            elif dim == 0:
                top_shape[idx] = bottom_shape[idx]
        if infer_idx is not None:
            num_items = reduce(lambda a, b: a * b, bottom_shape)
            accounted_items = (
                1
                if len(top_shape) == 1
                else reduce(lambda a, b: a * b, [dim for dim in top_shape if dim != -1])
            )
            top_shape[infer_idx] = num_items // accounted_items
        self.owning_graph.set_tensor_shape(self.output[0], top_shape, dtype)

    @staticmethod
    def make_reshape(name, inp, out, shape):
        """Make a reshape op given input, output and output shape.

        Returns:
            list<Op>: the new reshape op along with its reshape constant op.
        """
        if not isinstance(shape, np.ndarray):
            shape = np.array(shape, dtype=np.int64)
        shape_const_name = name + "_shape_const"
        shape_op = Constant.make_constant(shape_const_name, shape)
        ret = Op.make_op("Reshape", name, [inp, shape_const_name], [out])

        return [shape_op, ret]


class Softmax(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {"axis": (Op.INT, 1)}


class Softmax_13(PassThroughOp, op_type="Softmax", version=13):
    GROUP = GroupType.ONNX

    attr_dict = {"axis": (Op.INT, 1)}


class CaffeSoftmax(PassThroughOp):
    """A softmax op whose semantic obeys that in Caffe,
    that is, the ``axis`` attribute means softmax will be done ON THAT axis.
    """

    GROUP = GroupType.CASE

    attr_dict = {"axis": (Op.INT, 1)}


class TopkSoftmax(PassThroughOp):
    """A softmax op whose semantic obeys that in Caffe,
    that is, the ``axis`` attribute means softmax will be done ON THAT axis.
    """

    GROUP = GroupType.CASE

    attr_dict = {"axis": (Op.INT, 1), "topk": (Op.INT, 10)}


class Transpose(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"perm": (Op.LIST_OF_INTS,)}

    def infer_shape(self):
        attributes = self.attributes
        bottom_shape = self.owning_graph.get_tensor_shape(self.input[0])
        dtype = self.owning_graph.get_tensor_dtype(self.input[0])

        top_shape = bottom_shape[:]
        for i, j in enumerate(attributes["perm"].ints):
            top_shape[i] = bottom_shape[j]
        self.owning_graph.set_tensor_shape(self.output[0], top_shape, dtype)

    @staticmethod
    def make_transpose(name, inp, out, perm):
        """Create a transpose op.

        Returns:
            Op: the created transpose op.
        """
        return Op.make_op("Transpose", name, [inp], [out], {"perm": perm})


class Upsample(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        "mode": (Op.STRING, "nearest"),
        # the following two attribtues are for maintaining compatibility with nart.tools.
        "height": (Op.INT,),
        "width": (Op.INT),
    }

    def infer_shape(self):
        input_size = self.owning_graph.get_tensor_shape(self.input[0])
        if len(self.input) == 1:
            # compatible with nart.tools.pytorch
            h = self.get_attribute_value("height")
            w = self.get_attribute_value("width")
        else:
            if self.input[1] in self.owning_graph.initializer:
                scale = numpy_helper.to_array(
                    self.owning_graph.initializer[self.input[1]]
                )
            else:
                constant_node = self.owning_graph.get_tensor_producer(self.input[1])[-1]
                scale = constant_node.to_ndarray()
            scale = list(scale)
            h = math.floor(np.float32(np.float32(scale[2]) * input_size[2]))
            w = math.floor(np.float32(np.float32(scale[3]) * input_size[3]))
        self.owning_graph.set_tensor_shape(
            self.output[0], [input_size[0], input_size[1], h, w]
        )


class Constant(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"value": (Op.TENSOR,)}

    def infer_shape(self):
        shape = list(self.attributes["value"].t.dims)
        dtype = self.owning_graph.get_nart_dtype_from_onnx(
            self.attributes["value"].t.data_type
        )
        # FIXME:
        #   workaround for scalar constant
        if shape == []:
            shape = [1]
            self.attributes["value"].t.dims[:] = [1]
        self.owning_graph.set_tensor_shape(self.output[0], shape, dtype)

    @staticmethod
    def make_constant(name, value):
        """Make a Constant layer with value.
        Args:
            name (str): the op and output name.
            value (numpy.ndarray): the tensor value.
        Returns:
            the created Constant op.
        """
        assert isinstance(value, np.ndarray), "value is required to be a numpy.ndarray"
        op = Op.gen_op("Constant", name)
        value_tensor = numpy_helper.from_array(value)
        op.attributes["value"] = helper.make_attribute("value", value_tensor)
        op.add_output(name)
        return op

    def to_ndarray(self):
        tensor = self.attributes["value"].t
        res = numpy_helper.to_array(tensor)
        return res

    def __repr__(self):
        tensor = self.attributes["value"].t
        num_elements = reduce(mul, tensor.dims, 1)
        if num_elements <= 8:
            return f"Node{'::'+self.op_type if self.op_type != '' else ''}[name={self._name}, output={self.output}, value={self.to_ndarray()}]"
        else:
            return f"Node{'::'+self.op_type if self.op_type != '' else ''}[name={self._name}, output={self.output}, shape={tensor.dims}]"

    def check_similarity(self, other):
        if other.op_type != "Constant":
            return False
        tensor_a = self.attributes["value"].t
        tensor_b = other.attributes["value"].t
        if tensor_a.dims != tensor_b.dims:
            return False
        return np.allclose(
            numpy_helper.to_array(tensor_a), numpy_helper.to_array(tensor_b)
        )


class Gemm(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        "transA": (Op.INT, 0),
        "transB": (Op.INT, 0),
    }

    def infer_shape(self):
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        transA = self.get_attribute_value("transA", 0)
        if not transA:
            M, K1 = input_shape
        else:
            K1, M = input_shape
        # M = input_shape[0] if not transA else input_shape[1]

        weight_shape = self.owning_graph.get_tensor_shape(self.input[1])
        transB = self.get_attribute_value("transB", 0)
        if not transB:
            K2, N = weight_shape
        else:
            N, K2 = weight_shape
        # N = weight_shape[1] if not transB else weight_shape[0]
        assert K1 == K2, (
            "The reduction axis of Gemm's two operands should be equal, but got"
            "{0} vs {1}".format(K1, K2)
        )

        if not (len(input_shape) == 2 and len(weight_shape) == 2):
            logger.error(
                f"Gemm requires inputs to be 2D, but the inputs of `{self.name} is "
                f"{input_shape} and {weight_shape}"
            )

        self.owning_graph.set_tensor_shape(self.output[0], [M, N])


# class MatMul(Gemm):
#     pass


class Corr(Op):
    """Corss correlation operator.
    Inputs:
        feature: the input feature.
        kernel: the kernel input acts as conv kernel.
    """

    GROUP = GroupType.TOOLS

    attr_dict = {"groups": (Op.INT, 1)}

    def infer_shape(self):
        attributes = self.attributes
        kernel = self.owning_graph.get_tensor_shape(self.input[1])
        feat = self.owning_graph.get_tensor_shape(self.input[0])

        kernel_h, kernel_w = kernel[2:4]
        n, c, h, w = feat
        h = h - kernel_h + 1
        w = w - kernel_w + 1
        groups = attributes["groups"].i
        if groups == 1:
            self.owning_graph.set_tensor_shape(
                self.output[0], [n, kernel[1] // c, h, w]
            )
        else:
            self.owning_graph.set_tensor_shape(self.output[0], [n, c, h, w])


class Correlation1D(Op):

    GROUP = GroupType.TOOLS

    attr_dict = {
        "max_displacement": (Op.INT,),
        "pad": (Op.INT, 0),
        "single_direction": (Op.INT, 0),
        "kernel_size": (Op.INT,),
    }

    def infer_shape(self):
        attributes = self.attributes
        max_displacement = attributes["max_displacement"].i

        left = self.owning_graph.get_tensor_shape(self.input[0])
        right = self.owning_graph.get_tensor_shape(self.input[1])

        assert left == right
        n, c, h, w = left

        self.owning_graph.set_tensor_shape(
            self.output[0], [n, max_displacement + 1, h, w]
        )


class RoiPool(Op):

    Group = GroupType.TOOLS

    attr_dict = {
        "pooled_height": (Op.INT, 0),
        "pooled_width": (Op.INT, 0),
        "spatial_scale": (Op.FLOAT, 1.0),
    }

    def infer_shape(self):
        attributes = self.attributes
        feature = self.owning_graph.get_tensor_shape(self.input[0])
        roi = self.owning_graph.get_tensor_shape(self.input[1])
        shape = [
            roi[0],
            feature[1],
            attributes["pooled_height"].i,
            attributes["pooled_width"].i,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], shape)


class MaxRoiPool(Op):
    Group = GroupType.TOOLS

    attr_dict = {"pooled_shape": (Op.LIST_OF_INTS,), "spatial_scale": (Op.FLOAT, 1.0)}

    def infer_shape(self):
        feature = self.owning_graph.get_tensor_shape(self.input[0])
        roi = self.owning_graph.get_tensor_shape(self.input[1])
        pooled_shape = self.get_attribute_value("pooled_shape")
        assert (
            len(pooled_shape) == 2
        ), "pooled_shape should be two integer (height & width)"
        shape = [roi[0], feature[1], pooled_shape[0], pooled_shape[1]]
        self.owning_graph.set_tensor_shape(self.output[0], shape)


class RoiAlign(Op):
    """RoiAlign layer.

    Inputs:
        feature: the input feature map, 4D.
        rois: ROIs, [N, 5]. Each ROI is [batch_index, x1, y1, x2, y2]
    """

    Group = GroupType.TOOLS

    attr_dict = {
        "pooled_height": (Op.INT, 0),
        "pooled_width": (Op.INT, 0),
        "spatial_scale": (Op.FLOAT, 1.0),
        "sample_num": (Op.INT, 1),
    }

    def infer_shape(self):
        attributes = self.attributes
        feature = self.owning_graph.get_tensor_shape(self.input[0])
        roi = self.owning_graph.get_tensor_shape(self.input[1])
        shape = [
            roi[0],
            feature[1],
            attributes["pooled_height"].i,
            attributes["pooled_width"].i,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], shape)


class PODRoiAlign(Op):

    Group = GroupType.TOOLS

    attr_dict = {
        "pooled_height": (Op.INT, 0),
        "pooled_width": (Op.INT, 0),
        "spatial_scale": (Op.FLOAT, 1.0),
        "sample_num": (Op.INT, 1),
    }

    def infer_shape(self):
        attributes = self.attributes
        feature = self.owning_graph.get_tensor_shape(self.input[0])
        roi = self.owning_graph.get_tensor_shape(self.input[1])
        shape = [
            roi[0],
            feature[1],
            attributes["pooled_height"].i,
            attributes["pooled_width"].i,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], shape)


class PSRoiPool(Op):

    GROUP = GroupType.TOOLS

    attr_dict = {
        "spatial_scale": (Op.FLOAT, 1.0),
        "output_dim": (Op.INT,),
        "group_size": (Op.INT,),
    }

    def infer_shape(self):
        attributes = self.attributes
        roi = self.owning_graph.get_tensor_shape(self.input[1])
        shape = [
            roi[0],
            attributes["output_dim"].i,
            attributes["group_size"].i,
            attributes["group_size"].i,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], shape)


class PSRoiMaskPool(Op):

    GROUP = GroupType.TOOLS

    attr_dict = {
        "spatial_scale": (Op.FLOAT,),
        "output_dim": (Op.INT,),
        "group_size": (Op.INT,),
        "roi_scale": (Op.FLOAT, 1.0),
        "bin_scale": (Op.FLOAT, 1.0),
    }

    def infer_shape(self):
        attributes = self.attributes
        roi = self.owning_graph.get_tensor_shape(self.input[1])
        shape = [
            roi[0],
            attributes["output_dim"].i,
            attributes["group_size"].i,
            attributes["group_size"].i,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], shape)


class ShuffleChannel(PassThroughOp):

    GROUP = GroupType.TOOLS

    attr_dict = {"group": (Op.INT, 1)}


class HeatMap2Coord(Op):

    GROUP = GroupType.CASE

    attr_dict = {
        "coord_h": (Op.INT, 0),
        "coord_w": (Op.INT, 0),
        "coord_reposition": (Op.BOOL, False),
    }

    def infer_shape(self):
        shape = self.owning_graph.get_tensor_shape(self.input[0])
        self.owning_graph.set_tensor_shape(self.output[0], [shape[0], shape[1] * 3])


class Clip(PassThroughOp):

    GROUP = GroupType.TOOLS

    attr_dict = {"max": (Op.FLOAT, 3.402823e38), "min": (Op.FLOAT, -3.402823e38)}

    @staticmethod
    def make_clip(name, input, output, min_val, max_val):
        return Op.make_op(
            "Clip", name, [input], [output], {"max": max_val, "min": min_val}
        )


class Slice_1(Op, op_type="Slice", version=1):
    GROUP = GroupType.ONNX

    attr_dict = {
        "axes": (Op.LIST_OF_INTS,),
        "starts": (Op.LIST_OF_INTS,),
        "ends": (Op.LIST_OF_INTS,),
    }

    def infer_shape(self):
        def _clamp_index(v, upbound):
            if v < 0:
                return v + upbound
            if v > upbound:
                return upbound
            return v

        starts = self.get_attribute_value("starts")
        ends = self.get_attribute_value("ends")
        graph = self.owning_graph
        data_shape = graph.get_tensor_shape(self.input[0])
        out_shape = data_shape.copy()

        axes = self.get_attribute_value("axes")
        for idx, axis in enumerate(axes):
            # handle negtive index and outbound index
            upbound = data_shape[axis]
            out_shape[axis] = _clamp_index(ends[idx], upbound) - _clamp_index(
                starts[idx], upbound
            )
            out_shape[axis] = out_shape[axis]
        dtype = graph.get_tensor_dtype(self.input[0])
        graph.set_tensor_shape(self.output[0], out_shape, dtype)


class Slice_10(Op, op_type="Slice", version=10):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    def infer_shape(self):
        attributes = self.attributes
        graph = self.owning_graph
        # the shape of input tensor to be sliced from.
        data_shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        # the slice parameters: starts, ends, axis

        def _clamp_index(v, upbound):
            if v < 0:
                return v + upbound
            if v > upbound:
                return upbound
            return v

        # Slice-10 implement
        starts = graph.get_const_tensor_as_array(self.input[1])
        assert starts is not None, "cannot infer shape for dynamic slice layer"
        starts = list(starts)

        ends = graph.get_const_tensor_as_array(self.input[2])
        assert ends is not None, "cannot infer shape for dynamic slice layer"
        ends = list(ends)

        if self.has_input(3):
            axis = graph.get_const_tensor_as_array(self.input[3])
            assert axis is not None, "cannot get the axes of slice layer"
            axis = list(axis)
        else:
            # if axes not given, means all axes.
            axis = range(len(data_shape))
        if self.has_input(4):
            step = graph.get_const_tensor_as_array(self.input[4])
            step = list(step)
            if any(x < 0 for x in step):
                raise NotImplementedError(
                    "infer_shape not implemented for negatively strided slice"
                )
        else:
            step = [1] * len(data_shape)

        out_shape = data_shape.copy()
        for idx, (axis, step) in enumerate(zip(axis, step)):
            # handle negtive index and outbound index
            upbound = data_shape[axis]
            out_shape[axis] = _clamp_index(ends[idx], upbound) - _clamp_index(
                starts[idx], upbound
            )
            out_shape[axis] = (out_shape[axis] + step - 1) // step
        graph.set_tensor_shape(self.output[0], out_shape, dtype)

    def to_ndarray(self):
        def _get_const_array(node, _input):
            if node == "input":
                ret = numpy_helper.to_array(self.owning_graph.initializer[_input])
            else:
                ret = node.to_ndarray()
            return ret

        try:
            graph = self.owning_graph
            axes_node = graph.get_tensor_producer(self.input[3])[-1]
            axes = _get_const_array(axes_node, self.input[3])
            assert len(axes) == 1 and axes[0] == 0, f"{len(axes)}, {axes[0]}"
            constant_node = graph.get_tensor_producer(self.input[0])[-1]
            constant = _get_const_array(constant_node, self.input[0])
            starts_node = graph.get_tensor_producer(self.input[1])[-1]
            starts = _get_const_array(starts_node, self.input[2])
            assert len(starts) == 1
            ends_node = graph.get_tensor_producer(self.input[2])[-1]
            ends = _get_const_array(ends_node, self.input[3])
            assert len(ends) == 1
            return constant[starts[0] : ends[0]]
        except:
            raise NotImplementedError(f"to_ndarray not implemented for {self.name}.")


class Dropout(PassThroughOp):
    """not intended to be used, after graph construction, use EliminateNodes pass to eliminate Dropout nodes."""

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Cast(Op):
    GROUP = GroupType.ONNX

    attr_dict = {"to": (Op.INT,)}

    @staticmethod
    def make_cast(name, input, output, to):
        """Make a cast node.

        Args:
            to (DataType): the target data type.

        Returns:
            Cast: the new Cast node.
        """
        node = Op.gen_op("Cast", name)
        node.add_input(input)
        node.add_output(output)
        node.attributes["to"] = helper.make_attribute("to", int(to))
        return node

    def to_ndarray(self):
        to = self.get_attribute_value("to")
        input_node = self.owning_graph.get_tensor_producer(self.input[0])[-1]
        if hasattr(input_node, "to_ndarray"):
            input_value = input_node.to_ndarray()
        if to == DataType.FLOAT:
            input_value = input_value.astype(np.float32)
        elif to == DataType.UINT8:
            input_value = input_value.astype(np.uint8)
        elif to == DataType.INT8:
            input_value = input_value.astype(np.int8)
        elif to == DataType.UINT16:
            input_value = input_value.astype(np.uint16)
        elif to == DataType.INT16:
            input_value = input_value.astype(np.int16)
        elif to == DataType.INT32:
            input_value = input_value.astype(np.int32)
        elif to == DataType.INT64:
            input_value = input_value.astype(np.int64)
        elif to == DataType.STRING:
            input_value = input_value.astype(np.string_)
        elif to == DataType.BOOL:
            input_value = input_value.astype(np.bool)
        elif to == DataType.FLOAT16:
            input_value = input_value.astype(np.float16)
        elif to == DataType.DOUBLE:
            input_value = input_value.astype(np.double)
        elif to == DataType.UINT32:
            input_value = input_value.astype(np.uint32)
        elif to == DataType.UINT64:
            input_value = input_value.astype(np.uint64)
        elif to == DataType.COMPLEX64:
            input_value = input_value.astype(np.complex64)
        elif to == DataType.COMPLEX128:
            input_value = input_value.astype(np.complex128)
        else:
            raise NotImplementedError(
                f"to_ndarray not implemented for DataType UNDEFINED and BFLOAT16."
            )
        return input_value

    def infer_shape(self):
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        to = self.get_attribute_value("to", Dtype.Float32)
        dtype = graph.get_nart_dtype_from_onnx(int(to))
        for output in self.output:
            graph.set_tensor_shape(output, shape, dtype)


class GlobalAveragePool(Op):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        # The first two dimensions of output shape are the same as the input (N x C),
        # while the other dimensions are all 1.
        out_shape = input_shape[:2]
        out_shape.extend([1] * (len(input_shape) - 2))

        graph.set_tensor_shape(self.output[0], out_shape)

    @staticmethod
    def make_global_average_pool(name, input, output):
        node = Op.gen_op("GlobalAveragePool", name)
        node.add_input(input)
        node.add_output(output)
        return node


class GlobalMaxPool(Op):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        # The first two dimensions of output shape are the same as the input (N x C),
        # while the other dimensions are all 1.
        out_shape = input_shape[:2]
        out_shape.extend([1] * (len(input_shape) - 2))

        graph.set_tensor_shape(self.output[0], out_shape)

    @staticmethod
    def make_global_max_pool(name, input, output):
        node = Op.gen_op("GlobalMaxPool", name)
        node.add_input(input)
        node.add_output(output)
        return node


class GridSample(Op):
    GROUP = GroupType.CASE

    attr_dict = {
        "mode": (Op.STRING, "bilinear"),
        "padding_mode": (Op.STRING, "zeros"),
        "align_corners": (Op.BOOL, False),
    }

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        grid_shape = graph.get_tensor_shape(self.input[1])

        out_shape = list(input_shape)
        out_shape[2:] = grid_shape[1:-1]
        graph.set_tensor_shape(self.output[0], out_shape)


class Unsqueeze(Op):
    GROUP = GroupType.ONNX

    attr_dict = {"axes": (Op.LIST_OF_INTS,)}

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        axes = self.get_attribute_value("axes")
        out_dim = len(input_shape) + len(axes)
        axes = [x % out_dim for x in axes]
        it = iter(input_shape)
        out_shape = []
        for axis in range(out_dim):
            if axis in axes:
                # fill a 1
                out_shape.append(1)
            else:
                # copy from original shape
                out_shape.append(next(it))
        graph.set_tensor_shape(self.output[0], out_shape, dtype)

    @staticmethod
    def make_unsqueeze(name, input, output, axes):
        return Op.make_op(
            "Unsqueeze", name, [input], [output], {"axes": [int(x) for x in axes]}
        )

    def to_ndarray(self):
        try:
            axes = helper.get_attribute_value(self.attributes["axes"])
            assert len(axes) == 1 and axes[0] == 0, f"{len(axes)}， {axes[0]}"
            input_node = self.owning_graph.get_tensor_producer(self.input[0])[-1]
            input_value = input_node.to_ndarray()
            if np.isscalar(input_value):
                return np.array([input_value], dtype=input_value.dtype)
            else:
                return input_value[np.newaxis]
        except:
            raise NotImplementedError(f"to_ndarray not implemented for {self.name}.")


class Unfold(Op):

    GROUP = GroupType.CASE

    attr_dict = {
        "dilation": (Op.LIST_OF_INTS,),
        "kernel_size": (Op.LIST_OF_INTS,),
        "padding": (Op.LIST_OF_INTS, [0, 0]),
        "stride": (Op.LIST_OF_INTS,),
    }

    def infer_shape(self):
        def _get_list(name, val):
            if isinstance(val, (list, tuple)):
                if len(val) == 2:
                    return val
                elif len(val) == 1:
                    return val * 2
                else:
                    logger.warning("Can only support 2-D {self.__class__.__name__}")
                    return val[:2]
            return [val] * 2

        attributes = self.attributes
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        assert len(input_shape) == 4, "Can only support 2-D {self.__class__.__name__}"
        n, c, h, w = input_shape

        pad_h, pad_w = _get_list("padding", self.get_attribute_value("padding"))
        kernel_h, kernel_w = _get_list(
            "kernel_size", self.get_attribute_value("kernel_size")
        )
        stride_h, stride_w = _get_list(
            "stride", self.get_attribute_value("strides", [1, 1])
        )
        dilation_h, dilation_w = _get_list(
            "dilation", self.get_attribute_value("dilation", [1, 1])
        )

        kernel_h_eff = kernel_h + (kernel_h - 1) * (dilation_h - 1)
        kernel_w_eff = kernel_w + (kernel_w - 1) * (dilation_w - 1)
        h = math.floor((h + 2 * pad_h - kernel_h_eff) / stride_h + 1)
        w = math.floor((w + 2 * pad_w - kernel_w_eff) / stride_w + 1)

        self.owning_graph.set_tensor_shape(
            self.output[0], [n, c * kernel_h * kernel_w, h * w]
        )


class Sum(BroadcastOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Eltwise(Op):
    GROUP = GroupType.CASE

    attr_dict = {"coeff": (Op.LIST_OF_FLOATS,)}

    def infer_shape(self):
        graph = self.owning_graph
        input1_shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        for input in self.input[1:]:
            shape = graph.get_tensor_shape(input)
            assert shape == input1_shape, (
                f"Eltwise requires input operands have same shape, but input shapes of `{self.name}` are "
                f"{self.input[0]}:{input1_shape} and {input}:{shape}"
            )
        graph.set_tensor_shape(self.output[0], input1_shape, dtype)

    def __repr__(self):
        coeff = self.attributes["coeff"].floats
        return (
            f"Node{'::'+self.op_type if self.op_type != '' else ''}[name={self._name}, input={self.input}, "
            + f"coeff={coeff} output={self.output}]"
        )


###### Family of Unary ops ######


class Round(PassThroughOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Floor(PassThroughOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Ceil(PassThroughOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Abs(PassThroughOp):
    GROUP = GroupType.ONNX
    attr_dict = {
        # emtpy
    }


class FloorDiv(BroadcastOp):
    """A dummy op which represents Floor(Div(a, b)). This is used with tensorrt, which supports the Floor_Div op."""

    GROUP = GroupType.CASE

    attr_dict = {
        # empty
    }


class OplibOp(Op):
    def infer_shape(self):
        shape_data = list(self.attributes["output_shapes"].ints)
        for each_output in self.output:
            count = shape_data.pop(0)
            shape_slice = shape_data[:count]
            del shape_data[:count]
            if 0 == len(self.owning_graph.get_tensor_shape(each_output)):
                self.owning_graph.set_tensor_shape(each_output, shape_slice)


class Flatten(Op):
    GROUP = GroupType.ONNX

    attr_dict = {"axis": (Op.INT, 1)}

    def infer_shape(self):
        from functools import reduce

        graph = self.owning_graph
        import operator

        def reduce_mul(x):
            return reduce(operator.mul, x, 1)

        axis = self.get_attribute_value("axis", 1)
        input_shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        output_shape = [reduce_mul(input_shape[0:axis]), reduce_mul(input_shape[axis:])]
        graph.set_tensor_shape(self.output[0], output_shape, dtype)


class LeakyRelu(PassThroughOp):
    GROUP = GroupType.ONNX

    attr_dict = {"alpha": (Op.FLOAT, 0.01)}


class LpNormalization(PassThroughOp):

    GROUP = GroupType.TOOLS

    attr_dict = {"p": (Op.INT, 2), "axis": (Op.INT, -1)}


class Hswish(PassThroughOp):
    GROUP = GroupType.CASE

    attr_dict = {
        # empty
    }


class Hsigmoid(PassThroughOp):
    GROUP = GroupType.CASE

    attr_dict = {"alpha": (Op.FLOAT, 0.2), "beta": (Op.FLOAT, 0.5)}


class ReductionOp(Op, is_abstract=True):
    """A type of Op which does reduction operation."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    GROUP = GroupType.ONNX

    attr_dict = {"axes": (Op.LIST_OF_INTS,), "keepdims": (Op.INT, 1)}

    def infer_shape(self):
        """Reduction shape infer."""
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        ndim = len(shape)
        keepdims = self.get_attribute_value("keepdims", 1)
        # the default is to reduce over all the dimensions of input tensors.
        axes = set(self.get_attribute_value("axes", range(ndim)))
        assert len(axes) != 0, f"no reduce axis set in node: {self.name}"
        axes = {x if x >= 0 else x + ndim for x in axes}
        out_shape = []
        for axis, value in enumerate(shape):
            if axis in axes:
                # this is an axis to be reduced
                if keepdims:
                    out_shape.append(1)
            else:
                out_shape.append(value)
        # if output shape is empty, it actually should be [1].
        out_shape = out_shape if len(out_shape) != 0 else [1]
        self.owning_graph.set_tensor_shape(self.output[0], out_shape, dtype)


###### Family of reduction ops ######


class ReduceL1(ReductionOp):
    pass


class ReduceL2(ReductionOp):
    pass


class ReduceLogSum(ReductionOp):
    pass


class ReduceLogSumExp(ReductionOp):
    pass


class ReduceMax(ReductionOp):
    pass


class ReduceMean(ReductionOp):
    pass


class ReduceMin(ReductionOp):
    pass


class ReduceProd(ReductionOp):
    pass


class ReduceSum(ReductionOp):
    pass


class ReduceSumSquare(ReductionOp):
    pass


class GroupNorm(PassThroughOp):
    GROUP = GroupType.CASE

    attr_dict = {"num_groups": (Op.INT, 1), "eps": (Op.FLOAT, 1e-5)}


class CaffeBatchNorm(Op):
    """Mimics caffe BatchNrom layer.
    The parameter `use_global_stats` is not present, just take it as True.

    Inputs: data, mean and var.
    """

    GROUP = GroupType.CASE

    attr_dict = {"eps": (Op.FLOAT, 1e-5)}

    def infer_shape(self):
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        graph.set_tensor_shape(self.output[0], shape, dtype)


class CaffeScale(Op):
    """Mimic caffe Scale layer.

    Inputs: data, scale, [bias]
    """

    GROUP = GroupType.CASE

    attr_dict = {"axis": (Op.INT, 1)}

    def infer_shape(self):
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        graph.set_tensor_shape(self.output[0], shape, dtype)


class Pad(Op):
    GROUP = GroupType.ONNX

    attr_dict = {
        "mode": (Op.STRING, "CONSTANT"),
        "value": (Op.FLOAT, 0),
        "pads": (Op.LIST_OF_INTS,),
    }

    def infer_shape(self):
        attributes = self.attributes
        graph = self.owning_graph
        input_size = self.owning_graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        out_size = input_size
        j = 0
        if "pads" in attributes:
            for j, i in enumerate(attributes["pads"].ints):
                out_size[j % len(out_size)] = out_size[j % len(out_size)] + i
        else:
            graph = self.owning_graph
            paddings = graph.get_const_tensor_as_array(self.input[1])
            for j, i in enumerate(paddings):
                out_size[j % len(out_size)] = out_size[j % len(out_size)] + i
            # print(1)
        self.owning_graph.set_tensor_shape(self.output[0], out_size, dtype)


class DynamicUpsample(Upsample):
    """Mimics Caffe Interp layer, resize input[0] like the input[1].
    **NOTE**: only the height and width will be resized.
    """

    GROUP = GroupType.ONNX

    attr_dict = {"mode": (Op.STRING, "nearest")}

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        shape = graph.get_tensor_shape(self.input[1])
        assert (
            len(input_shape) == 4 and len(shape) == 4
        ), f"DynamicUpsample requires both inputs be 4D, but {len(input_shape)}D and {len(shape)}D got"
        output_shape = input_shape.copy()
        height, width = shape[2], shape[3]
        output_shape[2:4] = height, width
        graph.set_tensor_shape(self.output[0], output_shape)


class CaffePower(PassThroughOp):
    """Mimics caffe Power layer. Caffe Power layer computes outputs y = (shift + scale * x) ^ power.

    Inputs: x
    """

    GROUP = GroupType.CASE

    attr_dict = {
        "power": (Op.FLOAT, 1.0),
        "scale": (Op.FLOAT, 1.0),
        "shift": (Op.FLOAT, 0.0),
    }


class Pow(BroadcastOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # no attribute
    }

    @staticmethod
    def make_pow(name, x, y, output):
        return Op.make_op("Pow", name, [x, y], [output])


class Split(Op):
    """ONNX Split-2"""

    GROUP = GroupType.ONNX

    attr_dict = {"axis": (Op.INT, 0), "split": (Op.LIST_OF_INTS,)}

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        splits = self.get_attribute_value("split")
        assert len(splits) == len(
            self.output
        ), "only explicitly specified Split op is supported"
        axis = self.get_attribute_value("axis")
        assert input_shape[axis] == reduce(
            add, splits, 0
        ), f"invalid Split op, spliting the {axis}-th axis of shape {input_shape} into parts with sizes = {splits}"
        for out, size in zip(self.output, splits):
            output_shape = input_shape.copy()
            output_shape[axis] = size
            graph.set_tensor_shape(out, output_shape, dtype)


class SubPixel(Op):
    """Implementation of the SubPixel function (aka PixelShuffle in pytorch), which rearrange a tensor
    from (N, C*r^2, H, W) to (N, C, H * r, W * r).
    Furthermore, we support the inverse operation subpixel down.
    """

    GROUP = GroupType.CASE

    attr_dict = {
        # 0 represents subpixel down, and 1 represents subpixel up
        "method": (Op.INT, 1),
        "sample": (Op.INT, 1),
    }

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        out_shape = list(input_shape)
        sample = self.get_attribute_value("sample")
        assert (
            len(input_shape) == 4
        ), f"SubPixel only support 4D tensor, but get {input_shape}"
        if self.get_attribute_value("method") == 1:
            # subpixel up
            out_shape[1] = input_shape[1] // sample**2
            out_shape[2] *= sample
            out_shape[3] *= sample
        elif self.get_attribute_value("method") == 0:
            # subpixel down
            out_shape[1] *= sample**2
            out_shape[2] //= sample
            out_shape[3] //= sample
        else:
            raise Exception(
                f"Unsupported subpixel method {self.get_attribute_value('method')}, should be 0 (subpixel down) or 1 (subpixel up)"
            )

        from functools import reduce

        if reduce(lambda a, b: a * b, input_shape) == reduce(
            lambda a, b: a * b, out_shape
        ):
            graph.set_tensor_shape(self.output[0], out_shape)
        else:
            raise Exception(
                f"[SubPixel] Check output's shape failed. IN {input_shape} vs OUT {out_shape}"
            )


class Gather(Op):
    """ONNX Gather-1
    The operation is same as numpy.take
    """

    GROUP = GroupType.ONNX

    attr_dict = {"axis": (Op.INT, 0)}

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        indice_shape = graph.get_tensor_shape(self.input[1])
        axis = self.get_attribute_value("axis")
        out_shape = input_shape[0:axis] + indice_shape + input_shape[axis + 1 :]
        graph.set_tensor_shape(self.output[0], out_shape, dtype)

    def to_ndarray(self):
        try:
            graph = self.owning_graph
            data = graph.get_tensor_producer(self.input[0])[-1]
            constant_node = graph.get_tensor_producer(self.input[1])[-1]
            if constant_node == "input":
                constant = numpy_helper.to_array(graph.initializer[self.input[1]])
            else:
                constant = constant_node.to_ndarray()
            data = data.to_ndarray()
            assert len(constant.shape) <= 1
            return np.array(data[constant], dtype=np.int32)
        except:
            raise NotImplementedError(f"to_ndarray not implemented for {self.name}.")


class GeLU(PassThroughOp):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415

    Which computes outputs = 0.5 * x * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * x ^ 3)))
    """

    GROUP = GroupType.CASE

    attr_dict = {
        # empty
    }


class LayerNorm(PassThroughOp):
    """Implementation of `torch.nn.LayerNorm`"""

    GROUP = GroupType.CASE

    attr_dict = {
        # empty
    }


class MeanVarianceNormalization(PassThroughOp):
    """ONNX MeanVarianceNormalization-9.
    Performs mean variance normalization on the input tensor X using formula: (X-EX)/sqrt(Var(X))

    Inputs:
        X: input tensor

    Outputs:
        Y: tensor after normalization, which has same shape as X.
    """

    GROUP = GroupType.ONNX

    attr_dict = {"axes": (Op.LIST_OF_INTS, [0, 2, 3])}


class TopK(Op):
    """ONNX TopK-1.
    Retrieve the top-K elements along a specified axis.

    Inputs:
        X: input tensor
    Outputs:
        Values: top K values from the input tensor
        Indices: the corresponding input tensor indices for the top K values.
    """

    GROUP = GroupType.ONNX

    attr_dict = {
        "axis": (Op.INT, -1),
        "k": (Op.INT,),
    }

    def infer_shape(self):
        graph = self.owning_graph
        in_shape = graph.get_tensor_shape(self.input[0])
        axis = self.get_attribute_value("axis")
        k = self.get_attribute_value("k")
        import copy

        out_shape = copy.copy(in_shape)
        out_shape[axis] = k
        graph.set_tensor_shape(self.output[0], out_shape)
        graph.set_tensor_shape(self.output[1], out_shape)


class CaffeNormalize(Op):
    """Caffe's normalize op: x_i = x_i / (sqrt(sigma(x_j^2)) + eps) * scale_i,
        where j iterate through channels if across_spatial = false,
        or j iterates through channels, height and with if across_spatial = true

    Inputs:
        x: the input tensor
        scales: the scale factors. it should be a tensor with shape [1] when channel_shared=True,
            a tensor with shape [C] otherwise.
    """

    GROUP = GroupType.TOOLS

    attr_dict = {
        "eps": (Op.FLOAT, 1e-10),
        "across_spatial": (Op.INT, 1),
        "channel_shared": (Op.INT, 1),
    }

    def infer_shape(self):
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        for output in self.output:
            graph.set_tensor_shape(output, shape, dtype)


class CaffeThreshold(PassThroughOp):
    """Mimics Caffe's Threshold layer.
    Given input X, calculate Y = X > threshold ? 1.0 : 0.0

    Inputs:
        X: the input tensor

    Outputs:
        Y: Y = X > threshold ? 1.0 : 0.0
    """

    GROUP = GroupType.TOOLS

    attr_dict = {"threshold": (Op.FLOAT,)}


class LSTM(Op):
    """ONNX LSTM-7 op."""

    GROUP = GroupType.ONNX

    attr_dict = {
        "hidden_size": (Op.INT,),
        "direction": (Op.STRING, "forward"),
        "input_forget": (Op.INT, 0),
        "clip": (Op.FLOAT, None),
        "activations": (Op.LIST_OF_STRINGS, ["Sigmoid", "Tanh", "Tanh"]),
        "activation_alpha": (Op.LIST_OF_FLOATS, None),
        "activation_beta": (Op.LIST_OF_FLOATS, None),
    }

    # Define the enum of activation functions
    activation_mode = {"Relu": 1, "Tanh": 2, "Sigmoid": 3}

    # Define the default alpha and beta perameter of activation fuctions
    default_activation_alpha_beta = {"Sigmoid": (0, 0), "Relu": (0, 0), "Tanh": (0, 0)}

    # Define the enum of direction
    direction_mode = {"forward": 1, "reverse": 2, "bidirectional": 3}

    def infer_shape(self):
        # TODO: finish infer_shape
        hidden_size = self.get_attribute_value("hidden_size")

        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        assert len(shape) == 3, "input X should be 3-D"
        seq_length = shape[0]
        batch_size = shape[1]
        # input_size = shape[2]

        direction = self.get_attribute_value("direction")
        assert direction in self.direction_mode.keys()
        if direction == "bidirectional":
            num_directions = 2
        else:
            num_directions = 1

        output_num = len(self.output)
        assert output_num <= 3, "lstm supports at most 3 outputs"

        if output_num >= 1:
            output_shape_0 = [seq_length, num_directions, batch_size, hidden_size]
            graph.set_tensor_shape(self.output[0], output_shape_0)

        if output_num >= 2:
            output_shape_1 = [num_directions, batch_size, hidden_size]
            graph.set_tensor_shape(self.output[1], output_shape_1)

        if output_num >= 3:
            output_shape_2 = [num_directions, batch_size, hidden_size]
            graph.set_tensor_shape(self.output[2], output_shape_2)


class QuantizeLinear(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {}


class DequantizeLinear(PassThroughOp):

    GROUP = GroupType.ONNX

    attr_dict = {}


class CaffeExp(PassThroughOp):
    """Mimics caffe's exp layer, calculates y = base ^ (shift + scale * x), where base, shift and scale are
    scalars.
    """

    GROUP = GroupType.ONNX

    attr_dict = {
        "base": (Op.FLOAT, -1.0),
        "scale": (Op.FLOAT, 1.0),
        "shift": (Op.FLOAT, 0.0),
    }


class ConstantOfShape(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"value": (Op.TENSOR,)}

    def infer_shape(self):
        producers = self.owning_graph.get_tensor_producer(self.input[0])
        if len(producers) != 1:
            raise RuntimeError(
                f"producer cannot be determined for tensor`{self.input[0]}`, producers={producers}"
            )
        if producers[0] == "input":
            shape = numpy_helper.to_array(self.owning_graph.initializer[self.input[0]])
        else:
            constant_node = producers[-1]
            shape = constant_node.to_ndarray().tolist()
            shape = [int(_) for _ in shape]
        shape = list(shape)
        self.owning_graph.set_tensor_shape(self.output[0], shape)

    def to_ndarray(self):
        tensor = self.attributes["value"].t
        res = numpy_helper.to_array(tensor)
        # res can be a scalar or an ndarray
        if len(res.shape) == 1:
            res = res[0]
        producers = self.owning_graph.get_tensor_producer(self.input[0])
        if len(producers) != 1:
            raise RuntimeError(
                f"producer cannot be determined for tensor`{self.input[0]}`, producers={producers}"
            )
        if producers[0] == "input":
            shape = numpy_helper.to_array(self.owning_graph.initializer[self.input[0]])
        else:
            constant_node = producers[-1]
            shape = constant_node.to_ndarray()
        shape = list(shape)
        ret = np.ones(shape, dtype=res.dtype) * res
        return ret


class Shape(Op):
    GROUP = GroupType.ONNX

    attr_dict = {"start": (Op.INT, 0), "end": (Op.INT,)}

    def infer_shape(self):
        start = self.get_attribute_value("start")
        if "end" not in self.attributes:
            end = None
        else:
            end = self.get_attribute_value("end")
        input_shape_len = len(self.owning_graph.get_tensor_shape(self.input[0]))
        if start < 0:
            start = input_shape_len + start
        if not end:
            shape = [input_shape_len - start]
        else:
            if end < 0:
                end = input_shape_len + end
            shape = [end - start]
        self.owning_graph.set_tensor_shape(self.output[0], shape)

    def to_ndarray(self):
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        input_shape_len = len(input_shape)
        start = self.get_attribute_value("start")
        if "end" not in self.attributes:
            end = None
        else:
            end = self.get_attribute_value("end")
        if not end:
            shape = input_shape[start:]
        else:
            if end < 0:
                end = input_shape_len + end
            shape = input_shape[start:end]
        return np.array(shape, dtype=np.int32)


class Less(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class And(BroadcastOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Reciprocal(BroadcastOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Not(BroadcastOp):

    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Resize_11(Op, op_type="Resize", version=11):

    GROUP = GroupType.ONNX

    attr_dict = {
        "coordinate_transformation_mode": (Op.STRING, "half_pixel"),
        "cubic_coeff_a": (Op.FLOAT, -0.75),
        "exclude_outside": (Op.INT, 0),
        "extrapolation_value": (Op.FLOAT, 0.0),
        "mode": (Op.STRING, "nearest"),
        "nearest_mode": (Op.STRING, "round_prefer_floor"),
    }

    def infer_shape(self):
        # Only support nearest upsample with scales specified
        input_size = self.owning_graph.get_tensor_shape(self.input[0])
        graph = self.owning_graph
        if self.has_input(3):
            size = graph.get_const_tensor_as_array(self.input[3])
            assert (
                size is not None
            ), "Dynamic output size of Resize is not supported at present."
            h = size[2]
            w = size[3]
        else:
            scale = graph.get_const_tensor_as_array(self.input[2], False)
            scale = list(scale)
            h = math.floor(np.float32(scale[2] * input_size[2]))
            w = math.floor(np.float32(scale[3] * input_size[3]))
        self.owning_graph.set_tensor_shape(
            self.output[0], [input_size[0], input_size[1], h, w]
        )


class Tile(Op):

    GROUP = GroupType.ONNX

    attr_dict = {}

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        repeats_op = graph.get_tensor_producer(self.input[1])[0]
        assert (
            isinstance(repeats_op, Constant) or repeats_op == "input"
        ), "cannot infer shape for dynamic repeat layer"
        repeats = (
            repeats_op.to_ndarray()
            if isinstance(repeats_op, Constant)
            else numpy_helper.to_array(graph.initializer[self.input[1]])
        )
        repeats = list(repeats)
        output_shape = input_shape
        for i in range(len(input_shape)):
            output_shape[i] *= repeats[i]
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class Expand(Op):

    GROUP = GroupType.ONNX

    attr_dict = {}

    def infer_shape(self):
        graph = self.owning_graph
        input_shape = graph.get_tensor_shape(self.input[0])
        producers = self.owning_graph.get_tensor_producer(self.input[1])
        if len(producers) != 1:
            raise RuntimeError(
                f"producer cannot be determined for tensor`{self.input[1]}`, producers={producers}"
            )
        shape = graph.get_const_tensor_as_array(self.input[1])
        input_dims = len(input_shape)
        shape_dims = len(shape)
        output_dims = max(input_dims, shape_dims)
        output_shape = [1] * (output_dims - input_dims) + input_shape
        broadcast_shape = [1] * (output_dims - shape_dims) + list(shape)
        for i in range(output_dims):
            output_shape[i] = max(output_shape[i], broadcast_shape[i])
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class DepthToSpace(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"blocksize": (Op.INT,), "mode": (Op.STRING, "DCR")}

    def infer_shape(self):
        blocksize = self.get_attribute_value("blocksize")
        mode = self.get_attribute_value("mode")
        assert mode == "CRD"
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        c, h, w = input_shape[-3:]
        assert len(input_shape) >= 3
        assert c % (blocksize**2) == 0, (
            "c must be divisable by blocksize_squared, but c: %d, blocksize_squared: %d."
            % (c, blocksize**2)
        )
        output_shape = input_shape[:-3] + [
            c // blocksize**2,
            h * blocksize,
            w * blocksize,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class SpaceToDepth(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"blocksize": (Op.INT,)}

    def infer_shape(self):
        blocksize = self.get_attribute_value("blocksize")
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        c, h, w = input_shape[-3:]
        assert len(input_shape) >= 3
        assert h % blocksize == 0 and w % blocksize == 0, (
            "h and w must be divisable by blocksize, but h: %d, w: %d, blocksize: %d."
            % (h, w, blocksize)
        )
        output_shape = input_shape[:-3] + [
            c * blocksize**2,
            h // blocksize,
            w // blocksize,
        ]
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class ScatterND(Op):

    GROUP = GroupType.ONNX

    attr_dict = {"reduction": (Op.STRING, "")}

    def infer_shape(self):
        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        self.owning_graph.set_tensor_shape(self.output[0], input_shape)


class Parameter(Op):

    GROUP = GroupType.ONNX

    attr_dict = {
        "batch": (Op.INT,),
        "channel": (Op.INT,),
        "height": (Op.INT,),
        "width": (Op.INT,),
        "value": (Op.TENSOR,),
        "n": (Op.INT, -1),
        "m": (Op.INT, -1),
    }

    def infer_shape(self):
        batch = self.get_attribute_value("batch")
        channel = self.get_attribute_value("channel")
        height = self.get_attribute_value("height")
        width = self.get_attribute_value("width")
        m = self.get_attribute_value("m")
        n = self.get_attribute_value("n")
        if m != -1 and n != -1:
            output_shape = [m, n]
        else:
            output_shape = [batch, channel, height, width]
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class ConvexUpsample(Op):

    GROUP = GroupType.CASE

    attr_dict = {
        "scale": (Op.INT, 4),
    }

    def infer_shape(self):
        scale = self.get_attribute_value("scale")

        input_shape = self.owning_graph.get_tensor_shape(self.input[0])
        output_shape = input_shape[:]
        output_shape[-2:] *= scale
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class CostVolume(Op):

    GROUP = GroupType.CASE

    attr_dict = {
        "max_disp": (Op.INT, 4),
        "index_num": (Op.INT, 53),
        "bias_term": (Op.BOOL, True),
    }

    def infer_shape(self):
        max_disp = self.get_attribute_value("max_disp")
        index_num = self.get_attribute_value("index_num")
        bias_term = self.get_attribute_value("bias_term")

        feat0_shape = self.owning_graph.get_tensor_shape(self.input[0])
        feat1_shape = self.owning_graph.get_tensor_shape(self.input[1])
        output_shape = feat0_shape[:]
        output_shape[1] = (2 * max_disp + 1) ** 2
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class Warp(Op):

    GROUP = GroupType.CASE

    attr_dict = {}

    def infer_shape(self):
        feat_shape = self.owning_graph.get_tensor_shape(self.input[0])
        flow_shape = self.owning_graph.get_tensor_shape(self.input[1])
        output_shape = feat_shape[:-2] + flow_shape[-2:]
        self.owning_graph.set_tensor_shape(self.output[0], output_shape)


class Sign(PassThroughOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class RoundTo0(PassThroughOp):
    GROUP = GroupType.ONNX

    attr_dict = {
        # empty
    }


class Elu(PassThroughOp):
    """The ELU-6 op."""

    GROUP = GroupType.ONNX

    attr_dict = {
        "alpha": (Op.FLOAT, 1.0),
    }


class ClipCast(PassThroughOp):

    GROUP = GroupType.CASE

    attr_dict = {"to": (Op.INT,)}


class AddDivClipCast(Op):

    GROUP = GroupType.CASE

    attr_dict = {"to": (Op.INT,)}

    def infer_shape(self):
        """input of this node is net_input, addend, divisor, clip_min, clip_max"""
        assert len(self.input) == 5, logger.error(
            f"""input of AddDivClipCast Op shoud
            be 5, but got {len(self.input)}"""
        )
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])
        dtype = graph.get_tensor_dtype(self.input[0])
        addend_shape = graph.get_tensor_shape(self.input[1])
        divisor_shape = graph.get_tensor_shape(self.input[2])
        assert addend_shape == divisor_shape, logger.error(
            f"""addend_shape shoud equal divisor shape,
            but got {addend_shape} v.s. {divisor_shape}"""
        )
        shape = BroadcastOp.broadcast_shape(shape, addend_shape)
        for output in self.output:
            graph.set_tensor_shape(output, shape, dtype)


class ArgMax(Op, version=11):

    GROUP = GroupType.ONNX

    attr_dict = {
        "axis": (Op.INT, 0),
        "keepdims": (Op.INT, 1),
        "select_last_index": (Op.INT, 0),
    }

    def infer_shape(self):
        """ArgMax shape infer."""
        graph = self.owning_graph
        shape = graph.get_tensor_shape(self.input[0])

        keepdims = self.get_attribute_value("keepdims", 1)
        axis = self.get_attribute_value("axis")

        out_shape = list(shape)
        if keepdims == 1:
            out_shape[axis] = 1
        else:
            del out_shape[axis]

        # if output shape is empty, it actually should be [1].
        out_shape = out_shape if len(out_shape) != 0 else [1]
        self.owning_graph.set_tensor_shape(self.output[0], out_shape, Dtype.Int64)
