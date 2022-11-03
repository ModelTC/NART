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
from ...ops import Constant
import logging
from ...ops.op import DELIM, Constant
from abc import ABC, abstractmethod
from ...proto import caffe_pb2
from ...core import Node
from ...ops import Op
from onnx import TensorProto
from onnx import helper

import numpy as np
import warnings

CAFFE_TO_ONNX = {}


logger = logging.getLogger(__name__)

op_to_param = {
    "ArgMax": "argmax_param",
    "ConvolutionReLU": "convolution_param",
    "Convolution": "convolution_param",
    "BN": "bn_param",
    "BatchNorm": "batch_norm_param",
    "Deconvolution": "convolution_param",
    "Correlation": "correlation_param",
    "Correlation1D": "correlation_param",
    "Eltwise": "eltwise_param",
    "Pooling": "pooling_param",
    "InnerProductReLU": "inner_product_param",
    "InnerProduct": "inner_product_param",
    "PReLU": "prelu_param",
    "Concat": "concat_param",
    "Reshape": "reshape_param",
    "Softmax": "softmax_param",
    "Transpose": "transpose_param",
    "Interp": "interp_param",
    "NNInterp": "nninterp_param",
    "NNUpsample": "nn_upsample_param",
    "ROIPooling": "roi_pooling_param",
    "PSROIPooling": "psroi_pooling_param",
    "ROIAlignPooling": "roi_align_pooling_param",
    "PSROIMaskPooling": "psroi_mask_pooling_param",
    "PODROIAlignPooling": "podroi_align_pooling_param",
    "HeatMap2Coord": "heatmap_param",
    "ShuffleChannel": "shuffle_channel_param",
    "SubpixelUp": "subpixel_up_param",
    "SubpixelDown": "subpixel_down_param",
    "Pad": "pad_param",
    "Exp": "exp_param",
    "Parameter": "parameter_param",
    "Reshape3d": "reshape3d_param",
    "Convolution3d": "convolution3d_param",
    "Pool3d": "pooling3d_param",
    "BatchNorm3d": "batchnorm3d_param",
    "ReLU3d": "relu3d_param",
    "Resize": "resize_param",
}


class BaseNode(ABC):
    """Base class of caffe_to_onnx layer converter.

    Args:
        layer (caffe_pb2.LayerParameter): the caffe layer definition.
        input_shape_dict (dict): map from input name to shape. (str -> list of ints)
    """

    @classmethod
    def __init_subclass__(cls, op_name=None):
        """register in CAFFE_TO_ONNX.

        Args:
            op_name (str): The type name of caffe layer the subclass will handle.
        """
        if op_name is None:
            op_name = cls.__name__
        CAFFE_TO_ONNX[op_name] = cls

    def __init__(self, layer, input_shape_dict):
        self.layer = layer

        self.input_shape_dict = input_shape_dict

        self.weight = {}
        self.weight_names = None

    def trans(self):
        """Do the convert process. Derivate class can override this method to **fully control** the convert process.

        Returns:
            op.Op or list of op.Op: The ONNX ops converted from the caffe layer.
        """
        # return node or [node1,node2] or (node1,node2)
        self.onnx_op = self.get_onnx_op()

        # self.onnx_op.input_and_weight = list(self.layer.bottom)
        attrs = self.get_attr()
        if not attrs:
            attrs = {}
        self.onnx_op.attributes.update(
            {key: helper.make_attribute(key, value) for key, value in attrs.items()}
        )

        self.weight, self.weight_names = self.get_param()

        self.onnx_op.input.extend(self.get_inputs())
        self.onnx_op.output.extend(self.get_outputs())

        if not self.onnx_op.satisfy():
            warnings.warn(
                f"lack of some attr in onnx op {self.onnx_op.__class__.__name__}"
            )

        return [*self.get_extra_nodes(), self.onnx_op]

    def get_onnx_op(self):
        """Derivate class can override this method, to generate target ONNX op.

        This method only generates op Node, no need to do any other processing.

        Returns:
            op.Op: The target ONNX op node.
        """
        op_type = self.get_type()
        if isinstance(op_type, (tuple, list)):
            op_type, domain, version = op_type
        else:
            domain, version = "", 9
        return Op.gen_op(op_type, self.layer.name, domain=domain, version=version)

    def get_type(self):
        """Derivate class can override this method, to get target ONNX op type.

        The default implementation returns the caffe layer type.

        Returns:
            str: The disired op_type of target ONNX op.
        """
        return self.layer.type

    def get_attr(self):
        """Derivate class **should** override this method, to generate target ONNX op's attributes.

        Returns:
            dict: A dict contains attributes, which maps from attribute name to value.
        """
        raise NotImplementedError

    def get_inputs(self):
        """Derivate class can override this method, get generate target ONNX op's input names.

        The default implementation returns caffe layer's bottom names, followed by the layer's blob names.
        If the order is different from the definition of target ONNX op, derivate class should override this method.

        Returns:
            list of strs: A list containing input names of target ONNX op.
        """
        return list(self.layer.bottom) + self.weight_names

    def get_outputs(self):
        """Derivate class can override this method, get generate target ONNX op's input names.

        The default implementation returns caffe layer's top names.
        If the order is different from the definition of target ONNX op, derivate class should override this method.

        Returns:
            list of strs: A list containing output names of target ONNX op.
        """
        return list(self.layer.top)

    def get_param(self):
        weight_dict = {}
        weight_names = []
        import copy

        # TODO: Is there any need to deepcopy the blob?
        Params = copy.deepcopy(self.layer.blobs)
        param_id = 0
        for tensor in Params:
            param_id = param_id + 1
            param_name = self.layer.name + "_param_" + str(param_id)
            param_tensor_info = helper.make_tensor_value_info(
                param_name, TensorProto.FLOAT, list(tensor.shape.dim)
            )
            param_tensor = helper.make_tensor(
                param_name, TensorProto.FLOAT, list(tensor.shape.dim), tensor.data
            )
            # self.onnx_op.input_and_weight.append(param_name)
            # self.onnx_op.input.extend([param_name])
            weight_names.append(param_name)
            weight_dict[param_name] = (param_tensor_info, param_tensor)
        return weight_dict, weight_names

    def get_extra_nodes(self):
        """Derivates can override this method to generate extra nodes required to transfrom caffe layer to onnx node.

            Those extra nodes will be placed in front of self.onnx_op, so they are most probably constant nodes.

        Returns:
            list<Op>: a list of nodes to be placed in front of self.onnx_op.
        """
        return []


# convert layer directly


class Default(BaseNode):
    def get_attr(self):
        """A common implementation of BaseNode.get_attr."""
        settings = {}
        if self.layer.type in op_to_param:
            temp_parameter = getattr(self.layer, op_to_param[self.layer.type])
            for info in temp_parameter.ListFields():
                if type(info[1]) == caffe_pb2.FillerParameter:
                    continue
                if info[0].name == "pad":
                    settings["pad_h"] = info[1]
                    settings["pad_w"] = info[1]
                    continue
                elif info[0].name == "kernel_size":
                    settings["kernel_h"] = info[1]
                    settings["kernel_w"] = info[1]
                    continue
                elif info[0].name == "stride":
                    settings["stride_h"] = info[1]
                    settings["stride_w"] = info[1]
                    continue
                elif info[0].name == "hole":
                    settings["hole_h"] = info[1]
                    settings["hole_w"] = info[1]
                    continue
                settings[info[0].name] = info[1]
            # add defualt value
            for i in temp_parameter.DESCRIPTOR.fields:
                val = getattr(temp_parameter, i.name)
                if type(val) == caffe_pb2.FillerParameter:
                    continue
                if i.name not in settings:
                    settings[i.name] = val
        return settings

    # def get_param(self):
    #     weight_dict = {}
    #     import copy
    #     Params = copy.deepcopy(self.layer.blobs)
    #     param_id = 0
    #     for tensor in Params:
    #         param_id = param_id + 1
    #         param_name = self.layer.name + "_param_" + str(param_id)
    #         param_tensor_info = helper.make_tensor_value_info(param_name,
    #             TensorProto.FLOAT,
    #             list(tensor.shape.dim))
    #         param_tensor = helper.make_tensor(param_name,
    #             TensorProto.FLOAT,
    #             list(tensor.shape.dim),
    #             tensor.data)
    #         #self.onnx_op.input_and_weight.append(param_name)
    #         self.onnx_op.input.extend([param_name])
    #         weight_dict[param_name] = (param_tensor_info ,param_tensor)
    #     return weight_dict


def blob_to_array(blob):
    """Covnert caffe blob to numpy array.

    Args:
        blob (caffe_pb2.BlobProto): the blob to be converted.

    Returns:
        numpy.array: the numpy array converted from caffe blob.
    """
    array = np.array(blob.data, np.float32)
    shape = list(blob.shape.dim)
    array = np.reshape(array, shape)
    return array


class Pad(BaseNode):
    def get_type(self):
        """ """
        return "Pad"

    def get_attr(self):
        """ """
        settings = {}
        pad = self.layer.pad_param
        settings["mode"] = pad.mode
        settings["value"] = pad.value
        settings["pads"] = pad.pads
        return settings


class ConvolutionReLU(BaseNode):
    def get_attr(self):
        """Generate ConvolutionReLU attributes."""
        pass


class Convolution(BaseNode):
    def get_type(self):
        """Returns 'Conv'"""
        return "Conv"

    def get_attr(self):
        """Generate Convolution attributes"""
        settings = {}
        conv = self.layer.convolution_param
        if conv.HasField("kernel_h") and conv.HasField("kernel_w"):
            kernel_shape = [conv.kernel_h, conv.kernel_w]
        else:
            kernel_shape = [conv.kernel_size] * 2
        if conv.HasField("stride_h") and conv.HasField("stride_w"):
            strides = [conv.stride_h, conv.stride_w]
        else:
            strides = [conv.stride] * 2
        if conv.HasField("pad_h") and conv.HasField("pad_w"):
            pads = [conv.pad_h, conv.pad_w] * 2
        else:
            pads = [conv.pad] * 4
        group = conv.group

        # err_str = "convolution layer should not use hole parameter"
        # assert conv.hole == 1, err_str
        # assert conv.hole_h == 1, err_str
        # assert conv.hole_w == 1, err_str

        settings["dilations"] = [conv.hole_h, conv.hole_w]
        settings["kernel_shape"] = kernel_shape
        settings["strides"] = strides
        settings["pads"] = pads
        settings["group"] = group

        return settings

    def get_param(self):
        weight_dict = {}
        weight_names = []

        conv = self.layer.convolution_param
        if conv.HasField("kernel_h") and conv.HasField("kernel_w"):
            kernel_shape = [conv.kernel_h, conv.kernel_w]
        else:
            kernel_shape = [conv.kernel_size] * 2

        import copy

        # TODO: Is there any need to deepcopy the blob?
        Params = copy.deepcopy(self.layer.blobs)

        # weight
        param_name = self.layer.name + ".weight"
        tensor = Params[0]
        shape = [
            conv.num_output,
            self.input_shape_dict[self.layer.bottom[0]][1] // conv.group,
            *kernel_shape,
        ]
        shape = [int(x) for x in shape]
        assert len(tensor.data) == np.multiply.reduce(
            shape
        ), f"expected kernel size to be {np.multiply.reduce(shape)}, got {len(tensor.data)}"
        param_tensor_info = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, shape
        )
        param_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, shape, tensor.data
        )
        weight_names.append(param_name)
        weight_dict[param_name] = (param_tensor_info, param_tensor)

        # bias
        if self.layer.convolution_param.bias_term:
            param_name = self.layer.name + ".bias"
            tensor = Params[1]
            param_tensor_info = helper.make_tensor_value_info(
                param_name, TensorProto.FLOAT, list(tensor.shape.dim)
            )
            param_tensor = helper.make_tensor(
                param_name, TensorProto.FLOAT, list(tensor.shape.dim), tensor.data
            )
            weight_names.append(param_name)
            weight_dict[param_name] = (param_tensor_info, param_tensor)

        return weight_dict, weight_names


# batchnorm + scale
class BN(BaseNode):
    def get_type(self):
        """Returns BatchNormalization."""
        return "BatchNormalization"

    def get_param(self):
        weight_dict = {}
        weight_names = []
        import copy

        Params = copy.deepcopy(self.layer.blobs)
        param_id = 0
        for tensor in Params:
            param_id = param_id + 1
            param_name = self.layer.name + "_param_" + str(param_id)
            param_tensor_info = helper.make_tensor_value_info(
                param_name, TensorProto.FLOAT, tensor.shape.dim[1:2]
            )
            param_tensor = helper.make_tensor(
                param_name, TensorProto.FLOAT, tensor.shape.dim[1:2], tensor.data
            )
            # self.onnx_op.input_and_weight.append(param_name)
            # self.onnx_op.input.extend([param_name])
            weight_names.append(param_name)
            weight_dict[param_name] = (param_tensor_info, param_tensor)
        return weight_dict, weight_names

    def get_attr(self):
        """Generate BN attributes"""
        settings = {}
        batchnorm = self.layer.bn_param
        # settings['is_test'] = batchnorm.use_global_stats
        settings["epsilon"] = batchnorm.var_eps
        return settings


class BatchNorm(BaseNode):
    def get_type(self):
        return "CaffeBatchNorm"

    def get_attr(self):
        settings = {}
        batchnorm = self.layer.batch_norm_param
        if batchnorm.use_global_stats != True:
            logger.warning(
                "Encountered BatchNorm layer with use_global_stats==False,"
                "if you are doing precision benchmark, modify it to True"
            )
        settings["eps"] = batchnorm.eps
        return settings


class BatchNorm3d(BN):
    def get_type(self):
        """Returns BatchNormalization."""
        return "BatchNormalization"

    def get_attr(self):
        """Generate BN attributes"""
        settings = {}
        batchnorm = self.layer.batchnorm3d_param
        settings["epsilon"] = batchnorm.eps
        settings["training_mode"] = not batchnorm.use_global_stats
        settings["momentum"] = batchnorm.moving_average_fraction
        return settings


class Deconvolution(BaseNode):
    def get_type(self):
        """Returns ConvTranspose"""
        return "ConvTranspose"

    def get_attr(self):
        """Generate Deconvolution attributes"""
        settings = {}
        deconv = self.layer.convolution_param
        if deconv.HasField("kernel_h") and deconv.HasField("kernel_w"):
            kernel_shape = [deconv.kernel_h, deconv.kernel_w]
        else:
            kernel_shape = [deconv.kernel_size] * 2
        if deconv.HasField("stride_h") and deconv.HasField("stride_w"):
            strides = [deconv.stride_h, deconv.stride_w]
        else:
            strides = [deconv.stride] * 2
        if deconv.HasField("pad_h") and deconv.HasField("pad_w"):
            pads = [deconv.pad_h, deconv.pad_w] * 2
        else:
            pads = [deconv.pad] * 4
        group = deconv.group

        assert deconv.hole_h == 1
        assert deconv.hole_w == 1

        settings["dilations"] = [deconv.hole_h, deconv.hole_w]
        settings["kernel_shape"] = kernel_shape
        settings["strides"] = strides
        settings["pads"] = pads
        settings["group"] = group

        return settings


# own def
class Correlation(BaseNode):
    def get_type(self):
        """Returns Corr"""
        return "Corr"

    def get_inputs(self):
        return self.layer.bottom[::-1]

    def get_attr(self):
        """Generate Corr attributes"""
        settings = {}
        correlation = self.layer.correlation_param
        settings["groups"] = correlation.groups
        return settings


class Correlation1D(BaseNode):
    def get_type(self):
        """Returns Corr"""
        return "Correlation1D"

    def get_attr(self):
        """Generate Correlation1D attributes"""
        settings = {}
        correlation = self.layer.correlation_param
        settings["max_displacement"] = correlation.max_displacement
        settings["pad"] = correlation.pad
        settings["single_direction"] = correlation.single_direction
        settings["kernel_size"] = correlation.kernel_size
        return settings


class Eltwise(BaseNode):
    MUL = 0
    ADD = 1
    MAX = 2

    def __init__(self, layer, input_shape_dict):
        eltwise = layer.eltwise_param
        self.op = None
        # nothing in weight
        self.weight = {}
        if eltwise.operation == Eltwise.MUL:
            self.op = Mul(layer, input_shape_dict)
        elif eltwise.operation == Eltwise.ADD:
            self.op = Add(layer, input_shape_dict)
        elif eltwise.operation == Eltwise.MAX:
            self.op = Max(layer, input_shape_dict)
        else:
            warnings.warn("unknown eltwise operation type")

    def trans(self):
        """Redirects to corresponding elementwise node's transformer."""
        return self.op.trans()

    @staticmethod
    def gen_reduce(operands, gen_op, res_name):
        """Generate a series of binary Ops to do reduction.

        Args:
            operands (list of str): The operands names.
            gen_op (callable object): A callable object to generate a binary Op. The signature should be ``Op (int id)``.
            res_name: The result tensor name.

        Returns:
            list of Ops: the generated Ops.
        """
        last = operands[0]
        ret = []
        for id, operand in enumerate(operands[1:], start=1):
            node = gen_op(id)
            node.add_input(last)
            node.add_input(operand)
            out_name = (
                "{0}_out".format(node.name) if (id + 1 != len(operands)) else res_name
            )
            node.add_output(out_name)
            last = out_name
            ret.append(node)
        return ret


class Mul(BaseNode):
    def __init__(self, layer, input_shape_dict):
        self.layer = layer
        eltwise = self.layer.eltwise_param
        assert eltwise.operation == Eltwise.MUL
        self.input_shape_dict = input_shape_dict

        self.weight = {}

    def trans(self):
        """Do the trans process of caffe Mul eltwise layer.

        A caffe Mul eltwise layer will be transformed into a sequence of Mul nodes.
        """
        layer = self.layer
        return Eltwise.gen_reduce(
            list(layer.bottom),
            lambda id: Op.gen_op("Mul", self.layer.name + f"{DELIM}mul{id}"),
            layer.top[0],
        )


class Add(BaseNode):
    def __init__(self, layer, input_shape_dict):
        self.layer = layer
        eltwise = self.layer.eltwise_param
        assert eltwise.operation == Eltwise.ADD
        self.input_shape_dict = input_shape_dict

        self.weight = {}

    def trans(self):
        """Do the trans process of caffe Add eltwise layer.

        A caffe Add eltwise layer may be transformed into a sequence of Mul nodes followed by Add nodes, if the coefficients are not 1.
        """
        eltwise = self.layer.eltwise_param
        coeffs = list(eltwise.coeff)
        if len(self.layer.bottom) > len(eltwise.coeff):
            coeffs = list(eltwise.coeff) + [1.0] * (
                len(self.layer.bottom) - len(eltwise.coeff)
            )

        input_shape_dict = self.input_shape_dict
        layer = self.layer

        from ...ops.op import Constant
        import math

        if len(layer.bottom) == 2 and all(math.isclose(coef, 1.0) for coef in coeffs):
            # simply a+b
            add_op = Op.gen_op("Add", layer.name)
            add_op.input.extend(layer.bottom)
            add_op.add_output(layer.top[0])
            return add_op

        if (
            len(layer.bottom) == 2
            and math.isclose(coeffs[0], 1.0)
            and math.isclose(coeffs[1], -1.0)
        ):
            # a-b
            sub_op = Op.gen_op("Sub", layer.name)
            sub_op.input.extend(layer.bottom)
            sub_op.add_output(layer.top[0])
            return sub_op

        onnx_op = []  # [Op.gen_op('Add', self.layer.name + '_add')]
        sum_operands = []  # the operands of final summation.
        for index, coeff in enumerate(coeffs):
            # flaot ?
            """
            if coeff == 1.0:
                add_op = self.onnx_op[len(self.onnx_op) - 1]
                add_op.input_and_weight.insert(0, self.layer.bottom[input_ind])
                input_ind = input_ind + 1
                continue
            """
            if math.isclose(coeff, 1.0):
                # multiply not needed, the sum operand is exactly bottom[index]
                sum_operands.append(self.layer.bottom[index])
                continue
            # coefficient constant
            coef = np.array([coeff], dtype=np.float32)
            # reshape to same dimension as the another operand
            ndim = len(input_shape_dict[self.layer.bottom[index]])
            coef = np.reshape(coef, [1] * ndim)
            constant_op_name = f"{self.layer.name}{DELIM}constant{index}"
            constant_op = Constant.make_constant(constant_op_name, coef)
            onnx_op.append(constant_op)
            # Mul
            mul_op = Op.gen_op("Mul", f"{self.layer.name}{DELIM}mul{index}")
            mul_op.input.extend([constant_op_name, self.layer.bottom[index]])
            mul_op_outname = self.layer.name + "_mul" + str(index)
            mul_op.add_output(mul_op_outname)
            onnx_op.append(mul_op)
            # constant-mul-add
            sum_operands.append(mul_op_outname)

        # now sum them up
        onnx_op.extend(
            Eltwise.gen_reduce(
                sum_operands,
                lambda id: Op.gen_op("Add", f"{self.layer.name}{DELIM}add{id}"),
                layer.top[0],
            )
        )

        return onnx_op


class Max(BaseNode):
    def __init__(self, layer, input_shape_dict):
        self.layer = layer
        eltwise = self.layer.eltwise_param
        assert eltwise.operation == Eltwise.MAX
        self.input_shape_dict = input_shape_dict

        self.weight = {}

    def trans(self):
        """Do the trans process of caffe Mul eltwise layer.

        A caffe MAX eltwise layer will be transformed into a sequence of Max nodes.
        """
        layer = self.layer
        return Eltwise.gen_reduce(
            list(layer.bottom),
            lambda id: Op.gen_op("Max", f"{self.layer.name}{DELIM}max{id}"),
            layer.top[0],
        )


class Pooling(BaseNode):
    MAX = 0
    AVE = 1

    def get_type(self):
        """Get op type according to pooling type. It can be MaxPool or AveragePool."""
        pool = self.layer.pooling_param
        if not pool.global_pooling:
            if pool.pool == Pooling.MAX:
                version = 9 if pool.ceil_mode == 0 else 10
                return "MaxPool", "", version
            elif pool.pool == Pooling.AVE:
                version = 9 if pool.ceil_mode == 0 else 10
                return "AveragePool", "", version
        else:
            if pool.pool == Pooling.MAX:
                return "GlobalMaxPool"
            elif pool.pool == Pooling.AVE:
                return "GlobalAveragePool"

    def get_attr(self):
        """ """
        settings = {}
        pool = self.layer.pooling_param

        if pool.global_pooling:
            return settings

        if pool.ceil_mode:
            settings["ceil_mode"] = pool.ceil_mode

        if pool.HasField("kernel_h") and pool.HasField("kernel_w"):
            kernel_shape = [pool.kernel_h, pool.kernel_w]
        else:
            kernel_shape = [pool.kernel_size] * 2
        if pool.HasField("stride_h") and pool.HasField("stride_w"):
            strides = [pool.stride_h, pool.stride_w]
        else:
            strides = [pool.stride] * 2
        if pool.HasField("pad_h") and pool.HasField("pad_w"):
            pads = [pool.pad_h, pool.pad_w] * 2
        else:
            pads = [pool.pad] * 4

        settings["kernel_shape"] = kernel_shape
        settings["strides"] = strides
        settings["pads"] = pads
        if not pool.global_pooling and pool.pool == Pooling.AVE:
            # set count_include_pad for average pooling
            settings["count_include_pad"] = 1

        return settings


class Pooling3d(BaseNode):
    MAX = 0
    AVE = 1

    def get_type(self):
        """Get op type according to pooling type. It can be MaxPool or AveragePool."""
        if self.layer.pooling3d_param:
            pool = self.layer.pooling3d_param
        elif self.layer.maxpooling3d_param:
            pool = self.layer.maxpooling3d_param
        elif self.layer.avepooling3d_param:
            pool = self.layer.avepooling3d_param
        if not pool.global_pooling:
            if pool.pool == Pooling3d.MAX:
                return "MaxPool"
            elif pool.pool == Pooling3d.AVE:
                return "AveragePool"
        else:
            if pool.pool == Pooling3d.MAX:
                return "GlobalMaxPool"
            elif pool.pool == Pooling3d.AVE:
                return "GlobalAveragePool"

    def get_attr(self):
        """ """
        settings = {}
        if self.layer.pooling3d_param:
            pool = self.layer.pooling3d_param
        elif self.layer.maxpooling3d_param:
            pool = self.layer.maxpooling3d_param
        elif self.layer.avepooling3d_param:
            pool = self.layer.avepooling3d_param

        if pool.global_pooling:
            return settings

        settings["ceil_mode"] = pool.ceil_mode

        if (
            pool.HasField("kernel_d")
            and pool.HasField("kernel_h")
            and pool.HasField("kernel_w")
        ):
            kernel_shape = [pool.kernel_d, pool.kernel_h, pool.kernel_w]
        else:
            kernel_shape = [pool.kernel_size] * 3
        if (
            pool.HasField("stride_d")
            and pool.HasField("stride_h")
            and pool.HasField("stride_w")
        ):
            strides = [pool.stride_d, pool.stride_h, pool.stride_w]
        else:
            strides = [pool.stride] * 3
        if pool.HasField("pad_d") and pool.HasField("pad_h") and pool.HasField("pad_w"):
            pads = [pool.pad_d, pool.pad_h, pool.pad_w] * 2
        else:
            pads = [pool.pad] * 6

        settings["kernel_shape"] = kernel_shape
        settings["strides"] = strides
        settings["pads"] = pads
        if not pool.global_pooling and pool.pool == Pooling.AVE:
            # set count_include_pad for average pooling
            settings["count_include_pad"] = 1

        return settings


class InnerProductReLU(BaseNode):
    def get_attr(self):
        """ """
        pass


# need to change to gemm (how to solve axis)


class InnerProduct(BaseNode):
    def get_onnx_op(self):
        """ """
        return []

    def trans(self):
        """Do trans process of InnerProduct layer.

        An InnerProduct layer will be transformed into an optional Reshape node, followed by a GEMM node.
        The Reshape is needed when input is not two dimensional, to flatten the input tensor into 2 dimensional.

        The output tensor will always be 2 dimensional.
        """
        onnx_ops = []
        layer = self.layer
        input_shape = list(self.input_shape_dict[layer.bottom[0]])
        inner_product = self.layer.inner_product_param
        axis = inner_product.axis
        if len(input_shape) == 2 and axis == 1:
            # if the input is a two dimensional matrix, simply transform to Gemm op.
            self.weight, self.weight_names = self.get_param()
            gemm_op = Op.gen_op("Gemm", layer.name)
            gemm_op.add_input(layer.bottom[0])
            gemm_op.input.extend(self.weight_names)
            gemm_op.add_output(layer.top[0])
            # transB = 1 if not inner_product.transpose else 0
            transB = 1
            gemm_op.attributes["transB"] = helper.make_attribute("transB", transB)
            return gemm_op
        else:
            # first reshape
            from functools import reduce

            flatten_shape = [0, -1]
            reshape1_out_name = layer.bottom[0] + "_flatten"
            from ...ops.op import Reshape

            onnx_ops.extend(
                Reshape.make_reshape(
                    f"{layer.name}{DELIM}reshape1",
                    layer.bottom[0],
                    reshape1_out_name,
                    flatten_shape,
                )
            )

            # then do gemm
            self.weight, self.weight_names = self.get_param()
            gemm_op = Op.gen_op("Gemm", layer.name)
            gemm_op.add_input(reshape1_out_name)
            gemm_op.input.extend(self.weight_names)
            gemm_out_name = layer.top[0]
            gemm_op.add_output(gemm_out_name)
            # transB = 1 if not inner_product.transpose else 0
            transB = 1
            gemm_op.attributes["transB"] = helper.make_attribute("transB", transB)
            onnx_ops.append(gemm_op)

            # In CaffeInfer's behavior, the output dimension of InnerProduct is 2, so no need to reshape back.
            # second reshape
            # out_shape = input_shape[0:axis] + [inner_product.num_output] + [1] * (len(input_shape) - axis - 1)
            # onnx_ops.extend(Reshape.make_reshape(layer.name + '_reshape2', gemm_out_name, layer.top[0], out_shape))
            return onnx_ops


class ReLU(BaseNode):
    def get_type(self):
        """ """
        relu = self.layer.relu_param
        if relu.HasField("negative_slope"):
            return "LeakyRelu"
        return "Relu"

    def get_attr(self):
        """ """
        settings = {}
        relu = self.layer.relu_param
        if relu.HasField("negative_slope"):
            settings["alpha"] = relu.negative_slope
        return settings


class ReLU3d(BaseNode):
    def get_type(self):
        """ """
        return "ReLU3d"

    def get_attr(self):
        """ """
        settings = {}
        relu = self.layer.relu3d_param
        if relu.HasField("negative_slope"):
            settings["negative_slope"] = relu.negative_slope
        if relu.HasField("channel_shared"):
            settings["channel_shared"] = relu.channel_shared
        return settings


class PReLU(BaseNode):
    def get_type(self):
        return "PRelu"

    def get_attr(self):
        return {}

    def get_param(self):
        weight_dict = {}
        weight_names = []
        import copy

        Params = copy.deepcopy(self.layer.blobs)
        layer = self.layer
        # only exepct a slops weight
        assert len(Params) == 1
        tensor = Params[0]
        param_name = layer.name + "_param_slops"
        input_nb_dim = len(self.input_shape_dict[layer.bottom[0]])
        slops_shape = [1] + list(tensor.shape.dim)
        slops_shape.extend([1] * (input_nb_dim - len(slops_shape)))
        param_tensor_info = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, slops_shape
        )
        param_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, slops_shape, tensor.data
        )

        weight_names.append(param_name)
        weight_dict[param_name] = (param_tensor_info, param_tensor)
        return weight_dict, weight_names


class TanH(BaseNode):
    def get_type(self):
        """ """
        return "Tanh"

    def get_attr(self):
        """ """
        # nothing to do
        pass


class Sigmoid(BaseNode):
    def get_attr(self):
        """ """
        # nothing to do
        pass


class Hsigmoid(BaseNode):
    def get_attr(self):
        """ """
        settings = {
            "alpha": 1 / 6.0,
            "beta": 0.5,
        }
        pass


class HSwish(BaseNode):
    def get_type(self):
        """Returns 'Hswish'"""
        return "Hswish"

    def get_attr(self):
        """ """
        # nothing to do
        pass


class HSigmoid(BaseNode):
    def get_type(self):
        return "Hsigmoid"

    def get_attr(self):
        settings = {
            "alpha": 1 / 6.0,
            "beta": 0.5,
        }
        return settings


class Hswish(BaseNode):
    def get_type(self):
        return "Hswish"

    def get_attr(self):
        settings = {}
        return settings


class Concat(BaseNode):
    def get_attr(self):
        """ """
        settings = {}
        concat = self.layer.concat_param
        settings["axis"] = concat.axis
        return settings


# split to Constant and Reshape
class Reshape(BaseNode):
    def __init__(self, layer, input_shape_dict):
        self.layer = layer

        onnx_op = []
        onnx_op.append(Op.gen_op("Constant", f"{self.layer.name}{DELIM}constant"))
        onnx_op.append(Op.gen_op(self.get_type(), self.layer.name))
        self.param = self.layer.reshape_param
        self.onnx_ops = onnx_op
        # Constant and Reshape
        assert len(self.onnx_ops) == 2
        self.weight = {}

    def trans(self):
        """ """
        # Constant
        constant_op = self.onnx_ops[0]
        # constant_op.input_and_weight = []
        constant_op.output.extend([self.layer.name + "_shape_const"])

        reshape = self.param
        value_tensor = helper.make_tensor(
            self.layer.name + "_shape_const",
            TensorProto.INT64,
            [
                len(reshape.shape.dim[:]),
            ],
            np.array(reshape.shape.dim).tostring(),
            raw=True,
        )

        constant_op.attributes["value"] = helper.make_attribute("value", value_tensor)
        # Reshape
        reshape_op = self.onnx_ops[1]
        # reshape_op.input_and_weight = list(self.layer.bottom)
        # reshape_op.input_and_weight.append(self.layer.name + '_constant')
        reshape_op.input.extend(list(self.layer.bottom))
        reshape_op.input.extend(
            [
                self.layer.name + "_shape_const",
            ]
        )
        reshape_op.output.extend(list(self.layer.top))

        self.weight = {}
        if not self.onnx_ops[0].satisfy():
            warnings.warn(
                f"lack of some attr in onnx op {self.onnx_ops.__class__.__name__}"
            )
        if not self.onnx_ops[1].satisfy():
            warnings.warn(
                f"lack of some attr in onnx op {self.onnx_ops.__class__.__name__}"
            )

        node_list = []
        for node in self.onnx_ops:
            node_list.append(node)
        return node_list


class Softmax(BaseNode):
    def get_type(self):
        return "CaffeSoftmax"

    def get_attr(self):
        """ """
        settings = {}
        softmax = self.layer.softmax_param
        settings["axis"] = softmax.axis
        return settings


class Transpose(BaseNode):
    def get_attr(self):
        """ """
        settings = {}
        transpose = self.layer.transpose_param
        settings["perm"] = transpose.dim[:]
        return settings


def adjusted_scale(a, b, init_scale=None):
    """calculate s=a/b, which guarantees floor(s * a) == b."""
    # force convert to float32
    dtype = np.float32
    a = dtype(a)
    b = dtype(b)
    if init_scale is None:
        scale = a / b
    else:
        scale = dtype(init_scale)
    while np.floor(scale * b) < a:
        # change scale to the next larger floating-point (float32) value after it, if floor(scale*b) < a.
        scale = np.nextafter(scale, np.inf, dtype=dtype)
        assert scale.dtype == np.float32
    scale = np.nextafter(scale, np.inf, dtype=dtype)
    return scale


class _InterpCommon(BaseNode):
    def __init__(self, layer, input_shape_dict):
        self.layer = layer
        self.input_shape_dict = input_shape_dict
        self.weight = {}

    def trans(self):
        if len(self.layer.bottom) == 2:
            # dynamic upsample
            output_shape = self.input_shape_dict[self.layer.bottom[1]]
            dynupsample = Op.make_op(
                "DynamicUpsample",
                self.layer.name,
                self.layer.bottom,
                self.layer.top,
                self.get_attr(),
            )
            return [dynupsample]
        param = self.get_param()
        layer = self.layer
        # compute scale
        all_fields = [field[0].name for field in param.ListFields()]
        input_size = self.input_shape_dict[layer.bottom[0]]
        if (
            "height" in all_fields
            and param.HasField("height")
            and param.HasField("width")
        ):
            height_scale = adjusted_scale(param.height, input_size[2])
            width_scale = adjusted_scale(param.width, input_size[3])
        elif "scale_factor" in all_fields and param.HasField("scale_factor"):
            height_scale = width_scale = param.scale_factor
        elif "resize" in all_fields and param.HasField("resize"):
            height_scale = width_scale = param.resize
        elif "zoom_factor" in all_fields and param.HasField("zoom_factor"):
            logger.warn("Not available dynamic input shape for zoom_factor!")
            oh = input_size[2] + (input_size[2] - 1) * (param.zoom_factor - 1)
            ow = input_size[3] + (input_size[3] - 1) * (param.zoom_factor - 1)
            height_scale = adjusted_scale(oh, input_size[2])
            width_scale = adjusted_scale(ow, input_size[3])
        elif "shrink_factor" in all_fields and param.HasField("shrink_factor"):
            logger.warn("Not available dynamic input shape for shrink_factor!")
            oh = (input_size[2] - 1) / param.shrink_factor + 1
            ow = (input_size[3] - 1) / param.shrink_factor + 1
            height_scale = adjusted_scale(oh, input_size[2])
            width_scale = adjusted_scale(ow, input_size[3])
        else:
            logger.fatal("Unsupported interpolate layer: {}")
        scales = np.array([1.0, 1.0, height_scale, width_scale], dtype=np.float32)
        from ...ops.op import Constant

        constant_op = Constant.make_constant(f"{self.layer.name}_scale_const", scales)
        upsample_op = Op.make_op(
            "Upsample",
            layer.name,
            [self.layer.bottom[0], constant_op.output[0]],
            self.layer.top,
            self.get_attr(),
        )
        ret = [constant_op, upsample_op]
        return ret

    def get_param(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} should override get_attr method"
        )

    def get_attr(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} should override get_attr method"
        )


class Interp(_InterpCommon):
    def get_param(self):
        return self.layer.interp_param

    def get_attr(self):
        settings = {}
        settings["mode"] = "linear"
        return settings


class Resize(_InterpCommon):
    def get_param(self):
        return self.layer.resize_param

    def get_attr(self):
        settings = {}
        settings["mode"] = (
            "linear" if self.layer.resize_param.interp_mode == 1 else "nearest"
        )
        return settings


class NNInterp(_InterpCommon):
    def get_param(self):
        return self.layer.nninterp_param

    def get_attr(self):
        settings = {}
        settings["mode"] = "nearest"
        return settings


class NNUpsample(_InterpCommon):
    def get_param(self):
        return self.layer.nn_upsample_param

    def get_attr(self):
        settings = {}
        settings["mode"] = "nearest"
        return settings


# how to name (RoiPool not in onnx)
class ROIPooling(BaseNode):
    def get_type(self):
        """ """
        # MaxPoiPool or PorPool
        return "MaxRoiPool"

    def get_attr(self):
        """ """
        settings = {}
        roipooling = self.layer.roi_pooling_param
        settings["spatial_scale"] = roipooling.spatial_scale
        settings["pooled_shape"] = [roipooling.pooled_h, roipooling.pooled_w]
        return settings


class ROIAlignPooling(BaseNode):
    def get_type(self):
        """ """
        return "RoiAlign"

    def get_attr(self):
        """ """
        settings = {}
        roialignpooling = self.layer.roi_align_pooling_param
        settings["spatial_scale"] = roialignpooling.spatial_scale
        settings["pooled_height"] = roialignpooling.pooled_h
        settings["pooled_width"] = roialignpooling.pooled_w
        settings["sample_num"] = roialignpooling.sample_num
        return settings


class PODROIAlignPooling(BaseNode):
    def get_type(self):
        """ """
        return "PODRoiAlign"

    def get_attr(self):
        """ """
        settings = {}
        podroialignpooling = self.layer.podroi_align_pooling_param
        settings["spatial_scale"] = podroialignpooling.spatial_scale
        settings["pooled_height"] = podroialignpooling.pooled_h
        settings["pooled_width"] = podroialignpooling.pooled_w
        settings["sample_num"] = podroialignpooling.sample_num
        return settings


class PSROIPooling(BaseNode):
    def get_type(self):
        """ """
        return "PSRoiPool"

    def get_attr(self):
        """ """
        settings = {}
        psroipooling = self.layer.psroi_pooling_param
        settings["spatial_scale"] = psroipooling.spatial_scale
        settings["group_size"] = psroipooling.group_size
        settings["output_dim"] = psroipooling.output_dim
        return settings


class PSROIMaskPooling(BaseNode):
    def get_type(self):
        """ """
        return "PSRoiMaskPool"

    def get_attr(self):
        """ """
        settings = {}
        psroimaskpooling = self.layer.psroi_mask_pooling_param
        settings["spatial_scale"] = psroimaskpooling.spatial_scale
        settings["output_dim"] = psroimaskpooling.output_dim
        settings["group_size"] = psroimaskpooling.group_size
        settings["roi_scale"] = psroimaskpooling.roi_scale
        settings["bin_scale"] = psroimaskpooling.bin_scale
        return settings


# transpose
class Exchange(BaseNode):
    def get_type(self):
        """ """
        return "Transpose"

    def get_attr(self):
        """ """
        settings = {}
        # nchw to wnch
        settings["perm"] = [3, 0, 1, 2]
        return settings


class ShuffleChannel(BaseNode):
    def get_attr(self):
        """ """
        settings = {}
        assert self.layer.HasField("shuffle_channel_param")
        shufflechannel = self.layer.shuffle_channel_param
        # coordinate to tools
        settings["group"] = shufflechannel.group
        return settings


class HeatMap2Coord(BaseNode):
    def get_attr(self):
        """ """
        settings = {}
        heatmap2coord = self.layer.heatmap_param
        settings["coord_h"] = heatmap2coord.coord_h
        settings["coord_w"] = heatmap2coord.coord_w
        settings["coord_reposition"] = heatmap2coord.coord_reposition
        return settings


class ReLU6(BaseNode):
    def get_type(self):
        """ """
        return "Clip"

    def get_attr(self):
        """ """
        settings = {}
        relu6 = self.layer.relu6_param
        settings["min"] = 0.0
        settings["max"] = 6.0
        return settings


class Slice(BaseNode):
    def get_type(self):
        return "Split"

    def get_attr(self):
        settings = {}
        layer = self.layer
        input_shape = self.input_shape_dict[layer.bottom[0]]
        slice_param = layer.slice_param
        # split axis
        axis = slice_param.axis
        axis = axis if axis >= 0 else axis + len(input_shape)
        settings["axis"] = axis
        slice_points = slice_param.slice_point
        # infer split sizes from slice point. subtract ([0] + slice_points[0:-1]) from slice_points element-wisely.
        splits = list(map(lambda x, y: x - y, slice_points, [0] + slice_points[:-1]))
        splits.append(input_shape[axis] - slice_points[-1])
        settings["split"] = splits

        return settings


class Dropout(BaseNode):
    def get_type(self):
        return "Dropout"

    def get_attr(self):
        settings = {}
        return settings


class ChannelShuffle(BaseNode):
    def get_type(self):
        return "ShuffleChannel"

    def get_attr(self):
        settings = {}
        layer = self.layer
        assert layer.HasField("channel_shuffle_param")
        group = layer.channel_shuffle_param.group
        settings["group"] = group
        return settings


class HoleConvolution(BaseNode):
    def get_type(self):
        return "Conv"

    def get_attr(self):
        settings = {}
        layer = self.layer
        conv = self.layer.convolution_param
        if conv.HasField("kernel_h") and conv.HasField("kernel_w"):
            kernel_shape = [conv.kernel_h, conv.kernel_w]
        else:
            kernel_shape = [conv.kernel_size] * 2
        if conv.HasField("stride_h") and conv.HasField("stride_w"):
            strides = [conv.stride_h, conv.stride_w]
        else:
            strides = [conv.stride] * 2
        if conv.HasField("pad_h") and conv.HasField("pad_w"):
            pads = [conv.pad_h, conv.pad_w] * 2
        else:
            pads = [conv.pad] * 4
        group = conv.group
        if conv.HasField("hole_h") and conv.HasField("hole_w"):
            dilations = [conv.hole_h, conv.hole_w]
        else:
            assert conv.HasField("hole")
            dilations = [conv.hole, conv.hole]

        settings["dilations"] = dilations
        settings["kernel_shape"] = kernel_shape
        settings["strides"] = strides
        settings["pads"] = pads
        settings["group"] = group

        return settings


class Reduction(BaseNode):
    def get_type(self):
        from ...proto import caffe_pb2

        ReductionPara = caffe_pb2.ReductionParameter
        type_map = {
            ReductionPara.SUM: "ReduceSum",
            ReductionPara.ASUM: "ReduceL1",
            ReductionPara.SUMSQ: "ReduceSumSquare",
            ReductionPara.MEAN: "ReduceMean",
        }
        operation = self.layer.reduction_param.operation
        assert (
            operation in type_map
        ), f"unhandled reduction operation encountered: {self.layer.operation}"
        return type_map[operation]

    def get_attr(self):
        settings = {}
        reduction_param = self.layer.reduction_param
        axis = reduction_param.axis if reduction_param.HasField("axis") else 0
        ndim = len(self.input_shape_dict[self.layer.bottom[0]])
        # caffe reduction means do reduction along all tail axes.
        settings["axes"] = range(axis, ndim)
        settings["keepdims"] = 0
        return settings

    def trans(self):
        """Do transformation of Reduction layer."""
        layer = self.layer
        reduction_param = self.layer.reduction_param
        ops = []
        import math

        has_coeff = not math.isclose(reduction_param.coeff, 1.0)
        # generate reduce op
        reduce_op_name = layer.name if not has_coeff else f"{layer.name}{DELIM}reduce"
        reduce_op = Op.gen_op(self.get_type(), reduce_op_name)
        # set input and output
        reduce_op.add_input(layer.bottom[0])
        reduce_out_name = (
            layer.top[0] if not has_coeff else f"{layer.top[0]}{DELIM}interm"
        )
        reduce_op.add_output(reduce_out_name)
        # set attribute
        reduce_op.attributes.update(
            {
                key: helper.make_attribute(key, value)
                for key, value in self.get_attr().items()
            }
        )
        ops.append(reduce_op)
        self.weight, self.weight_names = self.get_param()

        if not reduce_op.satisfy():
            warnings.warn(
                f"lack of some attr in onnx op {self.reduce_op.__class__.__name__}"
            )

        if has_coeff:
            logger.warn(
                "Reduction layer with coeff encountered, converting to [ReduceXXX+Mul], which may cause convert error later"
            )
            coeff_op = Constant.make_constant(
                f"{layer.name}{DELIM}coeff",
                np.array(reduction_param.coeff, dtype=np.float32),
            )
            ops.append(coeff_op)
            # make multiply node
            from ...ops import Mul

            mul_op = Mul.make_mul(
                f"{layer.name}{DELIM}mul",
                reduce_out_name,
                coeff_op.output[0],
                layer.top[0],
            )
            ops.append(mul_op)
        return ops


class Scale(BaseNode):
    def get_type(self):
        return "CaffeScale"

    def get_attr(self):
        settings = {}
        layer = self.layer
        scale_param = layer.scale_param
        axis = scale_param.axis

        settings["axis"] = axis

        return settings


class Power(BaseNode):
    """Transfrom Caffe Power layer to onnx.
    Caffe Power layer computes outputs y = (shift + scale * x) ^ power, so at most 3 nodes will be generated (Mul+Add+Pow).
    """

    def get_attr(self):
        settings = {}
        param = self.layer.power_param
        if param.HasField("power"):
            settings["power"] = param.power
        if param.HasField("scale"):
            settings["scale"] = param.scale
        if param.HasField("shift"):
            settings["shift"] = param.shift
        return settings

    def trans(self):
        layer = self.layer
        ops = []
        data = layer.bottom[0]
        nb_dim = len(self.input_shape_dict[data])
        tensor = data

        import math

        power_param = layer.power_param
        scale = power_param.scale
        if not math.isclose(scale, 1.0):
            # need Mul to do scale
            scale_const = Constant.make_constant(
                f"{layer.name}{DELIM}scale_const", np.array(scale, dtype=np.float32)
            )
            mul_out_name = f"{tensor}_scaled"
            scale_op = Op.make_op(
                "Mul",
                f"{layer.name}{DELIM}mul",
                [tensor, scale_const.output[0]],
                [mul_out_name],
            )
            ops.extend([scale_const, scale_op])
            tensor = mul_out_name
        shift = power_param.shift
        if not math.isclose(shift, 0.0):
            # need Add to do bias
            shift_const = Constant.make_constant(
                f"{layer.name}{DELIM}shift_const", np.array(shift, dtype=np.float32)
            )
            add_out_name = f"{tensor}_shifted"
            shift_op = Op.make_op(
                "Add",
                f"{layer.name}{DELIM}add",
                [tensor, shift_const.output[0]],
                [add_out_name],
            )
            ops.extend([shift_const, shift_op])
            tensor = add_out_name

        # now generate Pow op.
        power = power_param.power
        power_const = Constant.make_constant(
            f"{layer.name}{DELIM}power_const", np.array(power, dtype=np.float32)
        )
        pow_op = Op.make_op(
            "Pow", f"{layer.name}", [tensor, power_const.output[0]], [layer.top[0]]
        )
        attrs = self.get_attr()
        if not attrs:
            attrs = {}
        pow_op.attributes.update(
            {key: helper.make_attribute(key, value) for key, value in attrs.items()}
        )
        ops.extend([power_const, pow_op])

        return ops


class Scales(BaseNode):
    """Transfrom Caffe Scales layer to onnx.
    Caffe Scales layer computes outputs y = beta + alpha * x, so at most 2 nodes will be generated (Mul+Add).
    """

    def trans(self):
        layer = self.layer
        ops = []
        data = layer.bottom[0]
        nb_dim = len(self.input_shape_dict[data])
        tensor = data

        import math

        scales_param = layer.scales_param
        scale = scales_param.alpha

        if not math.isclose(scale, 1.0):
            # need Mul to do scale
            scale_const = Constant.make_constant(
                f"{layer.name}{DELIM}scale_const", np.array(scale, dtype=np.float32)
            )
            mul_out_name = f"{tensor}_scaled"
            scale_op = Op.make_op(
                "Mul",
                f"{layer.name}{DELIM}mul",
                [tensor, scale_const.output[0]],
                [mul_out_name],
            )
            ops.extend([scale_const, scale_op])
            tensor = mul_out_name
        shift = scales_param.beta
        if not math.isclose(shift, 0.0):
            # need Add to do bias
            shift_const = Constant.make_constant(
                f"{layer.name}{DELIM}shift_const", np.array(shift, dtype=np.float32)
            )
            add_out_name = f"{tensor}_shifted"
            shift_op = Op.make_op(
                "Add",
                f"{layer.name}{DELIM}add",
                [tensor, shift_const.output[0]],
                [add_out_name],
            )
            ops.extend([shift_const, shift_op])
            tensor = add_out_name

        ops[-1].output[:] = [layer.top[0]]

        return ops


class MVN(BaseNode):
    def get_type(self):
        return "MeanVarianceNormalization"

    def get_attr(self):
        settings = {}
        layer = self.layer
        mvn_param = layer.mvn_param
        assert (
            mvn_param.normalize_variance == True
        ), "MVN layer that normalize mean only is not supported"
        ndim = len(self.input_shape_dict[layer.bottom[0]])
        if mvn_param.across_channels:
            axes = list(range(1, ndim))
        else:
            axes = list(range(2, ndim))
        settings["axes"] = axes
        return settings


class ArgMax(BaseNode):
    def get_type(self):
        return "TopK"

    def get_attr(self):
        settings = {}
        layer = self.layer
        argmax_param = layer.argmax_param
        # axis doesn't have default value
        if argmax_param.HasField("axis"):
            axis = argmax_param.axis
        else:
            # when axis not set, it means flattening trailing axes.
            axis = 1
            shape = self.input_shape_dict[layer.bottom[0]]
            if not all(x == 1 for x in shape[axis + 1 :]):
                logging.error(
                    f"ArgMax layer {layer.name} requires flattening trailing axes, which is not supported"
                )
        settings["axis"] = axis
        settings["k"] = argmax_param.top_k
        return settings

    def get_outputs(self):
        """ """
        layer = self.layer
        argmax_param = layer.argmax_param
        if argmax_param.out_max_val:
            assert (
                len(layer.top) == 2
            ), f"ArgMax layer {layer.name} out_max_val==True, but it has only {len(layer.top)} output(s)"
            indices, values = layer.top
            return [values, indices]
        else:
            assert (
                len(layer.top) == 1
            ), f"ArgMax layer {layer.name} out_max_val==False, but it has {len(layer.top)} outputs"
            indices = layer.top[0]
            values = f"{layer.name}_value_ignore"
            return [values, indices]


class SubpixelUp(BaseNode):
    def get_type(self):
        return "SubPixel"

    def get_attr(self):
        settings = {}
        layer = self.layer
        subpixel_param = layer.subpixel_up_param
        settings["method"] = 1
        settings["sample"] = subpixel_param.upsample
        return settings


class SubpixelDown(BaseNode):
    def get_type(self):
        return "SubPixel"

    def get_attr(self):
        settings = {}
        layer = self.layer
        subpixel_param = layer.subpixel_down_param
        settings["method"] = 0
        settings["sample"] = subpixel_param.downsample
        return settings


class Exp(BaseNode):
    def get_type(self):
        return "CaffeExp"

    def get_attr(self):
        settings = {}
        exp_param = self.layer.exp_param
        settings["base"] = exp_param.base
        settings["scale"] = exp_param.scale
        settings["shift"] = exp_param.shift
        return settings


class Parameter(BaseNode):
    def get_type(self):
        return "Parameter"

    def get_attr(self):
        settings = {}
        param = self.layer.parameter_param
        settings["batch"] = param.batch
        settings["channel"] = param.channel
        settings["height"] = param.height
        settings["width"] = param.width
        settings["m"] = param.m
        settings["n"] = param.n
        return settings

    def get_param(self):
        weight_dict = {}
        weight_names = []
        import copy

        Params = copy.deepcopy(self.layer.blobs)
        layer = self.layer
        # only exepct a slops weight
        assert len(Params) == 1
        tensor = Params[0]
        param_name = layer.top[0] + ".weight"
        param_tensor_info = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, list(tensor.shape.dim)
        )
        param_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, list(tensor.shape.dim), tensor.data
        )

        weight_names.append(param_name)
        weight_dict[param_name] = (param_tensor_info, param_tensor)
        return weight_dict, weight_names

    def get_onnx_op(self):
        """Derivate class can override this method, to generate target ONNX op.

        This method only generates op Node, no need to do any other processing.

        Returns:
            op.Op: The target ONNX op node.
        """
        return Op.gen_op(self.get_type(), self.get_type() + ":" + self.layer.name)

    def trans(self):
        # return node or [node1,node2] or (node1,node2)
        self.onnx_op = self.get_onnx_op()
        attrs = self.get_attr()
        if not attrs:
            attrs = {}
        self.onnx_op.attributes.update(
            {key: helper.make_attribute(key, value) for key, value in attrs.items()}
        )
        # from onnx import numpy_helper
        self.weight, self.weight_names = self.get_param()
        self.onnx_op.attributes["value"] = helper.make_attribute(
            "value", self.weight[self.weight_names[0]][1]
        )
        self.onnx_op.add_output(self.layer.top[0])

        return self.onnx_op


class Reshape3d(Reshape):
    def __init__(self, layer, input_shape_dict):
        super(Reshape3d, self).__init__(layer, input_shape_dict)
        self.param = self.layer.reshape3d_param

    def get_type(self):
        return "Reshape"


class Convolution3d(BaseNode):
    def get_type(self):
        """Returns 'Conv'"""
        return "Conv"

    def get_attr(self):
        """Generate Convolution3d attributes"""
        settings = {}
        conv = self.layer.convolution3d_param
        if (
            conv.HasField("kernel_d")
            and conv.HasField("kernel_h")
            and conv.HasField("kernel_w")
        ):
            kernel_shape = [conv.kernel_d, conv.kernel_h, conv.kernel_w]
        else:
            kernel_shape = [conv.kernel_size] * 3
        if (
            conv.HasField("stride_d")
            and conv.HasField("stride_h")
            and conv.HasField("stride_w")
        ):
            strides = [conv.stride_d, conv.stride_h, conv.stride_w]
        else:
            strides = [conv.stride] * 3
        if conv.HasField("pad_d") and conv.HasField("pad_h") and conv.HasField("pad_w"):
            pads = [conv.pad_d, conv.pad_h, conv.pad_w] * 2
        else:
            pads = [conv.pad] * 6
        group = conv.group
        if (
            conv.HasField("hole_d")
            and conv.HasField("hole_h")
            and conv.HasField("hole_w")
        ):
            hole = [conv.hole_d, conv.hole_h, conv.hole_w]
        else:
            hole = [conv.hole] * 3
        # err_str = "convolution layer should not use hole parameter"
        # assert conv.hole == 1, err_str
        # assert conv.hole_h == 1, err_str
        # assert conv.hole_w == 1, err_str

        settings["dilations"] = hole
        settings["kernel_shape"] = kernel_shape
        settings["strides"] = strides
        settings["pads"] = pads
        settings["group"] = group

        return settings

    def get_param(self):
        weight_dict = {}
        weight_names = []

        conv = self.layer.convolution3d_param
        if (
            conv.HasField("kernel_d")
            and conv.HasField("kernel_h")
            and conv.HasField("kernel_w")
        ):
            kernel_shape = [conv.kernel_d, conv.kernel_h, conv.kernel_w]
        else:
            kernel_shape = [conv.kernel_size] * 3

        import copy

        # TODO: Is there any need to deepcopy the blob?
        Params = copy.deepcopy(self.layer.blobs)

        # weight
        param_name = self.layer.name + ".weight"
        tensor = Params[0]
        shape = [
            conv.num_output,
            self.input_shape_dict[self.layer.bottom[0]][1] // conv.group,
            *kernel_shape,
        ]
        shape = [int(x) for x in shape]
        assert len(tensor.data) == np.multiply.reduce(
            shape
        ), f"expected kernel size to be {np.multiply.reduce(shape)}, got {len(tensor.data)}"
        param_tensor_info = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, shape
        )
        param_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, shape, tensor.data
        )
        weight_names.append(param_name)
        weight_dict[param_name] = (param_tensor_info, param_tensor)

        # bias
        if self.layer.convolution3d_param.bias_term:
            param_name = self.layer.name + ".bias"
            tensor = Params[1]
            param_tensor_info = helper.make_tensor_value_info(
                param_name, TensorProto.FLOAT, list(tensor.shape.dim)
            )
            param_tensor = helper.make_tensor(
                param_name, TensorProto.FLOAT, list(tensor.shape.dim), tensor.data
            )
            weight_names.append(param_name)
            weight_dict[param_name] = (param_tensor_info, param_tensor)

        return weight_dict, weight_names


class MatMul(BaseNode):

    # class MatMulMode(Enum):
    #     NN = 1
    #     NT = 2
    #     TN = 3
    #     TT = 4

    def __init__(self, layer, input_shape_dict):
        super(MatMul, self).__init__(layer, input_shape_dict)

    def get_type(self):
        return "MatMul"

    def get_onnx_op(self):
        return Op.gen_op(self.get_type(), self.get_type() + "::" + self.layer.name)

    def get_attr(self):
        settings = {}
        param = self.layer.matmul_param
        if param.HasField("mode"):
            settings["mode"] = param.mode
        return settings


class Abs(BaseNode):
    def get_type(self):
        return "Abs"

    def get_attr(self):
        settings = {}
        return settings


class Reciprocal(BaseNode):
    def get_type(self):
        return "Reciprocal"

    def get_attr(self):
        settings = {}
        return settings
