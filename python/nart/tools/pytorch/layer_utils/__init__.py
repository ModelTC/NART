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

from .convolution import Convolution
from .correlation import Correlation
from .holeconvolution import HoleConvolution
from .deformconvolution import DeformableConvolution
from .pooling import Pooling
from .relu import ReLU
from .inner_product import InnerProduct
from .reshape import Reshape
from .batchnorm import BatchNorm
from .bn import BN
from .scale import Scale
from .eltwise import Eltwise
from .prelu import PReLU
from .interp import Interp
from .nninterp import NNInterp
from .slice import Slice
from .concat import Concat
from .sigmoid import Sigmoid
from .softmax import Softmax
from .roipool import ROIPool
from .roialignpooling import ROIAlignPooling
from .podroialignpooling import PODROIAlignPooling
from .psroipooling import PSROIPooling
from .psroimaskpooling import PSROIMaskPooling
from .dropout import Dropout
from .transpose import Transpose
from .relu6 import ReLU6
from .clip import Clip
from .deconvolution import Deconvolution
from .shuffle_channel import ShuffleChannel
from .tanh import TanH
from .lstm_unit import LSTMUnit
from .sllstm import SLLSTM
from .slgrnn import SLGRNN
from .reverse import Reverse
from .groupnorm import GroupNorm
from .reduce import Reduce
from .axpy import Axpy
from .correlation1d import Correlation1D
from .dummy_data import DummyData
from .unknown import Unknown
from ..onnx_utils import onnx_pb2
from .pad import Pad
import warnings

__all__ = ["make_layers"]


def make_layers(node, index, network):
    if node.op_type == "Conv":
        attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))
        if (
            attributes["dilations"].ints[0] == 1
            and attributes["dilations"].ints[1] == 1
        ):
            convolution_layer = Convolution(node, index, network)
            return [convolution_layer]
        else:
            holeconvolution_layer = HoleConvolution(node, index, network)
            return [holeconvolution_layer]
    elif node.op_type == "Corr":
        correlation_layer = Correlation(node, index, network)
        return [correlation_layer]
    elif node.op_type == "Correlation1D":
        correlation1d_layer = Correlation1D(node, index, network)
        return [correlation1d_layer]
    elif node.op_type == "DeformConv":
        attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))
        deform_convolution_layer = DeformableConvolution(node, index, network)
        return [deform_convolution_layer]
    elif node.op_type in [
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "GlobalMaxPool",
    ]:
        pooling_layer = Pooling(node, index, network)
        return [pooling_layer]
    elif node.op_type in ["Relu", "LeakyRelu"]:
        relu_layer = ReLU(node, index, network)
        return [relu_layer]
    elif node.op_type == "Gemm":
        inner_product_layer = InnerProduct(node, index, network)
        return [inner_product_layer]
    elif node.op_type in ["Reshape", "Flatten", "Squeeze", "Unsqueeze"]:
        reshape_layer = Reshape(node, index, network)
        return [reshape_layer]
    elif node.op_type == "BatchNormalization":
        # old
        """
        affine = True if node.input[1] and node.input[2] else False
        if affine:
            batchnorm_top = 'BatchNorm{}'.format(index)
            scale_top = node.output[0]
            node.output[0] = batchnorm_top
            batchnorm_layer = BatchNorm(node, index, network)
            node.input[0] = batchnorm_top
            node.output[0] = scale_top
            scale_layer = Scale(node, index + '_s', network)
            return [batchnorm_layer, scale_layer]
        else:
            batchnorm_layer = BatchNorm(node, index, network)
            return [batchnorm_layer]
        """
        affine = True if node.input[1] and node.input[2] else False
        if affine is False:
            warnings.warn("BN has no weight and bias")
        bn_layer = BN(node, index, network)
        return [bn_layer]
    elif node.op_type in ["Add", "Sub", "Mul", "Max"]:
        a = network.blobshape[node.input[0]]
        b = network.blobshape[node.input[1]]
        la = len(a)
        lb = len(b)
        same = 0
        assert la == lb
        for i in range(la):
            if a[i] == b[i]:
                same += 1
        if same == la:
            eltwise_layer = Eltwise(node, index, network)
            return [eltwise_layer]
        elif a[0] == b[0] and a[1] == b[1] and node.op_type == "Mul":
            if a[2] == 1 and a[3] == 1:
                tmp = node.input[0]
                node.input[0] = node.input[1]
                node.input[1] = tmp
            elif b[2] == 1 and b[3] == 1:
                pass
            else:
                raise NotImplementedError(node.op_type)
            node.op_type = "View"
            squeeze_top = "View{}".format(index)
            scale_top = node.output[0]
            scale_bottom0 = node.input[0]
            node.input[:] = node.input[1:2]
            node.output[0] = squeeze_top
            axes = onnx_pb2.AttributeProto()
            axes.name = "dims"
            axes.ints.append(0)
            axes.ints.append(-1)
            axes.type = 7
            node.attribute.extend([axes])
            reshape_layer = Reshape(node, index, network)
            node.op_type = "Mul"
            node.output[0] = scale_top
            node.input[0] = scale_bottom0
            node.input.append(squeeze_top)
            scale_layer = Scale(node, str(index) + "_s", network)
            return [reshape_layer, scale_layer]
        else:
            raise NotImplementedError(
                node.op_type,
                f"Shape Mismatch between node `{node.input[0]}` and `{node.input[1]}`, got {a} - {b}",
            )
    elif node.op_type == "PRelu":
        prelu_layer = PReLU(node, index, network)
        return [prelu_layer]
    elif node.op_type == "Upsample":
        attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))
        mode = attributes["mode"].s.decode("utf-8")
        if mode == "bilinear" or mode == "linear":
            interp_layer = Interp(node, index, network)
            return [interp_layer]
        elif mode == "nearest":
            nninterp_layer = NNInterp(node, index, network)
            return [nninterp_layer]
        else:
            raise NotImplementedError(node.op_type, mode)
    elif node.op_type in ["Split", "Slice"]:
        slice_layer = Slice(node, index, network)
        return [slice_layer]
    elif node.op_type == "Concat":
        concat_layer = Concat(node, index, network)
        return [concat_layer]
    elif node.op_type == "Sigmoid":
        sigmoid_layer = Sigmoid(node, index, network)
        return [sigmoid_layer]
    elif node.op_type == "Softmax":
        softmax_layer = Softmax(node, index, network)
        return [softmax_layer]
    elif node.op_type == "RoiAlign":
        roialignpooling_layer = ROIAlignPooling(node, index, network)
        return [roialignpooling_layer]
    elif node.op_type == "PODRoiAlign":
        podroialignpooling_layer = PODROIAlignPooling(node, index, network)
        return [podroialignpooling_layer]
    elif node.op_type == "RoiPool":
        roipool_layer = ROIPool(node, index, network)
        return [roipool_layer]
    elif node.op_type == "PSRoiPool":
        psroipooling_layer = PSROIPooling(node, index, network)
        return [psroipooling_layer]
    elif node.op_type == "PSRoiMaskPool":
        psroimaskpooling_layer = PSROIMaskPooling(node, index, network)
        return [psroimaskpooling_layer]
    elif node.op_type == "Dropout":
        dropout_layer = Dropout(node, index, network)
        return [dropout_layer]
    elif node.op_type == "Transpose":
        transpose_layer = Transpose(node, index, network)
        return [transpose_layer]
    elif node.op_type == "Hardtanh":
        attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))
        if int(attributes["min_val"].f) == 0 and int(attributes["max_val"].f) == 6:
            relu6_layer = ReLU6(node, index, network)
            return [relu6_layer]
        else:
            clip_layer = Clip(node, index, network)
            return [clip_layer]
    elif node.op_type == "Clip":
        attributes = dict(zip([attr.name for attr in node.attribute], node.attribute))
        if int(attributes["min"].f) == 0 and int(attributes["max"].f) == 6:
            relu6_layer = ReLU6(node, index, network)
            return [relu6_layer]
        else:
            clip_layer = Clip(node, index, network)
            return [clip_layer]
    elif node.op_type == "ConvTranspose":
        deconvolution_layer = Deconvolution(node, index, network)
        return [deconvolution_layer]
    elif node.op_type == "ShuffleChannel":
        shuffle_channel_layer = ShuffleChannel(node, index, network)
        return [shuffle_channel_layer]
    elif node.op_type == "Tanh":
        tanh_layer = TanH(node, index, network)
        return [tanh_layer]
    elif node.op_type == "LSTMCell":
        lstm_unit_layer = LSTMUnit(node, index, network)
        return [lstm_unit_layer]
    elif node.op_type == "LSTM":
        sllstm_layer = SLLSTM(node, index, network)
        return [sllstm_layer]
    elif node.op_type == "GRU":
        slgrnn_layer = SLGRNN(node, index, network)
        return [slgrnn_layer]
    elif node.op_type == "Flip":
        reverse_layer = Reverse(node, index, network)
        return [reverse_layer]
    elif node.op_type == "GroupNorm":
        affine = True if len(node.input) > 1 else False
        if affine:
            groupnorm_top = "GroupNorm{}".format(index)
            scale_top = node.output[0]
            node.output[0] = groupnorm_top
            groupnorm_layer = GroupNorm(node, index, network)
            node.input[0] = groupnorm_top
            node.output[0] = scale_top
            scale_layer = Scale(node, index + "_s", network)
            return [groupnorm_layer, scale_layer]
        else:
            groupnorm_layer = GroupNorm(node, index, network)
            return [groupnorm_layer]
    elif node.op_type in ["ReduceMean", "ReduceSum"]:
        reduce_layer = Reduce(node, index, network)
        return [reduce_layer]
    elif node.op_type == "Axpy":
        axpy_layer = Axpy(node, index, network)
        return [axpy_layer]
    elif node.op_type == "Constant":
        dummy_data_layer = DummyData(node, index, network)
        warnings.warn(
            "[FATAL WARNING] DummyData layer {} occured. ".format(
                dummy_data_layer.params.name
            )
            + "Please manually modify the prototxt file."
        )
        return [dummy_data_layer]
    elif node.op_type == "Pad":
        pad_layer = Pad(node, index, network)
        return [pad_layer]
    else:
        # raise NotImplementedError(node.op_type)
        unknown_layer = Unknown(node, index, network)
        message = "[FATAL WARNING] Unknown layer {} occured from {}. Please manually modify the prototxt file.\n".format(
            unknown_layer.params.name, node.op_type
        )
        warnings.warn(message)
        return [unknown_layer]
