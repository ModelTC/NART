#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# from nart.tools.proto import caffe_pb2 as caffe_pb2
from nart.tools.caffe.utils.graph import readNetStructure
import google.protobuf.text_format
import numpy as np
import argparse
import csv
import os
import math


class CaffeLayer(object):
    __name = None
    __param = 0
    __activation = 0
    __bottom = None
    __top = None
    __topShape = None
    __blobShape = None
    __layerParam = None
    __comp = 0
    __add = 0
    __div = 0
    __macc = 0
    __exp = 0
    __memacc = 0

    def __init__(self, layerParameter):
        self.__name = layerParameter.name
        self.__bottom = layerParameter.bottom
        self.__top = layerParameter.top
        self.__topShape = np.zeros((len(layerParameter.top), 4), dtype=np.int32)
        self.__blobShape = []

    @property
    def name(self):
        return self.__name

    @property
    def layerParam(self):
        return self.__layerParam

    @layerParam.setter
    def layerParam(self, parameter):
        self.__layerParam = parameter

    @property
    def bottom(self):
        return self.__bottom

    @property
    def top(self):
        return self.__top

    @property
    def topShape(self):
        return self.__topShape

    @topShape.setter
    def topShape(self, shape):
        self.__topShape = shape

    @property
    def blobShape(self):
        return self.__blobShape

    @property
    def comp(self):
        # return self.__computation[0]
        return self.__comp

    @comp.setter
    def comp(self, computation):
        self.__comp = computation

    @property
    def add(self):
        # return self.__computation[1]
        return self.__add

    @add.setter
    def add(self, computation):
        self.__add = computation

    @property
    def div(self):
        # return self.__computation[2]
        return self.__div

    @div.setter
    def div(self, computation):
        self.__div = computation

    @property
    def macc(self):
        # return self.__computation[3]
        return self.__macc

    @macc.setter
    def macc(self, computation):
        self.__macc = computation

    @property
    def exp(self):
        # return self.__computation[4]
        return self.__exp

    @exp.setter
    def exp(self, computation):
        self.__exp = computation

    @property
    def param(self):
        return self.__param

    @param.setter
    def param(self, usage):
        self.__param = usage

    @property
    def activation(self):
        return self.__activation

    @activation.setter
    def activation(self, usage):
        self.__activation = usage

    @property
    def memacc(self):
        return self.__memacc

    @memacc.setter
    def memacc(self, usage):
        self.__memacc = usage


class DefaultLayer(CaffeLayer):
    def __init__(self, layerParameter):
        pass

    def infer(self, prev):
        pass

    def setup(self):
        pass


class InputLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(InputLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.input_param

    def infer(self, prev):
        for i in range(len(self.layerParam.shape)):
            minl = min(len(self.topShape[i]), len(self.layerParam.shape[i].dim))
            self.topShape[i][:] = 1
            for x in range(minl):
                self.topShape[i][x] = self.layerParam.shape[i].dim[x]
        self.param = 0
        for shape in self.topShape:
            self.activation += shape.cumprod()[-1]

    def setup(self):
        pass


class ConvolutionLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ConvolutionLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.convolution_param

    def infer(self, prev):
        # print(prev[0])
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]

        # infer the output shape
        assert self.layerParam.HasField("pad_h") == self.layerParam.HasField("pad_w")
        if self.layerParam.HasField("pad_h"):
            pad_h = self.layerParam.pad_h
            pad_w = self.layerParam.pad_w
        else:
            pad_h = self.layerParam.pad
            pad_w = self.layerParam.pad
        assert self.layerParam.HasField("kernel_h") == self.layerParam.HasField(
            "kernel_w"
        )
        if self.layerParam.HasField("kernel_h"):
            kernel_h = self.layerParam.kernel_h
            kernel_w = self.layerParam.kernel_w
        else:
            kernel_w = self.layerParam.kernel_size
            kernel_h = self.layerParam.kernel_size
        assert self.layerParam.HasField("stride_h") == self.layerParam.HasField(
            "stride_w"
        )
        if self.layerParam.HasField("stride_h"):
            stride_h = self.layerParam.stride_h
            stride_w = self.layerParam.stride_w
        else:
            stride_w = self.layerParam.stride
            stride_h = self.layerParam.stride
        hole_h = (
            self.layerParam.hole_h
            if self.layerParam.HasField("hole_h")
            else self.layerParam.hole
        )
        hole_w = (
            self.layerParam.hole_w
            if self.layerParam.HasField("hole_w")
            else self.layerParam.hole
        )

        oc = self.layerParam.num_output
        kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1)
        kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1)
        oh = math.floor((ih + 2 * pad_h - kernel_h_eff) / stride_h + 1)
        ow = math.floor((iw + 2 * pad_w - kernel_w_eff) / stride_w + 1)
        on = bs * oc * oh * ow
        # print("%d * %d * %d * %d = %d" % (bs, oc, oh, ow, bs * oc * oh * ow))
        self.topShape[0][:] = [bs, oc, oh, ow]

        # calculate the computation
        if self.layerParam.HasField("group"):
            kernel_c = ic / self.layerParam.group
        else:
            kernel_c = ic
        # macc = (bs * oc * oh * ow) * (kernel_c * kernel_h * kernel_w)
        # self.computation[3] = on * (kernel_c * kernel_h * kernel_w)
        self.macc = on * (kernel_c * kernel_h * kernel_w)
        # print("conv outputnumber: %d, kc: %d, kh: %d, kw: %d" % (on, kernel_c, kernel_h, kernel_w))
        # print("conv macc computation: %d" % self.macc)
        # calcualte the memory usage
        self.param = oc * kernel_c * kernel_h * kernel_w
        self.blobShape.append([oc, kernel_c, kernel_h, kernel_w])
        if self.layerParam.bias_term:
            self.param += oc
            self.blobShape.append([oc])
        self.activation = on

    def setup(self, layer):
        # initial weights param
        # print(self.blobShape[1])
        number = 1
        for i in self.blobShape[0]:
            number *= i
        weights = layer.blobs.add()
        weights.shape.dim.extend(self.blobShape[0])
        # weights.data.extend(np.random.randn(number))
        weights.data.extend(im.guassianWithTruncate(number))
        # initial bias param
        if self.layerParam.bias_term:
            bias = layer.blobs.add()
            bias.shape.dim.extend(self.blobShape[1])
            bias.data.extend(np.zeros(self.blobShape[1][0], dtype=np.int32))


class Convolution3dLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(Convolution3dLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.convolution3d_param

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        bs = prev[0][0]
        ic = prev[0][1]
        idt = prev[0][2]
        ih = prev[0][3]
        iw = prev[0][4]

        # infer the output shape
        assert self.layerParam.HasField("pad_h") == self.layerParam.HasField("pad_w")
        if self.layerParam.HasField("pad_h") and self.layerParam.HasField("pad_w"):
            pad_d = self.layerParam.pad_d
            pad_h = self.layerParam.pad_h
            pad_w = self.layerParam.pad_w
        else:
            pad_d = self.layerParam.pad
            pad_h = self.layerParam.pad
            pad_w = self.layerParam.pad
        assert self.layerParam.HasField("kernel_h") == self.layerParam.HasField(
            "kernel_w"
        )
        if self.layerParam.HasField("kernel_h") and self.layerParam.HasField(
            "kernel_w"
        ):
            kernel_d = self.layerParam.kernel_d
            kernel_h = self.layerParam.kernel_h
            kernel_w = self.layerParam.kernel_w
        else:
            kernel_d = self.layerParam.kernel_size
            kernel_w = self.layerParam.kernel_size
            kernel_h = self.layerParam.kernel_size
        assert self.layerParam.HasField("stride_h") == self.layerParam.HasField(
            "stride_w"
        )
        if self.layerParam.HasField("stride_h") and self.layerParam.HasField(
            "stride_w"
        ):
            stride_d = self.layerParam.stride_d
            stride_h = self.layerParam.stride_h
            stride_w = self.layerParam.stride_w
        else:
            stride_d = self.layerParam.stride
            stride_w = self.layerParam.stride
            stride_h = self.layerParam.stride
        hole_d = (
            self.layerParam.hole_d
            if self.layerParam.HasField("hole_d")
            else self.layerParam.hole
        )
        hole_h = (
            self.layerParam.hole_h
            if self.layerParam.HasField("hole_h")
            else self.layerParam.hole
        )
        hole_w = (
            self.layerParam.hole_w
            if self.layerParam.HasField("hole_w")
            else self.layerParam.hole
        )

        oc = self.layerParam.num_output
        kernel_d_eff = kernel_d + (kernel_d - 1) * (hole_d - 1)
        kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1)
        kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1)
        od = math.floor((idt + 2 * pad_d - kernel_d_eff) / stride_d + 1)
        oh = math.floor((ih + 2 * pad_h - kernel_h_eff) / stride_h + 1)
        ow = math.floor((iw + 2 * pad_w - kernel_w_eff) / stride_w + 1)
        on = bs * oc * od * oh * ow
        # print("%d * %d * %d * %d = %d" % (bs, oc, oh, ow, bs * oc * oh * ow))
        self.topShape[0][:] = [bs, oc, od, oh, ow]

        # calculate the computation
        if self.layerParam.HasField("group"):
            kernel_c = ic / self.layerParam.group
        else:
            kernel_c = ic
        # macc = (bs * oc * oh * ow) * (kernel_c * kernel_h * kernel_w)
        # self.computation[3] = on * (kernel_c * kernel_h * kernel_w)
        self.macc = on * (kernel_c * kernel_d * kernel_h * kernel_w)
        # print("conv outputnumber: %d, kc: %d, kh: %d, kw: %d" % (on, kernel_c, kernel_h, kernel_w))
        # print("conv macc computation: %d" % self.macc)
        # calcualte the memory usage
        self.param = oc * kernel_c * kernel_d * kernel_h * kernel_w
        self.blobShape.append([oc, kernel_c, kernel_d, kernel_h, kernel_w])
        if self.layerParam.bias_term:
            self.param += oc
            self.blobShape.append([oc])
        self.activation = on

    def setup(self, layer):
        # initial weights param
        # print(self.blobShape[1])
        number = 1
        for i in self.blobShape[0]:
            number *= i
        weights = layer.blobs.add()
        weights.shape.dim.extend(self.blobShape[0])
        # weights.data.extend(np.random.randn(number))
        weights.data.extend(im.guassianWithTruncate(number))
        # initial bias param
        if self.layerParam.bias_term:
            bias = layer.blobs.add()
            bias.shape.dim.extend(self.blobShape[1])
            bias.data.extend(np.zeros(self.blobShape[1][0], dtype=np.int32))


class DeconvolutionLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(DeconvolutionLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.convolution_param

    def infer(self, prev):
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]

        # infer the output shape
        assert self.layerParam.HasField("pad_h") == self.layerParam.HasField("pad_w")
        if self.layerParam.HasField("pad_h"):
            pad_h = self.layerParam.pad_h
            pad_w = self.layerParam.pad_w
        else:
            pad_w = self.layerParam.pad
            pad_h = self.layerParam.pad
        assert self.layerParam.HasField("kernel_h") == self.layerParam.HasField(
            "kernel_w"
        )
        if self.layerParam.HasField("kernel_h"):
            kernel_w = self.layerParam.kernel_w
            kernel_h = self.layerParam.kernel_h
        else:
            kernel_w = self.layerParam.kernel_size
            kernel_h = self.layerParam.kernel_size
        assert self.layerParam.HasField("stride_h") == self.layerParam.HasField(
            "stride_w"
        )
        if self.layerParam.HasField("stride_h"):
            stride_w = self.layerParam.stride_w
            stride_h = self.layerParam.stride_h
        else:
            stride_w = self.layerParam.stride
            stride_h = self.layerParam.stride

        oc = self.layerParam.num_output
        oh = stride_h * (ih - 1) + kernel_h - 2 * pad_h
        ow = stride_w * (iw - 1) + kernel_w - 2 * pad_w
        self.topShape[0][:] = [bs, oc, oh, ow]

        # calculate the computation
        self.macc = bs * ic * ih * iw * oc * kernel_h * kernel_w
        # print("macc computation: %d" % self.macc)

        # calculate the memory usage
        self.param = kernel_h * kernel_w * ic * oc
        self.blobShape.append([oc, ic, kernel_h, kernel_w])
        if self.layerParam.bias_term:
            self.param += oc
            self.blobShape.append([oc])
        self.activation = bs * oc * oh * ow
        self.memacc = bs * oc * oh * ow + self.param + bs * ic * ih * iw

    def setup(self, layer):
        # initial weights param
        number = 1
        for i in self.blobShape[0]:
            number *= i
        weights = layer.blobs.add()
        weights.shape.dim.extend(self.blobShape[0])
        # weights.data.extend(np.random.randn(number))
        weights.data.extend(im.guassianWithTruncate(number))
        # initial bias param
        if self.layerParam.bias_term:
            bias = layer.blobs.add()
            bias.shape.dim.extend(self.blobShape[1])
            bias.data.extend(np.zeros(self.blobShape[1][0], dtype=np.int32))


class CorrelationLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(CorrelationLayer, self).__init__(layerParameter)
        # correlation has no param

    def infer(self, prev):
        ob = prev[0][0] * prev[1][0]
        oc = prev[0][1]
        oh = prev[1][2] - prev[0][2] + 1
        ow = prev[1][3] - prev[0][3] + 1
        self.topShape[0][:] = [ob, oc, oh, ow]
        on = ob * oc * oh * ow

        # calculate the computation
        self.macc = on * prev[1][0] * prev[0][2] * prev[0][3]
        # correlation has no param
        self.param = 0
        self.activation = on

    def setup(self, layer):
        # correlation has no param
        pass


class Correlation1DLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(Correlation1DLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.correlation_param

    def infer(self, prev):
        ob = prev[0][0]
        oc = self.layerParam.max_displacement + 1
        oh = prev[1][2]
        ow = prev[1][3]
        self.topShape[0][:] = [ob, oc, oh, ow]
        on = ob * oc * oh * ow

        # calculate the computation
        self.macc = on * prev[1][0] * prev[0][2] * prev[0][3]
        # correlation has no param
        self.param = 0
        self.activation = on

    def setup(self, layer):
        # correlation has no param
        pass


class PoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.pooling_param

    def infer(self, prev):
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]

        # infer the output shape
        if self.layerParam.global_pooling:
            pad_h = 0
            pad_w = 0
            kernel_h = ih
            kernel_w = iw
            stride_h = 1
            stride_w = 1
        else:
            assert self.layerParam.HasField("pad_h") == self.layerParam.HasField(
                "pad_w"
            )
            if self.layerParam.HasField("pad_h"):
                pad_h = self.layerParam.pad_h
                pad_w = self.layerParam.pad_w
            else:
                pad_h = self.layerParam.pad
                pad_w = self.layerParam.pad
            assert self.layerParam.HasField("kernel_h") == self.layerParam.HasField(
                "kernel_w"
            )
            if self.layerParam.HasField("kernel_h"):
                kernel_h = self.layerParam.kernel_h
                kernel_w = self.layerParam.kernel_w
            else:
                kernel_h = self.layerParam.kernel_size
                kernel_w = self.layerParam.kernel_size
            assert self.layerParam.HasField("stride_h") == self.layerParam.HasField(
                "stride_w"
            )
            if self.layerParam.HasField("stride_h"):
                stride_h = self.layerParam.stride_h
                stride_w = self.layerParam.stride_w
            else:
                stride_h = self.layerParam.stride
                stride_w = self.layerParam.stride
        # print("ih, ph, kh, sh: %f %f %f %f" % (ih, pad_h, kernel_h, stride_h))
        oc = ic
        if self.layerParam.ceil_mode:
            oh = math.ceil((float(ih) + 2 * pad_h - kernel_h) / stride_h + 1)
            ow = math.ceil((float(iw) + 2 * pad_w - kernel_w) / stride_w + 1)
        else:
            oh = math.floor((float(ih) + 2 * pad_h - kernel_h) / stride_h + 1)
            ow = math.floor((float(iw) + 2 * pad_w - kernel_w) / stride_w + 1)
        on = bs * oc * oh * ow
        self.topShape[0][:] = [bs, oc, oh, ow]
        # print("bs, oc, oh, ow: %f %f %f %f" % (bs, oc, oh, ow))
        # print("self.topShape: ", self.topShape[0])

        # calculate the computation
        if self.layerParam.pool == 0:
            # max pooling
            self.comp = on * (kernel_h * kernel_w)
        elif self.layerParam.pool == 1:
            # ave pooling
            self.add = on * (kernel_h * kernel_w - 1)
            self.div = on * 1
        else:
            # stochastic pooling
            pass

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class Pooling3dLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(Pooling3dLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.pooling3d_param

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        bs = prev[0][0]
        ic = prev[0][1]
        idt = prev[0][2]
        ih = prev[0][3]
        iw = prev[0][4]

        # infer the output shape
        if self.layerParam.global_pooling:
            pad_d = 0
            pad_h = 0
            pad_w = 0
            kernel_d = idt
            kernel_h = ih
            kernel_w = iw
            stride_d = 1
            stride_h = 1
            stride_w = 1
        else:
            assert self.layerParam.HasField("pad_h") == self.layerParam.HasField(
                "pad_w"
            )
            if self.layerParam.HasField("pad_h"):
                pad_d = self.layerParam.pad_d
                pad_h = self.layerParam.pad_h
                pad_w = self.layerParam.pad_w
            else:
                pad_d = self.layerParam.pad
                pad_h = self.layerParam.pad
                pad_w = self.layerParam.pad
            assert self.layerParam.HasField("kernel_h") == self.layerParam.HasField(
                "kernel_w"
            )
            if self.layerParam.HasField("kernel_h"):
                kernel_d = self.layerParam.kernel_d
                kernel_h = self.layerParam.kernel_h
                kernel_w = self.layerParam.kernel_w
            else:
                kernel_d = self.layerParam.kernel_size
                kernel_h = self.layerParam.kernel_size
                kernel_w = self.layerParam.kernel_size
            assert self.layerParam.HasField("stride_h") == self.layerParam.HasField(
                "stride_w"
            )
            if self.layerParam.HasField("stride_h"):
                stride_d = self.layerParam.stride_d
                stride_h = self.layerParam.stride_h
                stride_w = self.layerParam.stride_w
            else:
                stride_d = self.layerParam.stride
                stride_h = self.layerParam.stride
                stride_w = self.layerParam.stride
        # print("ih, ph, kh, sh: %f %f %f %f" % (ih, pad_h, kernel_h, stride_h))
        oc = ic
        if self.layerParam.ceil_mode:
            od = math.ceil((float(idt) + 2 * pad_d - kernel_d) / stride_d + 1)
            oh = math.ceil((float(ih) + 2 * pad_h - kernel_h) / stride_h + 1)
            ow = math.ceil((float(iw) + 2 * pad_w - kernel_w) / stride_w + 1)
        else:
            od = math.floor((float(idt) + 2 * pad_d - kernel_d) / stride_d + 1)
            oh = math.floor((float(ih) + 2 * pad_h - kernel_h) / stride_h + 1)
            ow = math.floor((float(iw) + 2 * pad_w - kernel_w) / stride_w + 1)
        on = bs * oc * od * oh * ow
        self.topShape[0][:] = [bs, oc, od, oh, ow]
        # print("bs, oc, oh, ow: %f %f %f %f" % (bs, oc, oh, ow))
        # print("self.topShape: ", self.topShape[0])

        # calculate the computation
        if self.layerParam.pool == 0:
            # max pooling
            self.comp = on * (kernel_d * kernel_h * kernel_w)
        elif self.layerParam.pool == 1:
            # ave pooling
            self.add = on * (kernel_d * kernel_h * kernel_w - 1)
            self.div = on * 1
        else:
            # stochastic pooling
            pass

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class EltwiseLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(EltwiseLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.eltwise_param

    def infer(self, prev):
        bs = prev[0][0]
        channels = np.multiply.reduce(prev[0][1:])
        # infer the output shape
        self.topShape = np.reshape(prev[0], [1, len(prev[0])]).copy()

        # calculate the computation
        if self.layerParam.operation == 0:
            # prod operation
            # self.computation[2] = (len(prev) - 1) * (ic * ih * iw)
            self.div = (len(prev) - 1) * channels
        elif self.layerParam.operation == 1:
            # sum operation
            # self.computation[1] = (len(prev) - 1) * (ic * ih * iw)
            # self.computation[2] = self.computation[1]
            self.add = (len(prev) - 1) * channels
            self.div = self.add
            if len(self.layerParam.coeff) == len(prev) + 1:
                # self.computation[1] += ic * ih * iw
                self.add += channels
        else:
            # max operation
            # self.computation[0] = (len(prev) - 1) * (ic * ih * iw)
            self.comp = (len(prev) - 1) * channels

        self.param = 0
        self.activation = bs * channels

    def setup(self, layer):
        pass


class ReductionLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ReductionLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.reduction_param

    def infer(self, prev):
        # infer the output shape
        axis = self.layerParam.axis
        if axis < 0:
            axis += len(prev[0])

        # reduction was done on ALL tailing axes.
        if axis > 0:
            self.topShape = np.zeros([1, axis], np.int32)
            self.topShape[0][:] = prev[0][0:axis]
        else:
            self.topShape = np.array([[1]], np.int32)

        self.add = prev[0].prod()
        self.param = 0
        self.activation = self.topShape[0].prod()

    def setup(self, layer):
        pass


class ReduceLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ReduceLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.reduce_param

    def infer(self, prev):
        # infer the output shape
        axis = self.layerParam.axis
        if axis < 0:
            axis += len(prev[0])
        self.topShape = np.zeros((1, len(prev[0])), dtype=np.int32)
        self.topShape[0][:] = prev[0]
        self.topShape[0][axis] = 1

        self.add = prev[0].sum()
        self.param = 0
        self.activation = self.topShape[0].sum()

    def setup(self, layer):
        pass


class ReluLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ReluLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.relu_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], dtype=np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()
        # print("bs: %d, ic: %d, ih: %d, iw: %d" % (bs, ic, ih, iw))

        # calculate the computation
        self.comp = on
        # print(self.computation[0])

        # calculate the memory usage
        self.param = 0
        self.activation = on
        # print("relu activation: %d" % self.activation)
        # print("relu param: %d" % self.param)

    def setup(self, layer):
        pass


class Relu3dLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(Relu3dLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.relu3d_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], dtype=np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        self.comp = on

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class PReluLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PReluLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.prelu_param

    def infer(self, prev):
        ic = prev[0][1]
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # self.computation[0] = 2 * on
        # self.computation[1] = on
        # self.computation[2] = on
        self.comp = 2 * on
        self.add = on
        self.div = on

        # calculate the memory usage
        self.param = ic
        self.blobShape.append([ic])
        self.activation = on

    def setup(self, layer):
        # initial the prob param
        prob = layer.blobs.add()
        prob.shape.dim.extend(self.blobShape[0])
        prob.data.extend([0.25] * self.blobShape[0][0])


class TanhLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(TanhLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.tanh_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # self.computation[1] = 2 * on
        # self.computation[2] = on
        # self.computation[4] = 2 * on
        self.add = 2 * on
        self.div = on
        self.exp = 2 * on

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class ScaleLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ScaleLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.scale_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        self.macc = on
        if self.layerParam.bias_term:
            self.add = on

        # calculate the memory usage
        if len(prev) > 1:
            self.param = 0
        else:
            # self.param = prev[0][self.layerParam.axis]
            if self.layerParam.num_axes == 0:
                self.param = 1
            elif self.layerParam.num_axes == -1:
                self.param = prev[0][self.layerParam.axis : len(prev[0])].cumprod()[-1]
            else:
                self.param = prev[0][
                    self.layerParam.axis : self.layerParam.axis
                    + self.layerParam.num_axes
                ].cumprod()[-1]
        self.blobShape.append([self.param])
        self.activation = on

    def setup(self, layer):
        # initial the scale param
        if self.param > 0:
            scale = layer.blobs.add()
            scale.shape.dim.extend([self.blobShape[0]])
            # scale.data.extend(np.random.randn(self.blobShape[0][0]))
            scale.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))


class BatchnormLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(BatchnormLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.batch_norm_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # if(not self.layerParam.use_global_stats):
        #     # calculate the mean
        #     self.add = on - ic
        #     self.div = ic
        #     # calculate the variance
        #     self.add += on + (on - ic)
        #     self.div += on + ic
        #     self.exp = ic
        self.add += on
        self.div += on

        # calculate memory usage
        self.param = 2 * prev[0][1]
        self.blobShape.append([self.param])
        self.blobShape.append([self.param])
        self.activation = on

    def setup(self, layer):
        # initial the mean
        mean = layer.blobs.add()
        mean.shape.dim.extend(self.blobShape[0])
        # mean.data.extend(np.random.randn(self.blobShape[0][0]))
        mean.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))
        # initial the variance
        variance = layer.blobs.add()
        variance.shape.dim.extend(self.blobShape[1])
        # variance.data.extend(np.random.randn(self.blobShape[1][0]))
        variance.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))


class Batchnorm3dLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(Batchnorm3dLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.batchnorm3d_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        self.add += on
        self.div += on

        # calculate memory usage
        self.param = 2 * prev[0][1]
        self.blobShape.append([self.param])
        self.blobShape.append([self.param])
        self.activation = on

    def setup(self, layer):
        # initial the mean
        mean = layer.blobs.add()
        mean.shape.dim.extend(self.blobShape[0])
        # mean.data.extend(np.random.randn(self.blobShape[0][0]))
        mean.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))
        # initial the variance
        variance = layer.blobs.add()
        variance.shape.dim.extend(self.blobShape[1])
        # variance.data.extend(np.random.randn(self.blobShape[1][0]))
        variance.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))


class BNLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(BNLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.bn_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # self.add = (on - ic) + on + (on - ic)
        # self.div = ic + on + ic
        # self.exp = ic
        self.add += 2 * on
        self.div += 2 * on

        # calculate the memory usage
        self.param = 4 * prev[0][1]
        for i in range(4):
            self.blobShape.append([self.param])
        self.activation = on

    def setup(self, layer):
        # initial the mean
        mean = layer.blobs.add()
        mean.shape.dim.extend(self.blobShape[0])
        # mean.data.extend(np.random.randn(self.blobShape[0][0]))
        mean.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))
        # initial the variance
        variance = layer.blobs.add()
        variance.shape.dim.extend(self.blobShape[1])
        # variance.data.extend(np.random.randn(self.blobShape[1][0]))
        variance.data.extend(im.guassianWithTruncate(self.blobShape[1][0]))
        # initial the scale
        scale = layer.blobs.add()
        scale.shape.dim.extend(self.blobShape[2])
        # scale.data.extend(np.random.randn(self.blobShape[2][0]))
        scale.data.extend(im.guassianWithTruncate(self.blobShape[2][0]))
        # initial the shift
        shift = layer.blobs.add()
        shift.shape.dim.extend(self.blobShape[3])
        # shift.data.extend(np.random.randn(self.blobShape[3][0]))
        shift.data.extend(im.guassianWithTruncate(self.blobShape[3][0]))


class SigmoidLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(SigmoidLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.sigmoid_param

    def infer(self, prev):
        # infer the output size
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        self.add = 2 * on
        self.div = on
        self.exp = on

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class CaffeExpLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(CaffeExpLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.exp_param

    def infer(self, prev):
        # infer the output size
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # self.add = on
        # self.exp = on
        # self.mul = on

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class HsigmoidLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(HsigmoidLayer, self).__init__(layerParameter)

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], dtype=np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()
        # print("bs: %d, ic: %d, ih: %d, iw: %d" % (bs, ic, ih, iw))

        # calculate the computation
        self.comp = on
        # print(self.computation[0])

        # calculate the memory usage
        self.param = 0
        self.activation = on
        # print("relu activation: %d" % self.activation)
        # print("relu param: %d" % self.param)

    def setup(self, layer):
        pass


class HSwishLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(HSwishLayer, self).__init__(layerParameter)

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], dtype=np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()
        # print("bs: %d, ic: %d, ih: %d, iw: %d" % (bs, ic, ih, iw))

        # calculate the computation
        self.comp = on
        # print(self.computation[0])

        # calculate the memory usage
        self.param = 0
        self.activation = on
        # print("relu activation: %d" % self.activation)
        # print("relu param: %d" % self.param)

    def setup(self, layer):
        pass


class SoftmaxLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(SoftmaxLayer, self).__init__(layerParameter)
        if layerParameter.type == "Softmax":
            self.layerParam = layerParameter.softmax_param
        elif layerParameter.type == "TopkSoftmax":
            self.layerParam = layerParameter.topksoftmax_param

    def infer(self, prev):
        # infer the output size
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # self.computation[0] = on
        self.comp = on
        # self.computation[1] = on
        self.add = on
        # self.computation[4] = on
        self.exp = on
        # self.computation[1] += on / prev[0][self.layerParam.axis]
        self.add += on / prev[0][self.layerParam.axis]
        # self.computation[3] = on
        self.div = on

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class LRNLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(LRNLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.lrn_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        if self.layerParam.norm_region == 0:
            local_area = self.layerParam.local_size * 1 * 1
        else:
            local_area = self.layerParam.local_size**2 * 1

        self.add = on * local_area
        self.div = on * ((3 * local_area) * local_area)
        self.exp = on

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class InnerproductLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(InnerproductLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.inner_product_param

    def infer(self, prev):

        # infer the output shape
        # self.topShape[0][:] = [bs, self.layerParam.num_output, 1, 1]
        # on = bs * ic * ih * iw
        axis = self.layerParam.axis
        obs = prev[0][0:axis].cumprod()[-1]
        ichw = prev[0][axis : len(prev[0])].cumprod()[-1]
        self.topShape = np.array([[obs, self.layerParam.num_output]], dtype=np.int32)
        on = obs * self.layerParam.num_output

        # calculate the computation
        # self.add = on * (ichw - 1)
        # self.div = on * ichw
        # if(self.layerParam.bias_term):
        #     self.add += on
        self.macc = on * ichw

        # calculate the memory usage
        self.param = ichw * self.layerParam.num_output
        self.blobShape.append([self.layerParam.num_output, ichw])
        if self.layerParam.bias_term:
            self.param += self.layerParam.num_output
            self.blobShape.append([self.layerParam.num_output])
        self.activation = on

    def setup(self, layer):
        # initial the weight param
        weights = layer.blobs.add()
        weights.shape.dim.extend(self.blobShape[0])
        # weights.data.extend(np.random.randn(self.blobShape[0][0]))
        weights.data.extend(im.guassianWithTruncate(self.blobShape[0][0]))
        # initail the bias param
        if self.layerParam.bias_term:
            bias = layer.blobs.add()
            bias.shape.dim.extend(self.blobShape[1])
            # bias.data.extend(np.random.randn(self.blobShape[1][0]))
            bias.data.extend(np.zeros(self.blobShape[1][0]))


class Heatmap2coordLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(Heatmap2coordLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.heatmap_param

    def infer(self, prev):
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]

        # infer the output shape
        self.topShape[0][:] = [bs, ic * 3, 1, 1]

        # calculate the computation
        self.comp = bs * ic * ih * iw
        self.add = 2 * bs * ic
        self.div = 4 * bs * ic

        # calculate the memory usage
        self.param = 0
        self.activation = self.topShape[0].cumprod()[-1]

    def setup(self, layer):
        pass


class InterpLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(InterpLayer, self).__init__(layerParameter)
        if layerParameter.HasField("nninterp_param"):
            self.layerParam = layerParameter.nninterp_param
        elif layerParameter.HasField("interp_param"):
            self.layerParam = layerParameter.interp_param
        elif layerParameter.HasField("nn_upsample_param"):
            self.layerParam = layerParameter.nn_upsample_param
        elif layerParameter.HasField("resize_param"):
            self.layerParam = layerParameter.resize_param

    def infer(self, prev):
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]

        # infer the ouput shape
        if self.layerParam is not None:
            all_fields = [field[0].name for field in self.layerParam.ListFields()]
        else:
            all_fields = []
        if "pad_beg" in all_fields:
            hineff = ih + self.layerParam.pad_beg + self.layerParam.pad_end
            wineff = iw + self.layerParam.pad_beg + self.layerParam.pad_end
        else:
            hineff = ih
            wineff = iw

        if "zoom_factor" in all_fields and self.layerParam.HasField("zoom_factor"):
            factor = self.layerParam.zoom_factor
            oh = hineff + (hineff - 1) * (factor - 1)
            ow = wineff + (wineff - 1) * (factor - 1)
        elif "shrink_factor" in all_fields and self.layerParam.HasField(
            "shrink_factor"
        ):
            factor = self.layerParam.shrink_factor
            oh = (hineff - 1) / factor + 1
            ow = (wineff - 1) / factor + 1
        elif (
            "height" in all_fields
            and self.layerParam.HasField("height")
            and self.layerParam.HasField("width")
        ):
            oh = self.layerParam.height
            ow = self.layerParam.width
        elif len(prev) == 2:
            oh = prev[1][2]
            ow = prev[1][3]
        elif "scale_factor" in all_fields and self.layerParam.HasField("scale_factor"):
            factor = self.layerParam.scale_factor
            oh = hineff * factor
            ow = wineff * factor
        elif "resize" in all_fields and self.layerParam.HasField("resize"):
            factor = self.layerParam.resize
            oh = hineff * factor
            ow = wineff * factor
        else:
            raise Exception("Illegal Interp Layer Parameter")
        self.topShape[0][:] = [bs, ic, oh, ow]
        on = bs * ic * oh * ow

        # calculate the computation
        self.add = on * (4 + 3)
        self.div = on * 4

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class DropoutLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(DropoutLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.dropout_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        on = prev[0].prod()

        # calculate the computation
        # self.computation[2] = 2 * on
        self.div = on * 2

        # calculate the memory usage
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class SliceLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(SliceLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.slice_param

    def infer(self, prev):

        self.topShape = np.zeros((len(self.topShape), len(prev[0])), np.int32)
        # infer the output shape
        axis = self.layerParam.axis
        start = 0
        end = 0
        for i in range(len(self.layerParam.slice_point)):
            end = self.layerParam.slice_point[i]
            self.topShape[i][:] = prev[0]
            self.topShape[i][axis] = end - start
            start = end
        end = prev[0][axis]
        self.topShape[i + 1][:] = prev[0]
        self.topShape[i + 1][axis] = end - start

        self.param = 0
        self.activation = prev[0].sum()

    def setup(self, layer):
        pass


class ConcatLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ConcatLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.concat_param

    def infer(self, prev):
        # infer the ouput shape
        axis = self.layerParam.axis
        concat = 0
        for bottom in prev:
            concat += bottom[axis]
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        self.topShape[0][axis] = concat

        self.param = 0
        self.activation = self.topShape[0].prod()

    def setup(self, layer):
        pass


class ReshapeLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ReshapeLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.reshape_param

    def infer(self, prev):
        # infer the output shape
        if self.layerParam.axis >= 0:
            axis = self.layerParam.axis
        else:
            axis = self.layerParam.axis + len(prev[0])
        axes = 0
        if self.layerParam.num_axes == -1:
            axes = len(prev[0]) - axis
        else:
            axes = self.layerParam.num_axes

        # left part
        res_shape = prev[0][:axis]

        # mid part
        sp = np.array(self.layerParam.shape.dim)
        test_negdim = np.array(sp)
        test_negdim[test_negdim > 0] = 0
        assert (
            test_negdim.sum() == 0 or test_negdim.sum() == -1
        ), f"Got invalid reshape_param for layer {self.name}: \n{self.layerParam}"

        for i in range(len(sp)):
            if sp[i] == 0:
                sp[i] = prev[0][i + axis]
        if any(sp[sp == -1]):
            sp[sp == -1] = np.array(prev[0]).prod() / np.prod(sp[sp > 0])
        res_shape = np.append(res_shape, sp)

        # right part
        res_shape = np.append(res_shape, prev[0][axis + axes :])

        self.topShape = res_shape.reshape(1, len(res_shape))

        self.param = 0
        self.activation = self.topShape[0].prod()

    def setup(self, layer):
        pass


class SLLSTMLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(SLLSTMLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.recurrent_param

    def infer(self, prev):
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]
        pass

    def setup(self, layer):
        pass


class PSROIPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PSROIPoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.psroi_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = self.layerParam.output_dim
        self.topShape[0][2] = self.layerParam.group_size
        self.topShape[0][3] = self.layerParam.group_size
        # calcualtion
        # unable to statistic add
        o_n = prev[1][0] * self.layerParam.output_dim * self.layerParam.group_size**2
        self.div = o_n
        # memory
        self.param = 0
        self.activation = o_n

    def setup(self, layer):
        # psroipooling has no param
        pass


class PSROIAlignPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PSROIAlignPoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.psroi_align_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = self.layerParam.output_dim
        self.topShape[0][2] = self.layerParam.group_size
        self.topShape[0][3] = self.layerParam.group_size
        # calculation
        o_n = prev[1][0] * self.layerParam.output_dim * self.layerParam.group_size**2
        self.add = o_n * self.layerParam.sample_num**2 * 3
        self.div = o_n * self.layerParam.sample_num**2 * 6 + o_n
        # memory
        self.param = 0
        self.activation = o_n

    def setup(self, layer):
        # psroialignpooling has no param
        pass


class PSROIMASKPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PSROIMASKPoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.psroi_mask_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = self.layerParam.output_dim
        self.topShape[0][2] = self.layerParam.group_size
        self.topShape[0][3] = self.layerParam.group_size
        # TODO calcualtion and activation

    def setup(self, layer):
        # psroimaskpooling has no param
        pass


class ROIAlignPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ROIAlignPoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.roi_align_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = prev[0][1]
        self.topShape[0][2] = self.layerParam.pooled_h
        self.topShape[0][3] = self.layerParam.pooled_w
        # TODO calcualtion and activation

    def setup(self, layer):
        # psroimaskpooling has no param
        self.topShape[0][1] = prev[0][1]
        self.topShape[0][2] = self.layerParam.pooled_h
        self.topShape[0][3] = self.layerParam.pooled_w
        # calculation
        o_n = (
            prev[1][0]
            * prev[0][1]
            * self.layerParam.pooled_h
            * self.layerParam.pooled_w
        )
        # memory
        self.param = 0
        self.activation = o_n


class PODROIAlignPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PODROIAlignPoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.podroi_align_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = prev[0][1]
        self.topShape[0][2] = self.layerParam.pooled_h
        self.topShape[0][3] = self.layerParam.pooled_w
        # calculation
        o_n = (
            prev[1][0]
            * prev[0][1]
            * self.layerParam.pooled_h
            * self.layerParam.pooled_w
        )
        # memory
        self.param = 0
        self.activation = o_n

    def setup(self, layer):
        # podroialignpooling has no param
        pass


class AxpyLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(AxpyLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.axpy_param

    def infer(self, prev):
        bs = prev[1][0]
        ic = prev[1][1]
        ih = prev[1][2]
        iw = prev[1][3]

        # infer the output shape
        self.topShape[0][:] = prev[2]
        on = bs * ic * ih * iw
        self.add = on
        self.div = on
        self.param = 0
        self.activation = on

    def setup(self, layer):
        pass


class ArgMaxLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ArgMaxLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.argmax_param

    def infer(self, prev):
        axis = self.layerParam.axis if self.layerParam.HasField("axis") else 1
        topk = self.layerParam.top_k

        # infer the output shape
        num_top = len(self.top)
        self.topShape = np.zeros([num_top, len(prev[0])], dtype=np.int32)
        for t in range(num_top):
            self.topShape[t] = prev[0]
            self.topShape[t][axis] = topk

        self.param = 0
        self.activation = np.multiply.reduce(self.topShape[0][:])

    def setup(self, layer):
        pass


class ExchangeLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ExchangeLayer, self).__init__(layerParameter)
        # exchange has no param

    def infer(self, prev):
        bs = prev[0][0]
        ic = prev[0][1]
        ih = prev[0][2]
        iw = prev[0][3]

        # infer the output shape
        self.topShape[0][:] = [iw, bs, ic, ih]
        # exchange has no calculation, just move data
        self.param = 0
        self.activation = bs * ic * ih * iw

    def setup(self, layer):
        # exchange has no param
        pass


class TransposeLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(TransposeLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.transpose_param

    def infer(self, prev):
        # infer the output shape
        self.topShape = np.zeros([1, len(prev[0])], dtype=np.int32)
        ndim = len(prev[0])
        for i in range(ndim):
            self.topShape[0][i] = prev[0][self.layerParam.dim[i]]
        # transpose has no calculation, just move data
        self.param = 0
        self.activation = self.topShape[0].prod()

    def setup(self, layer):
        # transpose has no param
        pass


class ReverseLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ReverseLayer, self).__init__(layerParameter)
        # reverse has no param

    def infer(self, prev):
        # infer the output shape
        if len(prev) > 1:
            self.topShape = np.zeros([1, len(prev[1])], np.int32)
            self.topShape[0][:] = prev[1]
        else:
            self.topShape = np.zeros([1, len(prev[0])], np.int32)
            self.topShape[0][:] = prev[0]
        # reverse has no calculation, just move data
        self.param = 0
        if len(prev) > 1:
            self.activation = prev[1].prod()
        else:
            self.activation = prev[0].prod

    def setup(self, layer):
        # reverse has no param
        pass


class ShuffleChannelLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ShuffleChannelLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.shuffle_channel_param

    def infer(self, prev):
        # infer the output shape
        # shufflechannel does not change shape, just move data
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]

        # shufflechannel has no calculation, just move data
        self.param = 0
        self.activation = prev[0].prod()

    def setup(self, layer):
        # shuffle has no param
        pass


class SubpixelUpLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(SubpixelUpLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.subpixel_up_param

    def infer(self, prev):
        sample = self.layerParam.upsample
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        self.topShape[0][1] = prev[0][1] // sample**2
        self.topShape[0][2] = prev[0][2] * sample
        self.topShape[0][3] = prev[0][3] * sample

        self.param = 0
        self.activation = prev[0].prod()

    def setup(self, layer):
        pass


class SubpixelDownLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(SubpixelDownLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.subpixel_down_param

    def infer(self, prev):
        sample = self.layerParam.downsample
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        self.topShape[0][1] = prev[0][1] * sample**2
        self.topShape[0][2] = prev[0][2] // sample
        self.topShape[0][3] = prev[0][3] // sample

        self.param = 0
        self.activation = prev[0].prod()

    def setup(self, layer):
        # shuffle has no param
        pass


class PadLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PadLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.pad_param

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]
        j = 0
        for i in self.layerParam.pads:
            self.topShape[0][j % len(self.topShape[0])] += i
            j = j + 1
        self.activation = self.topShape[0].prod()

    def setup(self, layer):
        pass


class PowerLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(PowerLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.power_param

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]

        self.param = 0
        total = prev[0].prod()
        self.activation = total
        self.macc = total
        self.exp = total


class ScalesLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ScalesLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.scales_param

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]

        self.param = 0
        total = prev[0].prod()
        self.activation = total
        self.macc = total
        self.exp = total


class ROIPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ROIPoolingLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.roi_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = prev[0][1]
        self.topShape[0][2] = self.layerParam.pooled_h
        self.topShape[0][3] = self.layerParam.pooled_w
        # cannot infer the calculation and activation statically


class MVNLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(MVNLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.mvn_param

    def infer(self, prev):
        self.topShape[0][:] = prev[0]

        self.param = 0
        total = prev[0].prod()
        self.activation = total
        if self.layerParam.across_channels:
            nc = prev[0][0]
        else:
            nc = prev[0][0] * prev[0][1]
        if not self.layerParam.normalize_variance:
            self.add = total * 2
            self.div = nc
        else:
            self.add = total * 3
            self.div = nc * 2 + total
            self.mul = total


class PSROIMaskPoolingLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super().__init__(layerParameter)
        self.layerParam = layerParameter.psroi_mask_pooling_param

    def infer(self, prev):
        # infer the output shape
        self.topShape[0][0] = prev[1][0]
        self.topShape[0][1] = self.layerParam.output_dim
        self.topShape[0][2] = self.layerParam.group_size
        self.topShape[0][3] = self.layerParam.group_size
        # calculation, not known.
        o_n = prev[1][0] * self.layerParam.output_dim * self.layerParam.group_size**2
        # memory
        self.param = 0
        self.activation = o_n

    def setup(self, layer):
        # psroialignpooling has no param
        pass


class ParameterLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ParameterLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.parameter_param

    def infer(self, prev):
        # infer the output shape
        if self.layerParam.HasField("height") and self.layerParam.HasField("width"):
            ib = self.layerParam.batch
            ic = self.layerParam.channel
            ih = self.layerParam.height
            iw = self.layerParam.width
            self.activation = ib * ic * ih * iw
            self.topShape[0][0] = ib
            self.topShape[0][1] = ic
            self.topShape[0][2] = ih
            self.topShape[0][3] = iw
        elif self.layerParam.HasField("m") and self.layerParam.HasField("n"):
            m = self.layerParam.m
            n = self.layerParam.n
            self.topShape = np.zeros((len(self.top), 2), dtype=np.int32)
            self.activation = m * n
            self.topShape[0][0] = m
            self.topShape[0][1] = n
        else:
            raise RuntimeError("ParameterLayer Parameter Error")
        self.param = 0

    def setup(self, layer):
        pass


class Reshape3dLayer(ReshapeLayer):
    def __init__(self, layerParameter):
        super(Reshape3dLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.reshape3d_param


class MatMulLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(MatMulLayer, self).__init__(layerParameter)
        self.layerParam = layerParameter.matmul_param

    def check_shape(self, lsp):
        lsp = list(lsp)
        if len(lsp) != 3:
            sp = lsp[:]
            if len(sp) < 3:
                sp = [1] * (3 - len(sp)) + sp
            else:
                if not all([s == 1 for s in sp[3:]]):
                    sp = [-1] + sp[-2:]
                else:
                    sp = sp[:3]
            return sp
        return lsp

    def infer(self, prev):
        # infer the output shape
        ipl, ipr = list(prev[0]), (prev[1])
        ipl = self.check_shape(ipl)
        ipr = self.check_shape(ipr)
        assert (
            ipl[-1] == ipr[1] and ipl[0] == ipr[0]
        ), f"input shapes are not consistent: f{ipl} vs {ipr}"
        self.topShape = np.zeros((len(self.top), 3), dtype=np.int32)
        self.topShape[0][0] = ipl[0]
        self.topShape[0][1] = ipl[1]
        self.topShape[0][2] = ipr[-1]
        self.param = 0

    def setup(self, layer):
        pass


class AbsLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(AbsLayer, self).__init__(layerParameter)

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]

        self.param = 0
        self.activation = prev[0].prod()


class ReciprocalLayer(CaffeLayer):
    def __init__(self, layerParameter):
        super(ReciprocalLayer, self).__init__(layerParameter)

    def infer(self, prev):
        self.topShape = np.zeros([1, len(prev[0])], np.int32)
        self.topShape[0][:] = prev[0]

        self.div = prev[0].prod()
        self.param = 0
        self.activation = prev[0].prod()


type2class = {
    "Input": InputLayer,
    "ArgMax": ArgMaxLayer,
    "Convolution": ConvolutionLayer,
    "HoleConvolution": ConvolutionLayer,
    "Deconvolution": DeconvolutionLayer,
    "Correlation": CorrelationLayer,
    "Correlation1D": Correlation1DLayer,
    "Pooling": PoolingLayer,
    "PSROIPooling": PSROIPoolingLayer,
    "PSROIAlignPooling": PSROIAlignPoolingLayer,
    "PSROIMASKPooling": PSROIMASKPoolingLayer,
    "ROIAlignPooling": ROIAlignPoolingLayer,
    "PODROIAlignPooling": PODROIAlignPoolingLayer,
    "Eltwise": EltwiseLayer,
    "Reduce": ReduceLayer,
    "ReLU6": ReluLayer,
    "ReLU": ReluLayer,
    "PReLU": PReluLayer,
    "Tanh": TanhLayer,
    "TanH": TanhLayer,
    "Scale": ScaleLayer,
    "BatchNorm": BatchnormLayer,
    "BN": BNLayer,
    "Sigmoid": SigmoidLayer,
    "Hsigmoid": HsigmoidLayer,
    "HSigmoid": HsigmoidLayer,
    "Hswish": HSwishLayer,
    "HSwish": HSwishLayer,
    "Softmax": SoftmaxLayer,
    "TopkSoftmax": SoftmaxLayer,
    "LRN": LRNLayer,
    "InnerProduct": InnerproductLayer,
    "HeatMap2Coord": Heatmap2coordLayer,
    "Resize": InterpLayer,
    "Interp": InterpLayer,
    "NNInterp": InterpLayer,
    "NNUpsample": InterpLayer,
    "Reshape": ReshapeLayer,
    "Concat": ConcatLayer,
    "Slice": SliceLayer,
    "Dropout": DropoutLayer,
    "SLLSTM": SLLSTMLayer,
    "Axpy": AxpyLayer,
    "Exchange": ExchangeLayer,
    "Transpose": TransposeLayer,
    "Reverse": ReverseLayer,
    "ShuffleChannel": ShuffleChannelLayer,
    "SubpixelUp": SubpixelUpLayer,
    "SubpixelDown": SubpixelDownLayer,
    "ChannelShuffle": ShuffleChannelLayer,
    "Reduction": ReductionLayer,
    "Pad": PadLayer,
    "Power": PowerLayer,
    "ROIPooling": ROIPoolingLayer,
    "MVN": MVNLayer,
    "Exp": CaffeExpLayer,
    "PSROIMaskPooling": PSROIMaskPoolingLayer,
    "Parameter": ParameterLayer,
    "Reshape3d": Reshape3dLayer,
    "Convolution3d": Convolution3dLayer,
    "BatchNorm3d": Batchnorm3dLayer,
    "ReLU3d": Relu3dLayer,
    "Pooling3d": Pooling3dLayer,
    "MatMul": MatMulLayer,
    "Abs": AbsLayer,
    "Reciprocal": ReciprocalLayer,
    "Scales": ScalesLayer,
}


def inferNet(net, mergeConcAndSli=False):
    index = 0
    blobDict = {}
    inferLayers = [None] * (len(net.layer) + len(net.input))
    concatPointDict = {}

    for i in range(len(net.input)):
        layer = caffe_pb2.LayerParameter()
        layer.name = net.input[i]
        layer.type = "Input"
        layer.top.append(layer.name)
        layer.input_param.shape.add()
        if len(net.input_shape) > 0:
            # layer.input_param.shape[0].dim[:] = np.array(net.input_shape[i].dim, dtype = np.int32)
            layer.input_param.shape[0].dim[:] = net.input_shape[i].dim
        elif len(net.input_dim) > 0:
            # layer.input_param.shape[0].dim[:] = np.array(net.input_dim[i * 4:(i + 1) * 4], dtype = np.int32)
            layer.input_param.shape[0].dim[:] = net.input_dim[i * 4 : (i + 1) * 4]
        else:
            layer.input_param.shape[0].dim[:] = [1, 3, 224, 224]
        inferLayer = type2class[layer.type](layer)
        inferLayer.infer(None)
        for i in range(len(inferLayer.top)):
            blobDict[inferLayer.top[i]] = inferLayer.topShape[i]
        inferLayers[index] = inferLayer
        index += 1

    for layer in net.layer:
        try:
            inferLayer = type2class[layer.type](layer)
        except Exception as e:
            print("warning: unsupport layer type: %s" % layer.type)
            print("Exception info:")
            print(e)
            for i in range(len(layer.top)):
                if len(layer.bottom) == 0:
                    blobDict[layer.top[i]] = [1, 3, 224, 224]
                else:
                    blobDict[layer.top[i]] = blobDict[layer.bottom[0]]
            inferLayers[index] = DefaultLayer(None)
            index += 1
            # print some related information
            print(layer)
            continue
        bottoms = []
        for bottom in inferLayer.bottom:
            bottoms.append(blobDict[bottom])
        inferLayer.infer(bottoms)
        for i in bottoms:
            memacc = 1
            for j in i:
                memacc *= j
            inferLayer.memacc += memacc
        for i in inferLayer.topShape:
            memacc = 1
            for j in i:
                memacc *= j
            inferLayer.memacc += memacc
        inferLayer.memacc += inferLayer.param

        if mergeConcAndSli and layer.type == "Concat":
            concatPoints = []
            for bottom in bottoms:
                concatPoints.append(bottom[1])
            concatPoints.pop()
            for i in range(1, len(concatPoints)):
                concatPoints[i] += concatPoints[i - 1]
            concatPointDict[layer.name] = concatPoints
        # update blob dict
        for i in range(len(inferLayer.top)):
            blobDict[inferLayer.top[i]] = inferLayer.topShape[i]
        # record inferlayer
        inferLayers[index] = inferLayer
        index += 1
    return inferLayers, concatPointDict, blobDict


def count(net):
    comp = 0
    add = 0
    div = 0
    macc = 0
    exp = 0
    param = 0
    activation = 0
    memacc = 0

    dataPerLayer = []
    inputData = ""
    inferLayers = inferNet(net)[0]
    counter = 0
    for i in inferLayers:
        counter += 1
        if i == None:
            print(counter)
            break
    for inferLayer in inferLayers:
        comp += inferLayer.comp
        add += inferLayer.add
        div += inferLayer.div
        macc += inferLayer.macc
        exp += inferLayer.exp
        param += inferLayer.param
        activation += inferLayer.activation
        memacc += inferLayer.memacc
        dataPerLayer.append(
            [
                inferLayer.name,
                inferLayer.param,
                inferLayer.activation,
                inferLayer.comp,
                inferLayer.add,
                inferLayer.div,
                inferLayer.macc,
                inferLayer.exp,
                inferLayer.memacc,
            ]
        )
        if isinstance(inferLayer, InputLayer):
            for shape in inferLayer.topShape:
                inputData += "%s:%s\n" % (inferLayer.name, shape)

    print(
        "comp: %f, add: %f, div: %f, macc: %f, exp: %f"
        % (
            comp / 1024.0 / 1024,
            add / 1024.0 / 1024,
            div / 1024.0 / 1024,
            macc / 1024.0 / 1024,
            exp / 1024.0 / 1024,
        )
    )
    print(
        "param: %f, activation: %f"
        % (param * 4.0 / 1024 / 1024, activation * 4 / 1024 / 1024)
    )
    print("memacc: %f" % (memacc * 4.0 / 1024 / 1024))
    print("mflop: %f" % (macc * 2 / 1024 / 1024))
    dataSum = [inputData[:-1], param, activation, comp, add, div, macc, exp, memacc]
    # dataSum = [param, activation, comp, add, div, macc, exp]
    return (dataPerLayer, dataSum)


def countFile(filename):
    net = readNetStructure(filename)[0]
    res = count(net)
    res[1].insert(0, os.path.splitext(os.path.basename(filename))[0])
    return res


def countDirectory(dirname):
    dirname = os.path.abspath(dirname)
    filenames = os.listdir(dirname)
    dataPerFile = []
    for filename in filenames:
        filename = os.path.join(dirname, filename)
        if os.path.isdir(filename):
            dataPerFile.extend(countDirectory(filename))
        else:
            net = readNetStructure(filename)[0]
            if net != None:
                print(filename)
                res = count(net)
                res[1].insert(
                    0,
                    os.path.basename(dirname)
                    + "/"
                    + os.path.splitext(os.path.basename(filename))[0],
                )
                dataPerFile.append(res)
    return dataPerFile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="count computation and memory usage for caffe model"
    )
    parser.add_argument(
        "modelpath",
        nargs="+",
        help="path to the caffe model or directory contains caffe model",
    )
    # parser.add_argument("-r", "--recursively", action = "store_true", help = "count all model in given directory")
    args = parser.parse_args()

    finalRes = []
    head1 = [
        "model_name",
        "input_shape",
        "param_size/MBytes",
        "mem_acces/MBytes",
        "comp/Mflop",
        "add/Mflop",
        "div/Mflop",
        "macc/Mflop",
        "exp/Mflop",
    ]
    head2 = [
        "layer_name",
        "param_size/MBytes",
        "mem_acces/MBytes",
        "comp/Mflop",
        "add/Mflop",
        "div/Mflop",
        "macc/Mflop",
        "exp/Mflop",
    ]
    for filename in args.modelpath:
        print(filename)
        if not os.path.exists(filename):
            print("file %s doesn't exist!" % filename)
        elif os.path.isfile(filename):
            finalRes.append(countFile(filename))
        else:
            # print("is a directory")
            finalRes.extend(countDirectory(filename))
    mergeFlag = len(finalRes) > 1
    # generate data info for each model
    for res in finalRes:
        for i in range(2, 4):
            res[1][i] /= 1024.0 * 1024 / 4
        for i in range(4, len(res[1])):
            res[1][i] /= 1024.0 * 1024
        with open(res[1][0].replace("/", "_") + ".csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(head2)
            for row in res[0]:
                for i in range(1, 3):
                    row[i] /= 1024.0 * 1024 / 4
                for i in range(3, len(row)):
                    row[i] /= 1024.0 * 1024
                writer.writerow(row)
            writer.writerow(["input_shape", res[1][1]])
            writer.writerow(["total"] + res[1][2:])

    if mergeFlag:
        # print("write a sum file")
        with open("sum_count.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(head1)
            for res in finalRes:
                writer.writerow(res[-1])
