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
import nart.tools.caffe.utils.graph as graph
import nart.tools.caffe.count as cm
from nart.tools.caffe.utils.graph import readNetStructure
import os.path
import numpy as np
import google.protobuf.text_format
import copy, argparse

# 给定线性操作的参数，将其合并到目标层中
# scale和shift均为np.array类型
def mergeLinearOperation(dest, scale, shift):
    bias_term = False
    if (
        dest.type == "Convolution"
        or dest.type == "HoleConvolution"
        or dest.type == "Deconvolution"
    ):
        bias_term = dest.convolution_param.bias_term
        dest.convolution_param.bias_term = True
    elif dest.type == "InnerProduct":
        bias_term = dest.inner_product_param.bias_term
        dest.inner_product_param.bias_term = True
    else:
        print("Unimplemented")

    if not bias_term:
        newVariance = dest.blobs.add()
        if dest.type == "Deconvolution":
            newVariance.shape.dim.append(dest.blobs[0].shape.dim[1])
            newVariance.data.extend([0] * newVariance.shape.dim[0])
        else:
            newVariance.shape.dim.append(dest.blobs[0].shape.dim[0])
            newVariance.data.extend([0] * newVariance.shape.dim[0])
    # update weights
    if dest.type == "Deconvolution":
        output = dest.blobs[0].shape.dim[1]
        weights = np.array(dest.blobs[0].data, dtype=np.float32).reshape(
            *dest.blobs[0].shape.dim
        )
        weights = np.transpose(weights, (1, 0, 2, 3)).reshape(output, -1).T
    else:
        output = dest.blobs[0].shape.dim[0]
        weights = np.array(dest.blobs[0].data, dtype=np.float32).reshape(output, -1).T

    weights *= scale

    if dest.type == "Deconvolution":
        shape = (
            dest.blobs[0].shape.dim[1],
            dest.blobs[0].shape.dim[0],
            dest.blobs[0].shape.dim[2],
            dest.blobs[0].shape.dim[3],
        )
        weights = np.transpose(np.reshape(weights.T, shape), (1, 0, 2, 3))
        weights = weights.reshape(-1)
    else:
        weights = weights.T.reshape(-1)

    for i in range(len(weights)):
        dest.blobs[0].data[i] = np.asscalar(weights[i])
    # update bias
    bias = np.array(dest.blobs[1].data, dtype=np.float32)
    bias *= scale
    bias += shift

    for i in range(len(bias)):
        dest.blobs[1].data[i] = np.asscalar(bias[i])


def mergeBeforeLinearOperation(dest, scale, shift):
    bias_term = False
    if dest.type == "Convolution":
        bias_term = dest.convolution_param.bias_term
        dest.convolution_param.bias_term = True
    elif dest.type == "InnerProduct":
        bias_term = dest.inner_product_param.bias_term
        dest.inner_product_param.bias_term = True
    else:
        print("Unimplemented")

    if not bias_term:
        newVariance = dest.blobs.add()
        newVariance.shape.dim.append(dest.blobs[0].shape.dim[0])
        newVariance.data.extend([0] * newVariance.shape.dim[0])
    # update weights and bias
    weights = np.array(dest.blobs[0].data, dtype=np.float32)
    bias = np.array(dest.blobs[1].data, dtype=np.float32)
    if dest.type == "Convolution":
        weights = weights.reshape(dest.blobs[0].shape.dim)
    elif dest.type == "InnerProduct":
        weights = weights.reshape(
            dest.inner_product_param.num_output,
            len(scale),
            dest.blobs[0].shape.dim[1] // len(scale),
            1,
        )

    for i in range(len(shift)):
        bias += np.sum(weights[:, i, :, :] * shift[i], axis=(1, 2))
    for i in range(len(scale)):
        weights[:, i, :, :] *= scale[i]
    weights = weights.reshape(-1)

    for i in range(len(weights)):
        dest.blobs[0].data[i] = np.asscalar(weights[i])
    for i in range(len(bias)):
        dest.blobs[1].data[i] = np.asscalar(bias[i])


# 将BatchNorm层的参数合并到卷积层中去
def mergeBatchNormLayer(prev, succ, net, handleBlob):
    if handleBlob:
        dest = prev.content
        bn = succ.content
        # 首先计算实际使用的均值和方差
        # 将归一化操作转化为线性运算
        bnMVF = bn.blobs[2].data[0]
        bnMVF = 1 / bnMVF if bnMVF != 0 else 1
        bnMean = np.array(bn.blobs[0].data, dtype=np.float32) * bnMVF
        bnVariance = np.array(bn.blobs[1].data, dtype=np.float32) * bnMVF
        # print(bnMean, bnVariance, bnMVF)
        scale = (bnVariance + bn.batch_norm_param.eps) ** -0.5
        shift = -bnMean * scale
        # 调用统一接口合并卷积操作和线性操作
        mergeLinearOperation(dest, scale, shift)
    # 删除被合并的层
    deleteLayer(succ, net)


# 将BN层的参数合并到卷积层中去
# 根据从同名cpp文件中读出的逻辑 BN层的blobs中存储的数据依次为scale shift mean variance
def mergeBNLayer(prev, succ, net, handleBlob):
    if handleBlob:
        dest = prev.content
        bn = succ.content
        # 取出bn层的参数
        bnScale = np.array(bn.blobs[0].data, dtype=np.float32)
        bnShift = np.array(bn.blobs[1].data, dtype=np.float32)
        bnMean = np.array(bn.blobs[2].data, dtype=np.float32)
        bnVariance = np.array(bn.blobs[3].data, dtype=np.float32)
        # 将BN层的归一化操作转化为线性运算
        newScale = bnScale * ((bnVariance + bn.bn_param.var_eps) ** -0.5)
        newShift = bnShift - newScale * bnMean
        # 调用统一接口合并卷积操作和线性操作
        mergeLinearOperation(dest, newScale, newShift)
    # 删除被合并的层
    deleteLayer(succ, net)


# 将Scale层的参数合并到卷积层中去
def mergeScaleLayer(prev, succ, net, handleBlob):
    if handleBlob:
        dest = prev.content
        scale = succ.content
        # 取出scale层的blob并转换为np.array类型
        alpha = np.array(scale.blobs[0].data, dtype=np.float32)
        if scale.scale_param.bias_term and len(scale.blobs) > 1:
            beta = np.array(scale.blobs[1].data, dtype=np.float32)
        else:
            beta = np.array([0] * scale.blobs[0].shape.dim[0], dtype=np.float32)
        # 调用统一接口合并卷积操作和线性操作
        mergeLinearOperation(dest, alpha, beta)
    # 删除被合并的层
    deleteLayer(succ, net)


# 将Scale层的参数合并到卷积层中去(Scale层在卷积层之前的情况)
# prev表示scale层 succ表示卷积层
def mergeBeforeScaleLayer(prev, succ, net, handleBlob):
    if handleBlob:
        dest = succ.content
        scale = prev.content
        # 取出scale层的blob并转换为np.array类型
        alpha = np.array(scale.blobs[0].data, dtype=np.float32)
        if scale.scale_param.bias_term and len(scale.blobs) > 1:
            beta = np.array(scale.blobs[1].data, dtype=np.float32)
        else:
            beta = np.array([0] * scale.blobs[0].shape.dim[0], dtype=np.float32)
        # 调用统一接口合并卷积操作和线性操作 差别在于原模型中线性操作先于卷积操作进行
        mergeBeforeLinearOperation(dest, alpha, beta)
    # 删除被合并的scale层
    deleteLayer(prev, net)


# 将BN层的参数合并到卷积层中去(Scale层在卷积层之前的情况)
# prev表示BN层 succ表示卷积层
def mergeBeforeBNLayer(prev, succ, net, handleBlob):
    if handleBlob:
        dest = succ.content
        bn = prev.content
        # 取出bn层的blob并转换为np.array类型
        bnScale = np.array(bn.blobs[0].data, dtype=np.float32)
        bnShift = np.array(bn.blobs[1].data, dtype=np.float32)
        bnMean = np.array(bn.blobs[2].data, dtype=np.float32)
        bnVariance = np.array(bn.blobs[3].data, dtype=np.float32)
        # 将BN层的归一化操作转化为线性运算
        newScale = bnScale * ((bnVariance + bn.bn_param.var_eps) ** -0.5)
        newShift = bnShift - newScale * bnMean
        # 调用统一接口合并卷积操作和线性操作
        mergeBeforeLinearOperation(dest, newScale, newShift)
    # 删除被合并的bn层
    deleteLayer(prev, net)


# 将batchnorm层的参数合并到卷积层中去(Scale层在卷积层之前的情况)
# prev表示batchnorm层 succ表示卷积层
def mergeBeforeBatchNormLayer(prev, succ, net, handleBlob):
    if handleBlob:
        dest = succ.content
        batchnorm = prev.content
        # 取出bn层的blob并转换为np.array类型
        bnMVF = batchnorm.blobs[2].data[0]
        bnMVF = 1 / bnMVF if bnMVF != 0 else 1
        bnMean = np.array(batchnorm.blobs[0].data, dtype=np.float32) * bnMVF
        bnVariance = np.array(batchnorm.blobs[1].data, dtype=np.float32) * bnMVF
        # 将BN层的归一化操作转化为线性运算
        scale = (bnVariance + batchnorm.batch_norm_param.eps) ** -0.5
        shift = -bnMean * scale
        # 调用统一接口合并卷积操作和线性操作
        mergeBeforeLinearOperation(dest, scale, shift)
    # 删除被合并的bn层
    deleteLayer(prev, net)


def dupConvLayer(succ, net, handleBlob):
    conv = succ.content

    import copy

    conv_su = succ.succ[0]
    new_conv = copy.deepcopy(conv)
    new_conv.name = conv.name + conv_su.content.top[0]
    new_conv.top[0] = new_conv.name
    conv_su.content.bottom[0] = new_conv.top[0]

    insertNewLayer(new_conv, conv.name, net)


# 将BN层拆分为batch_norm和scale两个层
# 参数succ表示当前BN层
def splitBNLayer(succ, net, handleBlob):
    bn = succ.content
    # 创建新的batch_norm和scale层
    batch_norm = caffe_pb2.LayerParameter()
    batch_norm.name = bn.name + "_spl_bn"
    batch_norm.type = "BatchNorm"
    batch_norm.bottom.extend(bn.bottom)
    batch_norm.batch_norm_param.use_global_stats = bn.bn_param.moving_average
    batch_norm.batch_norm_param.eps = bn.bn_param.var_eps
    batch_norm.batch_norm_param.moving_average_fraction = bn.bn_param.decay

    scale = caffe_pb2.LayerParameter()
    scale.name = bn.name + "-spl-scale"
    scale.type = "Scale"
    scale.top.extend(bn.top)

    # 将bantch_norm和scale连接起来
    batch_norm.top.append(bn.name + "-spl-link")
    scale.bottom.append(bn.name + "-spl-link")

    if handleBlob:
        scale.scale_param.bias_term = True
        # 取出bn层的参数
        # bnScale = np.array(bn.blobs[0].data, dtype = np.float32)
        # bnShift = np.array(bn.blobs[1].data, dtype = np.float32)
        # bnMean = np.array(bn.blobs[2].data, dtype = np.float32)
        # bnVariance = np.array(bn.blobs[3].data, dtype = np.float32)
        # 配给batch_norm和scale层
        blob = batch_norm.blobs.add()
        blob.data.extend(bn.blobs[2].data)
        blob.shape.dim.append(bn.blobs[2].shape.dim[1])

        blob = batch_norm.blobs.add()
        blob.data.extend(bn.blobs[3].data)
        blob.shape.dim.append(bn.blobs[3].shape.dim[1])

        blob = batch_norm.blobs.add()
        blob.data.append(1)
        blob.shape.dim.append(1)

        blob = scale.blobs.add()
        blob.data.extend(bn.blobs[0].data)
        blob.shape.dim.append(bn.blobs[0].shape.dim[1])

        blob = scale.blobs.add()
        blob.data.extend(bn.blobs[1].data)
        blob.shape.dim.append(bn.blobs[1].shape.dim[1])

    # 插入新的batch_norm和scale层
    insertNewLayer(scale, bn.name, net)
    insertNewLayer(batch_norm, bn.name, net)
    # 删除之前的BN层
    deleteLayer(succ, net)


# 将prelu层拆分为eltwise和relu层的组合
def splitPReLULayer(succ, net, handleBlob):
    prelu = succ.content
    # 创建新的eltwise层 relu直接利用现有的prelu层
    eltwise = caffe_pb2.LayerParameter()
    eltwise.name = prelu.name + "_elt"
    eltwise.type = "Eltwise"
    eltwise.eltwise_param.operation = caffe_pb2.EltwiseParameter.SUM
    eltwise.top.append(prelu.top[0])
    new_layers = [eltwise]

    if handleBlob:
        if prelu.prelu_param.channel_shared:
            eltwise.bottom.append(prelu.bottom[0])
            eltwise.bottom.append(prelu.top[0] + "_relu")
            eltwise.eltwise_param.coeff.append(prelu.blobs[0].data[0])
            eltwise.eltwise_param.coeff.append(
                1 - np.array(prelu.blobs[0].data[0], dtype=np.float32)
            )
        else:
            # create a new scale layer
            scale = caffe_pb2.LayerParameter()
            scale.name = prelu.name + "_scale"
            scale.type = "Scale"
            scale.bottom.append(prelu.bottom[0])
            scale.top.append(prelu.top[0] + "_scale")
            scale.blobs.add().data.extend(prelu.blobs[0].data)
            scale.blobs[0].shape.dim.append(len(prelu.blobs[0].data))

            # create another scale layer for relu
            relu_scale = caffe_pb2.LayerParameter()
            relu_scale.name = prelu.name + "_rescale"
            relu_scale.type = "Scale"
            relu_scale.bottom.append(prelu.top[0] + "_relu")
            relu_scale.top.append(prelu.top[0] + "_relu_scale")
            relu_scale.blobs.add().data.extend(
                1 - np.array(prelu.blobs[0].data, dtype=np.float32)
            )
            relu_scale.blobs[0].shape.dim.append(len(prelu.blobs[0].data))

            eltwise.bottom.append(scale.top[0])
            eltwise.bottom.append(relu_scale.top[0])
            eltwise.eltwise_param.coeff.append(1.0)
            eltwise.eltwise_param.coeff.append(1.0)

            new_layers = [scale, relu_scale] + new_layers
        prelu.ClearField("blobs")

    prelu.type = "ReLU"
    prelu.top[0] = prelu.top[0] + "_relu"
    prelu.ClearField("prelu_param")
    insertNewLayers(new_layers, prelu.name, net)


# 将relu层调整为inplace, 修改relu.prev的top和relu的top一致，以避免对网络输出tensor进行修改
# 参数中的prev表示relu succ表示relu后连的层(可能有多个)
def inplaceReluLayer(prev, succ, net, handleBlob):
    relu = prev.content
    consumer = prev.prev
    if len(consumer) != 1:
        print(
            f"ignore inplace {relu.name} due to {relu.name} has {len(consumer)} bottoms."
        )
        return
    for idx, t in enumerate(consumer[0].content.top):
        if t == relu.bottom[0]:
            consumer[0].content.top[idx] = relu.top[0]
    relu.bottom[0] = relu.top[0]


# 将新创建的层添加到网络中 参数name指定插入的位置（插入到name对应的层后面）
def insertNewLayer(layer, name, net, after=True):
    counter = 0
    for i in net.layer:
        counter += 1
        if i.name == name:
            break
    if not after:
        counter -= 1
    temp = caffe_pb2.NetParameter()
    temp.layer.extend(net.layer[:counter])
    temp.layer.add().CopyFrom(layer)
    temp.layer.extend(net.layer[counter:])
    net.ClearField("layer")
    net.layer.extend(temp.layer)
    return counter


# 功能相同 批量插入新创建的层
def insertNewLayers(layers, name, net, after=True):
    counter = 0
    for i in net.layer:
        counter += 1
        if i.name == name:
            break
    if not after:
        counter -= 1
    temp = caffe_pb2.NetParameter()
    temp.layer.extend(net.layer[:counter] + layers)
    temp.layer.extend(net.layer[counter:])
    net.ClearField("layer")
    net.layer.extend(temp.layer)


# 更新pair的引用信息 用于在net的层次被修改时保持一致
def updatePairs(pairs, net):
    for i in net.layer:
        for pair in pairs:
            if pair[0].content.name == i.name:
                pair[0].content = i
            if pair[1].content.name == i.name:
                pair[1].content = i
            if pair[0].content.type == "Concat" and pair[1].content.type == "Slice":
                for j in range(len(pair[1].succ)):
                    if pair[1].succ[j].content.name == i.name:
                        pair[1].succ[j].content = i


def updateNetGraph(net, netGraph):
    for i in net.layer:
        for node in netGraph.nodes():
            if node.content.name == i.name:
                node.content = i
                break


# 将slice层的合并到卷积层中去
# 主要逻辑为根据slice_point对卷积层的卷积核做切分
# 满足的相关条件: slice直连在卷积层之后 卷积层只有单个输出 slice层对channel进行分割
def mergeSliceLayer(prev, succ, net, handleBlob):
    dest = prev.content
    layer = succ.content
    # 如果卷积层有多个后继的话 需要引入concat层以保证结果正确性
    importConcat = len(prev.succ) > 1

    start = end = 0
    subLayers = []
    for i in range(len(layer.top)):
        if i == len(layer.top) - 1:
            if dest.type == "InnerProduct":
                end = dest.inner_product_param.num_output
            else:
                end = dest.convolution_param.num_output
        else:
            end = layer.slice_param.slice_point[i]
        # 创建新的子层
        subLayer = copy.deepcopy(dest)
        subLayer.name = dest.name + "_slice_" + str(i)
        subLayer.type = dest.type

        if importConcat:
            # 避免inplace可能带来的问题 不直接使用原来的top名
            subLayer.top[0] = layer.top[i] + "_slice"
            # 找到slice后面的使用其输出且会产生inplace的层 修改其bottom
            # 同样存在deletelayer函数中的问题 对网络的DAG图分析完成后进行修改
            for node in succ.succ:
                for j in range(len(node.content.bottom)):
                    if node.content.bottom[j] == layer.top[i]:
                        node.content.bottom[j] = subLayer.top[0]
        else:
            # 不引入的concat层的话 直接使用原来的top名 可以在inplace发生的时候发挥其节省显存的优势
            subLayer.top[0] = layer.top[i]
        # print(subConv.name, subConv.top[0])
        if handleBlob:
            # 计算卷积核的大小 后面需要以卷积核为单位对其做分割
            # kernelArea = conv.blobs[0].shape.dim[1] * conv.blobs[0].shape.dim[2] * conv.blobs[0].shape.dim[3]
            paramSize = 1
            for i in dest.blobs[0].shape.dim[1:]:
                paramSize *= i
            subLayer.blobs[0].shape.dim[0] = end - start
            subLayer.blobs[0].ClearField("data")
            subLayer.blobs[0].data.extend(
                dest.blobs[0].data[start * paramSize : end * paramSize]
            )

            # 根据具体情况处理bias参数
            if dest.convolution_param.bias_term:
                subLayer.blobs[1].shape.dim[0] = end - start
                subLayer.blobs[1].ClearField("data")
                subLayer.blobs[1].data.extend(dest.blobs[1].data[start:end])

        if dest.type == "Convolution":
            subLayer.convolution_param.num_output = end - start
        elif dest.type == "InnerProduct":
            subLayer.inner_product_param.num_output = end - start
        else:
            print("Unimplemented")
        # insertNewLayer(subLayer, dest.name, net)
        subLayers.append(subLayer)
        start = end
    # 统一添加所有新的layer
    # 如果还有其他使用了卷积层输出的层 创建一个concat层来恢复其输出
    if importConcat:
        concat = caffe_pb2.LayerParameter()
        concat.name = dest.name + "_concat"
        concat.type = "Concat"
        concat.top.append(dest.name)
        for i in layer.top:
            concat.bottom.append(i + "_slice")
        concat.concat_param.axis = 1
        insertNewLayer(concat, dest.name, net)
    # for conv in subLayers:
    #     insertNewLayer(conv, dest.name, net)
    insertNewLayers(subLayers, dest.name, net)
    # 删除之前的卷积层和slice层
    # 此时不再需要匹配新的卷积层和之后的层次的连接关系 直接删除即可 不使用统一接口
    # for temp in net.layer:
    #     if(temp.name == dest.name or temp.name == layer.name):
    #         net.layer.remove(temp)
    net.layer.remove(dest)
    net.layer.remove(layer)


# 合并直连在batchnorm层后面的scale层
# 主要方法为将batchnorm转换为bn层然后修改相应的参数
def mergeBatchNormAndScale(prev, succ, net, handleBlob):
    dest = prev.content
    scale = succ.content
    # 将原来的batchnorm层修改为相应的bn层
    dest.type = "BN"
    dest.bn_param.var_eps = dest.batch_norm_param.eps
    dest.bn_param.decay = dest.batch_norm_param.moving_average_fraction
    dest.bn_param.moving_average = True
    dest.ClearField("batch_norm_param")
    if handleBlob:
        mvf = dest.blobs[2].data[0]
        # dest.blobs.add().CopyFrom(dest.blobs[0])
        channel = len(dest.blobs[0].data)
        dest.blobs[2].CopyFrom(dest.blobs[0])
        dest.blobs.add().CopyFrom(dest.blobs[1])
        if mvf != 0:
            for i in range(len(dest.blobs[0].data)):
                dest.blobs[2].data[i] /= mvf
                dest.blobs[3].data[i] /= mvf
        # dest.blobs[0].CopyFrom(scale.blobs[0])
        dest.blobs[0].data[:] = scale.blobs[0].data
        if scale.scale_param.bias_term:
            # dest.blobs[1].CopyFrom(scale.blobs[1])
            dest.blobs[1].data[:] = scale.blobs[1].data
        else:
            # dest.blobs[1].CopyFrom(scale.blobs[0])
            dest.blobs[1].data[:] = [0] * len(dest.blobs[1].data)
        for i in range(4):
            dest.blobs[i].shape.ClearField("dim")
            dest.blobs[i].shape.dim.extend([1, channel, 1, 1])
    deleteLayer(succ, net)


# 合并直连在bn层后面的scale层
def mergeBNAndScale(prev, succ, net, handleBlob):
    dest = prev.content
    scale = succ.content
    if handleBlob:
        bnScale = np.array(dest.blobs[0].data, np.float32)
        bnShift = np.array(dest.blobs[1].data, np.float32)
        sScale = np.array(scale.blobs[0].data, np.float32)
        if scale.scale_param.bias_term:
            sShift = np.array(scale.blobs[1].data, np.float32)
        else:
            sShift = np.zeros(sScale.shape[0], np.float32)
        bnScale *= sScale
        bnShift *= sScale
        bnShift += sShift
        for i in range(len(bnScale)):
            dest.blobs[0].data[i] = np.asscalar(bnScale[i])
        for i in range(len(bnShift)):
            dest.blobs[1].data[i] = np.asscalar(bnShift[i])
    deleteLayer(succ, net)


def mergeConcatAndSlice(prev, succ, net, handleBlob):
    dest = prev.content
    sli = succ.content
    assert len(dest.bottom) == len(sli.top)
    assert dest.top[0] == sli.bottom[0]

    for i in range(len(sli.top)):
        for node in succ.succ:
            for j in range(len(node.content.bottom)):
                if node.content.bottom[j] == sli.top[i]:
                    node.content.bottom[j] = dest.bottom[i]
    net.layer.remove(sli)


def convertScale2BNLayer(prev, net, handleBlob):
    scale = prev.content

    if handleBlob:
        bias_term = scale.scale_param.bias_term
        channels = len(scale.blobs[0].data)
        # set bn's alpha equal to scale's alpha
        scale.blobs[0].shape.dim.insert(0, 1)
        scale.blobs[0].shape.dim.extend([1, 1])
        # set bn's beta equal to scale's beta
        if not bias_term:
            scale.blobs.add().data.extend(np.zeros((channels), dtype=np.float32))
            scale.blobs[1].shape.dim.extend([1, channels, 1, 1])
        else:
            scale.blobs[1].shape.dim.insert(0, 1)
            scale.blobs[1].shape.dim.extend([1, 1])
        # set bn's std to zero
        scale.blobs.add().data.extend(np.zeros((channels), dtype=np.float32))
        scale.blobs[2].shape.dim.extend([1, channels, 1, 1])
        # set bn's var to one
        scale.blobs.add().data.extend(np.ones((channels), dtype=np.float32))
        scale.blobs[3].shape.dim.extend([1, channels, 1, 1])

    scale.ClearField("scale_param")
    scale.bn_param.moving_average = True
    # mark the convert operation
    scale.name = scale.name + "_c2BN"
    scale.type = "BN"


# 先下几个论断
# 只是进行交换操作 虽然改变了layer 但不用插入和删除操作 交换它们的param就可以
def reverseConcatAndBN(prev, succ, net, handleBlob):
    concat = prev.content
    bn = succ.content
    # 根据之前临时计算的concat层的参数创建新的bn的子层
    subLayers = []
    start = end = 0

    if len(succ.succ) == 1 and (succ.succ[0].content.type == "ReLU"):
        insertReLU = True
    else:
        insertReLU = False
    concat.top[0] = succ.succ[0].content.top[0] if insertReLU else bn.top[0]
    for i in range(len(concat.bottom)):
        if i == len(concat.bottom) - 1:
            end = None
        else:
            end = concat.slice_param.slice_point[i]
        subLayer = copy.deepcopy(bn)
        subLayer.name = concat.bottom[i] + "_bn_" + str(i)
        subLayer.bottom[0] = concat.bottom[i]
        subLayer.top[0] = subLayer.name
        # concat.bottom[i] = subLayer.name
        # concat.top[0] = bn.top[0]
        subLayers.append(subLayer)

        # 附加的作用 迁移bn的时候还要把bn之后的relu层也移过去
        if insertReLU:
            subReluLayer = copy.deepcopy(succ.succ[0].content)
            subReluLayer.name = subLayer.name + "_relu"
            subReluLayer.bottom[0] = subLayer.name
            subReluLayer.top[0] = subLayer.name
            concat.top[0] = succ.succ[0].content.top[0]
            subLayers.append(subReluLayer)

        concat.bottom[i] = subLayer.name

        if handleBlob:
            for j in range(4):
                subLayer.blobs[j].shape.dim[1] = (
                    end - start if (end) else len(bn.blobs[0].data) - start
                )
                subLayer.blobs[j].ClearField("data")
                # subLayer.blobs[j].data.extend(bnData[j][start:end])
                subLayer.blobs[j].data.extend(bn.blobs[j].data[start:end])

        start = end

    insertNewLayers(subLayers, concat.name, net, after=False)
    concat.ClearField("slice_param")
    net.layer.remove(bn)
    if insertReLU:
        net.layer.remove(succ.succ[0].content)


# 合并完成后删除被合并的层
def deleteLayer(node, net):
    # node代表的层应只有一个输出和一个前驱层
    # 且其前驱层（一般为卷积层）应该只有一个输出
    assert len(node.prev) <= 1
    assert len(node.content.top) <= 1
    if len(node.prev) == 1 and len(node.prev[0].succ) == 1:
        assert len(node.prev[0].content.top) <= 1
        node.prev[0].content.top[0] = node.content.top[0]
    else:
        for n in node.succ:
            for i in range(len(n.content.bottom)):
                if n.content.bottom[i] == node.content.top[0]:
                    print("modify top 2")
                    n.content.bottom[i] = node.content.bottom[0]

    # update topo
    for idx, n in enumerate(node.prev[0].succ):
        if n == node:
            del node.prev[0].succ[idx]
            node.prev[0].succ.extend(node.succ[:])
    for s in node.succ:
        for idx, n in enumerate(s.prev):
            if n == node:
                del s.prev[idx]
                s.prev.append(node.prev[0])

    net.layer.remove(node.content)


# 对于将某些layer由非inplace改变为inplace的情况 需要保证新的layer是其逻辑前驱层的物理后继
# 否则会由于caffe的单次扫描机制出现问题
def netTopoAdjust(net):
    for i in range(len(net.layer)):
        # flag = False
        for j in range(i):
            # if ([item for item in net.layer[i].bottom if item in net.layer[j].top]):
            #     continue
            if [item for item in net.layer[i].top if item in net.layer[j].bottom]:
                tempLayer = net.layer.pop(i)
                tempNet = caffe_pb2.NetParameter()
                tempNet.layer.extend(net.layer[:j])
                tempNet.layer.add().CopyFrom(tempLayer)
                tempNet.layer.extend(net.layer[j:])
                net.ClearField("layer")
                net.layer.extend(tempNet.layer)
                # flag = True
                break


# 找出与卷积层直连的后继层 而且该后继层应是唯一后继
# 但对于slice层比较特殊 即使不唯一也可以通过引入concat层的方式来将其分解 此时需要对concat层的支持
def findLayerPairs(
    net,
    batchnorm,
    bn,
    scale,
    sliceLayer,
    batchNormScale,
    bnScale,
    concSli,
    concBN,
    splitBN,
    inplaceRelu,
    cs,
    scaleConv,
    bnConv,
    BNConv,
    scaleip,
    bnip,
    BNip,
    scale2BN,
    prelu,
):
    pairs = []
    prev = []
    concatPointDict = None
    netGraph = graph.gen_graph(net)
    # 为合并concat和slice准备shape
    # if(concSli):
    #     concatPointDict = cm.inferNet(net, mergeConcAndSli = True)[1]
    for node in netGraph.nodes():
        if node.content.name in prev:
            continue
        if (
            node.content.type == "Convolution"
            or node.content.type == "HoleConvolution"
            or node.content.type == "Deconvolution"
            or node.content.type == "InnerProduct"
        ):
            if len(node.succ) == 1:
                if (
                    (node.succ[0].content.type == "BN" and bn)
                    or (node.succ[0].content.type == "BatchNorm" and batchnorm)
                    or (node.succ[0].content.type == "Scale" and scale)
                    or (
                        node.succ[0].content.type == "Slice"
                        and node.succ[0].content.slice_param.axis == 1
                        and sliceLayer
                    )
                ):
                    pairs.append((node, node.succ[0]))
                    # prev = node.succ[0]
                    prev.append(node.succ[0].content.name)
            elif cs and sliceLayer:
                for succ in node.succ:
                    if (
                        succ.content.type == "Slice"
                        and succ.content.slice_param.axis == 1
                    ):
                        pairs.append((node, succ))
                        # prev = succ
                        prev.append(succ.content.name)
        elif node.content.type == "BatchNorm":
            if len(node.succ) == 1:
                if (
                    node.succ[0].content.type == "Scale"
                    and node.succ[0].content.scale_param.axis == 1
                    and batchNormScale
                ):
                    pairs.append((node, node.succ[0]))
                    # prev = node.succ[0]
                    prev.append(node.succ[0].content.name)
                elif node.succ[0].content.type == "Convolution" and bnConv:
                    if node.succ[0].content.convolution_param.HasField("pad"):
                        pad = node.succ[0].content.convolution_param.pad
                    else:
                        pad = (
                            node.succ[0].content.convolution_param.pad_h
                            + node.succ[0].content.convolution_param.pad_w
                        )
                    if pad == 0:
                        pairs.append((node, node.succ[0]))
                        prev.append(node.succ[0].content.name)
                elif node.succ[0].content.type == "InnerProduct" and bnip:
                    pairs.append((node, node.succ[0]))
                    prev.append(node.succ[0].content.name)
        elif node.content.type == "BN":
            if splitBN:
                pairs.append((node, None))
                break
            if len(node.succ) == 1:
                if (
                    node.succ[0].content.type == "Scale"
                    and node.succ[0].content.scale_param.axis == 1
                    and bnScale
                ):
                    pairs.append((node, node.succ[0]))
                    # prev = succ[0]
                    prev.append(node.succ[0].content.name)
                elif node.succ[0].content.type == "Convolution" and BNConv:
                    if node.succ[0].content.convolution_param.HasField("pad"):
                        pad = node.succ[0].content.convolution_param.pad
                    else:
                        pad = (
                            node.succ[0].content.convolution_param.pad_h
                            + node.succ[0].content.convolution_param.pad_w
                        )
                    if pad == 0:
                        pairs.append((node, node.succ[0]))
                        prev.append(node.succ[0].content.name)
                elif node.succ[0].content.type == "InnerProduct" and BNip:
                    pairs.append((node, node.succ[0]))
                    prev.append(node.succ[0].content.name)
        elif node.content.type == "Concat":
            if node.content.concat_param.axis != 1:
                continue
            if concatPointDict == None and (concSli or concBN):
                concatPointDict = cm.inferNet(net, mergeConcAndSli=True)[1]
            if len(node.succ) == 1:
                if node.succ[0].content.type == "BN" and concBN:
                    node.content.slice_param.slice_point.extend(
                        concatPointDict[node.content.name]
                    )
                    pairs.append((node, node.succ[0]))
                    prev.append(node.succ[0].content.name)
            for succ in node.succ:
                if (
                    succ.content.type == "Slice"
                    and succ.content.slice_param.axis == 1
                    and concSli
                    and succ.content.slice_param.slice_point
                    == concatPointDict[node.content.name]
                ):
                    pairs.append((node, succ))
                    # prev = succ
                    prev.append(succ.content.name)
        elif node.content.type == "ReLU":
            if (
                inplaceRelu
                and node.content.bottom[0] != node.content.top[0]
                and len(node.prev) >= 1
                and len(node.prev[0].succ) == 1
            ):
                pairs.append((node, node.succ))
        elif node.content.type == "Scale":
            if len(node.succ) == 1:
                if node.succ[0].content.type == "Convolution" and scaleConv:
                    if node.succ[0].content.convolution_param.HasField("pad"):
                        pad = node.succ[0].content.convolution_param.pad
                    else:
                        pad = (
                            node.succ[0].content.convolution_param.pad_h
                            + node.succ[0].content.convolution_param.pad_w
                        )
                    if pad == 0:
                        pairs.append((node, node.succ[0]))
                        prev.append(node.succ[0].content.name)
                        continue
                elif node.succ[0].content.type == "InnerProduct" and scaleip:
                    pairs.append((node, node.succ[0]))
                    prev.append(node.succ[0].content.name)
                    continue
            if scale2BN:
                pairs.append((node, None))
        elif node.content.type == "PReLU" and prelu:
            pairs.append((node, None))
    return pairs, netGraph


# 生成新的模型 包括proto文件和bin文件
def generateModel(net, oldPath, suffix, withBinFile, pureOutput=False):
    dirName, fileName = os.path.split(oldPath)
    splitIndex = fileName.find(".")
    if splitIndex == -1:
        newProtoName = fileName + "-" + suffix + ".prototxt"
        newModelName = fileName + "-" + suffix + ".caffemodel"
    else:
        newProtoName = fileName[:splitIndex] + "-" + suffix + ".prototxt"
        newModelName = fileName[:splitIndex] + "-" + suffix + ".caffemodel"

    newProtoName = os.path.join(dirName, newProtoName)
    newModelName = os.path.join(dirName, newModelName)

    if withBinFile:
        if pureOutput:
            tempNet = copy.deepcopy(net)
            for layer in tempNet.layer:
                tempLayer = caffe_pb2.LayerParameter()
                tempLayer.name = layer.name
                for blob in layer.blobs:
                    tempLayer.blobs.add().CopyFrom(blob)
                layer.CopyFrom(tempLayer)
            with open(newModelName, "wb") as f:
                f.write(tempNet.SerializeToString())
        else:
            with open(newModelName, "wb") as f:
                f.write(net.SerializeToString())
        for layer in net.layer:
            layer.ClearField("blobs")

    with open(newProtoName, "w") as f:
        f.write(str(net))


def resolve(conv, net, handleBlob):
    group = conv.convolution_param.group
    num_output = conv.convolution_param.num_output
    groupKernelSize = 0
    if handleBlob:
        groupKernelSize = (len(conv.blobs[0].data) / conv.blobs[0].shape.dim[0]) * (
            num_output / group
        )
    # add a slice layer
    sliceLayer = caffe_pb2.LayerParameter()
    sliceLayer.name = conv.name + "_slice"
    sliceLayer.type = "Slice"
    sliceLayer.bottom.append(conv.bottom[0])
    sliceLayer.slice_param.axis = 1
    # add a concat layer
    concatLayer = caffe_pb2.LayerParameter()
    concatLayer.name = conv.name + "_concat"
    concatLayer.type = "Concat"
    concatLayer.top.append(conv.top[0])
    concatLayer.concat_param.axis = 1
    # add some sub convolution layers
    subConvs = []
    for i in range(group):
        subConv = copy.deepcopy(conv)
        subConv.ClearField("blobs")
        # 修改prototxt文件中的相关参数
        subConv.name = conv.name + "_group" + str(i)
        subConv.convolution_param.group = 1
        subConv.convolution_param.num_output = num_output / group
        subConv.bottom[0] = conv.bottom[0] + "_group" + str(i)
        subConv.top[0] = conv.top[0] + "_group" + str(i)
        # handle weights and bias
        if handleBlob:
            # weights = np.array(conv.blobs[0].data, dtype = np.float32)
            subConv.blobs.add()
            # subConv.blobs[0].shape.dim.extend([num_output / group, conv.blobs[0].shape.dim[1] / group] + conv.blobs[0].shape.dim[2:])
            subConv.blobs[0].shape.dim.extend(
                [num_output / group] + conv.blobs[0].shape.dim[1:]
            )
            subConv.blobs[0].data.extend(
                conv.blobs[0].data[i * groupKernelSize : (i + 1) * groupKernelSize]
            )
            if conv.convolution_param.bias_term:
                subConv.blobs.add()
                # subConv.blobs[1].shape.dim.extend([num_output / group, 1, 1, 1])
                subConv.blobs[1].shape.dim.append(num_output / group)
                subConv.blobs[1].data.append(conv.blobs[1].data[i])
        subConvs.append(subConv)
        sliceLayer.top.append(subConv.bottom[0])
        concatLayer.bottom.append(subConv.top[0])
        if i != 0:
            sliceLayer.slice_param.slice_point.append(i * num_output / group)
    # print(subConvs[0].blobs[0].shape)
    # print(len(subConvs[0].blobs[0].data))
    # insert new layers into the net
    insertNewLayer(concatLayer, conv.name, net)
    insertNewLayers(subConvs, conv.name, net)
    insertNewLayer(sliceLayer, conv.name, net)
    # delete origin convolution layer
    # 之前的插入操作会导致整个net被更新 其中所有的layer的引用都和之前的不一样 需要重新确定被删除的conv layer
    # for temp in net.layer:
    #     if(temp.name == conv.name):
    #         net.layer.remove(temp)
    #         break
    net.layer.remove(conv)


# 将带有group参数的卷积层分解为多个子卷积层
def resolveGroupConv(net, handleBlob, verbose=False):
    for layer in net.layer:
        if layer.type == "Convolution" and layer.convolution_param.group > 1:
            if verbose:
                print("resolve conv layer: %s" % layer.name)
            resolve(layer, net, handleBlob)
    return net


# 总控函数 同时可作为接口暴露给其他程序直接使用
def mergeLayers(
    net,
    handleBlob,
    batchnorm=False,
    bn=False,
    scale=False,
    sliceLayer=False,
    bns=False,
    BNs=False,
    concsli=False,
    concbn=False,
    splitbn=False,
    inrelu=False,
    cs=False,
    scaleconv=False,
    bnconv=False,
    BNconv=False,
    scaleip=False,
    bnip=False,
    BNip=False,
    prelu=False,
    scale2BN=False,
    verbose=False,
):
    counter = 0
    pairs, netGraph = findLayerPairs(
        net,
        batchnorm,
        bn,
        scale,
        sliceLayer,
        bns,
        BNs,
        concsli,
        concbn,
        splitbn,
        inrelu,
        cs,
        scaleconv,
        bnconv,
        BNconv,
        scaleip,
        bnip,
        BNip,
        scale2BN,
        prelu,
    )
    while len(pairs) > 0:
        counter += len(pairs)
        for pair in pairs:
            if verbose:
                if pair[1] == None:
                    print("split %s layer" % pair[0].content.name)
                elif type(pair[1]) == list:
                    print("inplace %s layer" % pair[0].content.name)
                else:
                    print("%s -> %s" % (pair[1].content.name, pair[0].content.name))
            if (
                pair[0].content.type == "Convolution"
                or pair[0].content.type == "HoleConvolution"
                or pair[0].content.type == "Deconvolution"
                or pair[0].content.type == "InnerProduct"
            ):
                if pair[1].content.type == "BatchNorm":
                    mergeBatchNormLayer(pair[0], pair[1], net, handleBlob)
                elif pair[1].content.type == "BN":
                    mergeBNLayer(pair[0], pair[1], net, handleBlob)
                elif pair[1].content.type == "Scale":
                    mergeScaleLayer(pair[0], pair[1], net, handleBlob)
                elif pair[1].content.type == "Slice":
                    mergeSliceLayer(pair[0], pair[1], net, handleBlob)
                    # updatePairs(pairs, net)
                    updateNetGraph(net, netGraph)
            elif pair[0].content.type == "BatchNorm":
                if pair[1].content.type == "Scale":
                    mergeBatchNormAndScale(pair[0], pair[1], net, handleBlob)
                elif (
                    pair[1].content.type == "Convolution"
                    or pair[1].content.type == "InnerProduct"
                ):
                    mergeBeforeBatchNormLayer(pair[0], pair[1], net, handleBlob)
            elif pair[0].content.type == "BN":
                if pair[1] == None:
                    splitBNLayer(pair[0], net, handleBlob)
                    updateNetGraph(net, netGraph)
                elif pair[1].content.type == "Scale":
                    mergeBNAndScale(pair[0], pair[1], net, handleBlob)
                elif (
                    pair[1].content.type == "Convolution"
                    or pair[1].content.type == "InnerProduct"
                ):
                    mergeBeforeBNLayer(pair[0], pair[1], net, handleBlob)
            elif pair[0].content.type == "Concat":
                if pair[1].content.type == "BN":
                    reverseConcatAndBN(pair[0], pair[1], net, handleBlob)
                    updateNetGraph(net, netGraph)
                elif pair[1].content.type == "Slice":
                    mergeConcatAndSlice(pair[0], pair[1], net, handleBlob)
            elif pair[0].content.type == "ReLU":
                inplaceReluLayer(pair[0], pair[1], net, handleBlob)
            elif pair[0].content.type == "Scale":
                if pair[1] == None:
                    convertScale2BNLayer(pair[0], net, handleBlob)
                else:
                    mergeBeforeScaleLayer(pair[0], pair[1], net, handleBlob)
            elif pair[0].content.type == "PReLU":
                splitPReLULayer(pair[0], net, handleBlob)
                updateNetGraph(net, netGraph)
            else:
                print("Unimplemented")
                exit(1)
        pairs, netGraph = findLayerPairs(
            net,
            batchnorm,
            bn,
            scale,
            sliceLayer,
            bns,
            BNs,
            concsli,
            concbn,
            splitbn,
            inrelu,
            cs,
            scaleconv,
            bnconv,
            BNconv,
            scaleip,
            bnip,
            BNip,
            scale2BN,
            prelu,
        )
    print("%d layer pairs have been merged" % counter)
    return net


# 根据给出的的参数实现对输入图像的normalize操作 本质即为添加scale layer
def normalize_net(net, names, mean, std, handleBlob, eps=1e-6):
    # input parameter check
    channel_num = 0
    channels = [0]
    for name in names:
        assert name in net.input
        # index = net.input.index(name)
        index = 0
        for item in net.input:
            if item != name:
                index += 1
            else:
                break
        assert index != len(net.input)
        if len(net.input_dim) == 0:
            channel_num += net.input_shape[index].dim[1]
            channels.append(channel_num)
        else:
            channel_num += net.input_dim[index * 4 + 1]
            channels.append(channel_num)
    assert len(mean) == channel_num
    assert len(mean) == len(std)

    if handleBlob:
        # 根据mean和std计算对应的scale层的参数
        alpha = 1 / (np.array(std, dtype=np.float32) + eps)
        beta = (
            -1
            * np.array(mean, dtype=np.float32)
            / (np.array(std, dtype=np.float32) + eps)
        )
    for i in range(len(names)):
        scale = caffe_pb2.LayerParameter()
        scale.name = names[i] + "_norm"
        scale.type = "Scale"
        scale.bottom.append(names[i])
        scale.top.append(names[i])
        scale.scale_param.axis = 1
        scale.scale_param.bias_term = True

        if handleBlob:
            scale.blobs.add()
            scale.blobs[0].data.extend(alpha[channels[i] : channels[i + 1]])
            scale.blobs[0].shape.dim.append(channels[i + 1] - channels[i])

            scale.blobs.add()
            scale.blobs[1].data.extend(beta[channels[i] : channels[i + 1]])
            scale.blobs[1].shape.dim.append(channels[i + 1] - channels[i])

        # 将新的scale layer插入到net中
        insertNewLayer(scale, net.layer[0].name, net, after=False)

    return net


# 某些特殊情况下 删掉仅在训练阶段起作用的layer
def filterTrainLayer(net):
    train_layer = []
    for i in net.layer:
        if len(i.include) > 0 and (i.include[0].phase) == 1:
            train_layer.append(i)
    for i in train_layer:
        net.layer.remove(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="merge specific layers into conv layer"
    )
    parser.add_argument("modelpath1", help="path to the caffe model")
    parser.add_argument(
        "modelpath2", nargs="?", default=None, help="path to the caffe model"
    )
    parser.add_argument(
        "-b", "--batchnorm", action="store_true", help="conv + batchnorm -> conv"
    )
    parser.add_argument("-B", "--bn", action="store_true", help="conv + BN -> conv")
    parser.add_argument(
        "-s", "--scale", action="store_true", help="conv + scale -> conv"
    )
    parser.add_argument("-S", "--slice", action="store_true", help="merge slice layer")
    parser.add_argument("--bns", action="store_true", help="batchnorm + scale -> BN")
    parser.add_argument("--BNs", action="store_true", help="BN + scale -> BN")
    parser.add_argument("--scaleconv", action="store_true", help="scale + conv -> conv")
    parser.add_argument(
        "--bnconv", action="store_true", help="batchnorm + conv -> conv"
    )
    parser.add_argument("--BNconv", action="store_true", help="BN + conv -> conv")
    parser.add_argument("--scaleip", action="store_true", help="scale + ip -> ip")
    parser.add_argument("--bnip", action="store_true", help="batchnorm + ip -> ip")
    parser.add_argument("--BNip", action="store_true", help="BN + ip -> ip")
    parser.add_argument("--scale2BN", action="store_true", help="scale -> BN")
    parser.add_argument(
        "-n", "--normalize", action="store_true", help="normalize the input image"
    )
    parser.add_argument("--name", nargs="*", help="name for input blob")
    parser.add_argument(
        "--mean", type=float, nargs="*", help="mean value for normalization"
    )
    parser.add_argument(
        "--std", type=float, nargs="*", help="standard deviation for normalization"
    )
    parser.add_argument("--concsli", action="store_true", help="merge concat and slice")
    parser.add_argument("--concbn", action="store_true", help="reverse concat and bn")
    parser.add_argument(
        "--splitbn", action="store_true", help="split bn into batch_norm and scale"
    )
    parser.add_argument("--inrelu", action="store_true", help="inplace relu layer")
    parser.add_argument(
        "--prelu", action="store_true", help="convert prelu to eltwise and relu"
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="merge all supported layer"
    )
    parser.add_argument(
        "--cs", action="store_true", help="concat layer support while merge slice layer"
    )
    parser.add_argument(
        "--pure_output",
        action="store_true",
        help="output pure caffemodel which only contains layer name and blobs",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="output detail information"
    )
    parser.add_argument(
        "-r", "--resolve", action="store_true", help="resolve group convolution"
    )
    args = parser.parse_args()
    if args.all:
        args.batchnorm = (
            args.bn
        ) = (
            args.scale
        ) = args.slice = args.bns = args.BNs = args.concsli = args.inrelu = True

    if args.splitbn:
        args.bns = False

    net, withBinFile = readNetStructure(args.modelpath1, args.modelpath2)
    filterTrainLayer(net)
    if args.verbose:
        print("start to merge")
    if net == None:
        print("Error happened while read net structure!!")
        exit(1)
    if args.resolve:
        net = resolveGroupConv(net, withBinFile, args.verbose)
    if args.normalize:
        args.scaleconv = True
        net = normalize_net(net, args.name, args.mean, args.std, withBinFile)
    if (
        args.batchnorm
        or args.bn
        or args.scale
        or args.slice
        or args.bns
        or args.BNs
        or args.concsli
        or args.concbn
        or args.splitbn
        or args.inrelu
        or args.scaleconv
        or args.bnconv
        or args.BNconv
        or args.scaleip
        or args.bnip
        or args.BNip
        or args.prelu
        or args.scale2BN
    ):
        net = mergeLayers(
            net,
            withBinFile,
            args.batchnorm,
            args.bn,
            args.scale,
            args.slice,
            args.bns,
            args.BNs,
            args.concsli,
            args.concbn,
            args.splitbn,
            args.inrelu,
            args.cs,
            args.scaleconv,
            args.bnconv,
            args.BNconv,
            args.scaleip,
            args.bnip,
            args.BNip,
            args.prelu,
            args.scale2BN,
            args.verbose,
        )
    netTopoAdjust(net)
    generateModel(net, args.modelpath1, "convert", withBinFile, args.pure_output)
