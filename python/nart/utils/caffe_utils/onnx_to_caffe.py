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
from ...proto import caffe_pb2
from onnx import helper
import numpy as np
import math

# this file include some func that support onnx to caffe
# 1.merge func ,like tools merge func
# 2.node to layer


####################################################
#########      node parse function      ############
####################################################

from ...tools.pytorch.layer_utils.layer import Layer


def get_attr_with_default(node, attr_name, default_value=None):
    res = [attr for attr in node.attribute if (attr.name == attr_name)]
    if len(res) == 0:
        return default_value
    assert len(res) == 1
    return helper.get_attribute_value(res[0])


class HeatMap2Coord(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.heatmap_param.coord_h = attributes["coord_h"].i
        self.params.heatmap_param.coord_w = attributes["coord_w"].i
        self.params.heatmap_param.coord_reposition = attributes["coord_reposition"].i

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = [
            self.params.bottom[0][0],
            self.params.bottom[0][0] * 3,
        ]


class Eltwise(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.eltwise_param.coeff.extend(attributes["coeff"].floats)

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]


class Softmax(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.softmax_param.axis = attributes["axis"].i

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]


class Hswish(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        pass

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]


class Hsigmoid(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        pass

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]


class BatchNorm(Layer):
    def set_top(self):
        self.params.top[:] = self.node.output

    def set_bottom(self):
        self.params.bottom[:] = self.node.input[0:1]

    def set_param(self):
        batch_norm_param = self.params.batch_norm_param
        batch_norm_param.use_global_stats = True
        batch_norm_param.eps = get_attr_with_default(self.node, "eps", 1e-5)

        for idx in range(1, 4):
            weight = self.network.blobs[self.node.input[idx]]
            self.params.blobs.append(Layer.array_to_blobproto(weight))

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]


class Scale(Layer):
    def set_top(self):
        self.params.top[:] = self.node.output

    def set_bottom(self):
        self.params.bottom[:] = self.node.input[0:1]

    def set_param(self):
        scale_param = self.params.scale_param
        scale_param.axis = get_attr_with_default(self.node, "axis", 1)

        scale = self.network.blobs[self.node.input[1]]
        self.params.blobs.append(Layer.array_to_blobproto(scale))

        if len(self.node.input) >= 3:
            scale_param.bias_term = True
            bias = self.network.blobs[self.node.input[2]]
            self.params.blobs.append(Layer.array_to_blobproto(bias))
        else:
            scale_param.bias_term = False

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = self.network.blobshape[
            self.params.bottom[0]
        ]


own_convert_dict = {
    "HeatMap2Coord": HeatMap2Coord,
    "Eltwise": Eltwise,
    "CaffeSoftmax": Softmax,
    "Hswish": Hswish,
    "Hsigmoid": Hsigmoid,
    "CaffeBatchNorm": BatchNorm,
    "CaffeScale": Scale,
}

####################################################
#########          merge function       ############
####################################################


def merge_upsample_nodes(model):
    """
    Version specific for 0.4.1 and 1.0.x
    A sequence of constant-upsample nodes with consistent blob flow
    corresponds to one holistic upsample(interp nninterp) layer in caffe.
    This method merges the nodes into one node corresponding
    to that layer.
    """
    nodes = model.graph.node[:]
    del model.graph.node[:]

    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if node.op_type != "Constant":
            model.graph.node.extend([node])
            idx += 1
            continue

        constant_node = node
        while idx < len(nodes):
            idx += 1
            if idx >= len(nodes):
                break
            node = nodes[idx]
            if node.op_type != "Upsample":
                model.graph.node.extend([constant_node])
                break
            upsample_node = node
            assert constant_node.output[0] in upsample_node.input
            upsample_node.input.remove(constant_node.output[0])

            constant_attributes = dict(
                zip(
                    [attr.name for attr in constant_node.attribute],
                    constant_node.attribute,
                )
            )
            scale = np.frombuffer(
                constant_attributes["value"].t.raw_data, dtype=np.float64
            )
            input_size = list(upsample_node.input_shape_dict.values())[0]
            upsample_node.attribute.add(
                i=math.floor(scale[2] * input_size[2]), name="height"
            )
            upsample_node.attribute.add(
                i=math.floor(scale[3] * input_size[3]), name="width"
            )
            model.graph.node.extend([upsample_node])
            idx += 1
            break
