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

from .layer import Layer
from functools import reduce
import warnings


class Reshape(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        # only the first input of onnx reshape should be passed to caffe reshape.
        self.params.bottom.append(self.node.input[0])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        if self.node.op_type == "Squeeze":
            # FIXME:
            #   may get wrong top_shape when squeeze axis=0
            shape = self.network.blobshape[self.params.bottom[0]].copy()
            import numpy as np

            axes = np.array(attributes["axes"].ints)
            axes[axes < 0] += len(shape)
            axes = sorted(axes, reverse=True)
            for axis in axes:
                if shape[axis] != 1:
                    warnings.warn("`squeeze` on non-1 dim will be ignored")
                else:
                    shape.pop(axis)
            # self.params.reshape_param.shape.dim.extend([0] * len(shape))
            self.params.reshape_param.shape.dim.extend(
                [*[0] * axes[-1], *shape[axes[-1] :]]
            )
        elif self.node.op_type == "Unsqueeze":
            shape = self.network.blobshape[self.params.bottom[0]].copy()
            for axis in attributes["axes"].ints:
                shape.insert(axis, 1)
            self.params.reshape_param.shape.dim.extend(shape)
        elif self.node.op_type == "Flatten":
            shape = [0, -1]
            self.params.reshape_param.shape.dim.extend(shape)
        else:
            if "shape" in attributes.keys():
                bottom_shape = self.network.blobshape[self.params.bottom[0]]
                shape = attributes["shape"].ints
                for i in range(min(len(shape), len(bottom_shape))):
                    if shape[i] == bottom_shape[i]:
                        shape[i] = 0
                self.params.reshape_param.shape.dim.extend(shape)
            elif "dims" in attributes.keys():
                bottom_shape = self.network.blobshape[self.params.bottom[0]]
                shape = []
                for dim in attributes["dims"].ints:
                    if dim == -1:
                        shape.append(dim)
                    else:
                        shape.append(bottom_shape[dim])
                for i in range(min(len(shape), len(bottom_shape))):
                    if shape[i] == bottom_shape[i]:
                        shape[i] = 0
                self.params.reshape_param.shape.dim.extend(shape)
            elif len(self.node.input) >= 2:
                node = self.node
                # a second input as output shape
                shape = self.network.blobs[node.input[1]]
                self.params.reshape_param.shape.dim.extend(shape)
            else:
                raise RuntimeError("unhandled reshape type")

    def set_blobshape(self):
        bottom_shape = self.network.blobshape[self.params.bottom[0]]
        top_shape = self.params.reshape_param.shape.dim[:]

        infer_idx = None
        for idx in range(len(top_shape)):
            dim = top_shape[idx]
            if dim == -1:
                infer_idx = idx
            elif dim == 0:
                top_shape[idx] = bottom_shape[idx]
        if infer_idx is not None:
            num_items = reduce(lambda a, b: a * b, bottom_shape)
            accounted_items = reduce(
                lambda a, b: a * b, [dim for dim in top_shape if dim != -1]
            )
            top_shape[infer_idx] = num_items // accounted_items
        self.network.blobshape[self.params.top[0]] = top_shape
