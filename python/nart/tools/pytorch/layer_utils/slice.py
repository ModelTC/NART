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


class Slice(Layer):
    def set_top(self):
        # assert len(self.node.output) > 1
        self.params.top.extend(self.node.output)
        if self.node.op_type == "Slice":
            # pytorch would extract multi-index slice to multi slice
            assert (
                len(self.node.output) == 1
            ), "[nart] don't support multi-dims `Slice`."

            attributes = dict(
                zip([attr.name for attr in self.node.attribute], self.node.attribute)
            )

            if len(self.node.input) == 1:
                starts = attributes["starts"].ints
                ends = attributes["ends"].ints
                axes = attributes["axes"].ints
            else:
                starts = self.network.blobs[self.node.input[1]]
                ends = self.network.blobs[self.node.input[2]]
                axes = self.network.blobs[self.node.input[3]]

            if starts[0] != 0:
                self.params.top.insert(0, self.params.name + "_0")
            shape = self.network.blobshape[self.node.input[0]][axes[0]]
            if ends[0] < shape:
                self.params.top.append(self.params.name + "_2")

    def set_bottom(self):
        self.params.bottom.append(self.node.input[0])

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        if self.node.op_type == "Split":
            self.params.slice_param.axis = attributes["axis"].i
            slice_point = 0
            for split in attributes["split"].ints[:-1]:
                slice_point += split
                self.params.slice_param.slice_point.extend([slice_point])
        elif self.node.op_type == "Slice":
            if len(self.node.input) == 1:
                starts = attributes["starts"].ints
                ends = attributes["ends"].ints
                axes = attributes["axes"].ints
            else:
                starts = self.network.blobs[self.node.input[1]]
                ends = self.network.blobs[self.node.input[2]]
                axes = self.network.blobs[self.node.input[3]]
            assert len(axes) == 1
            assert len(starts) == len(ends)
            assert len(starts) == 1, "[nart] don't support multi-dims `Slice`."

            bottom_shape = self.network.blobshape[self.node.input[0]]
            axes = axes[0]
            starts = starts[0]
            ends = ends[0]

            starts = starts if starts >= 0 else bottom_shape[axes] + starts
            ends = ends if ends >= 0 else bottom_shape[axes] + ends
            slice_points = [starts, ends]

            self.params.slice_param.axis = axes
            # remove slice_points which eq to axis_min or axis_max
            if 0 in slice_points:
                slice_points.remove(0)
            axis_max = self.network.blobshape[self.params.bottom[0]][
                self.params.slice_param.axis
            ]
            # if axis_max in slice_points:
            if slice_points[-1] >= axis_max:
                slice_points.remove(slice_points[-1])

            self.params.slice_param.slice_point.extend(slice_points)

    def set_blobshape(self):
        axis = self.params.slice_param.axis
        bottom_shape = self.network.blobshape[self.params.bottom[0]]

        start = 0
        for i in range(len(self.params.top)):
            top_shape = bottom_shape.copy()
            if i == len(self.params.top) - 1:
                top_shape[axis] = bottom_shape[axis] - start
            else:
                top_shape[axis] = self.params.slice_param.slice_point[i] - start
                start = self.params.slice_param.slice_point[i]
            self.network.blobshape[self.params.top[i]] = top_shape
