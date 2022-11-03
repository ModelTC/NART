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


class PODROIAlignPooling(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        self.params.podroi_align_pooling_param.spatial_scale = attributes[
            "spatial_scale"
        ].f
        self.params.podroi_align_pooling_param.pooled_h = attributes["pooled_height"].i
        self.params.podroi_align_pooling_param.pooled_w = attributes["pooled_width"].i

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = [
            self.network.blobshape[self.params.bottom[1]][0],
            self.network.blobshape[self.params.bottom[0]][1],
            self.params.podroi_align_pooling_param.pooled_h,
            self.params.podroi_align_pooling_param.pooled_w,
        ]
