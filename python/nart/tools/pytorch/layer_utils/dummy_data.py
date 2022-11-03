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
from ...proto import caffe_pb2
import numpy as np


class DummyData(Layer):
    def set_top(self):
        self.params.top.extend(self.node.output)

    def set_bottom(self):
        self.params.bottom.extend(self.node.input)

    def set_param(self):
        attributes = dict(
            zip([attr.name for attr in self.node.attribute], self.node.attribute)
        )
        caffe_filler = caffe_pb2.FillerParameter()
        filler = np.frombuffer(attributes["value"].t.raw_data, dtype=np.float32)
        caffe_filler.value = filler[0]
        caffe_filler.type = "constant"
        self.params.dummy_data_param.data_filler.extend([caffe_filler])

    def set_blobshape(self):
        self.network.blobshape[self.params.top[0]] = [1, 1, 1, 1]
