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

from abc import ABC, abstractmethod
from ....proto import caffe_pb2


class Layer(ABC):
    def __init__(self, node, index, network):
        def _get_module_name(self, node, index):
            attributes = dict(
                zip([attr.name for attr in node.attribute], node.attribute)
            )
            if (
                "module_name" in attributes
                and attributes["module_name"].s.decode("utf-8") is not ""
            ):
                name = attributes["module_name"].s.decode("utf-8") + "_" + str(index)
            else:
                name = self.__class__.__name__ + "_" + str(index)
            return name

        self.params = caffe_pb2.LayerParameter(
            type=self.__class__.__name__, name=_get_module_name(self, node, index)
        )
        self.node = node
        self.network = network
        self.set_top()
        self.set_bottom()
        self.set_param()
        self.set_blobshape()

    @abstractmethod
    def set_top(self):
        raise NotImplementedError

    @abstractmethod
    def set_bottom(self):
        raise NotImplementedError

    @abstractmethod
    def set_param(self):
        raise NotImplementedError

    @abstractmethod
    def set_blobshape(self):
        raise NotImplementedError

    @staticmethod
    def array_to_blobproto(arr, diff=None):
        """
        Converts a N-dimensional array to blob proto.
        If diff is given, also convert the diff.
        You need to make sure that arr and diff have the same
        shape, and this function does not do sanity check.
        """
        blob = caffe_pb2.BlobProto()
        blob.shape.dim.extend(arr.shape)
        blob.data.extend(arr.astype(float).flat)
        if diff is not None:
            blob.diff.extend(diff.astype(float).flat)
        return blob

    @classmethod
    def get_list(cls, num, values):
        ans = []
        assert num > 0
        if len(values) < num:
            ans = [
                values[0],
            ] * num
        elif len(values) > num:
            for i in range(num):
                ans.append(values[i])
        else:
            ans = list(values)
        return ans
