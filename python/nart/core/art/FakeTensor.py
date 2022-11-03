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

# tianzhilong: class name underline too ugly
# tianzhilong: class name must be tuo2fe1ng style
from .Dtype import *
import numpy as np
import copy
from ... import _nart


class FakeTensor(object):
    """tensor in switch"""

    class Shape:
        """tensor shape"""

        def __init__(self, lst=None, channel_axis=1, batch_axis=0):
            if lst is None or len(lst) == 0:
                self.__dict__["dims"] = [1, 1, 1, 1]
            else:
                lst = list(lst)
                self.__dict__["dims"] = copy.deepcopy(lst)
            self.__dict__["channel_axis"] = channel_axis
            self.__dict__["batch_axis"] = batch_axis

        def __setattr__(self, k, v):
            if k in ["dims", "channel_axis", "batch_axis"]:
                raise

        def __repr__(self):
            return f"[FakeTensor.Shape(dims={self.dims}, channel_axis={self.channel_axis}, batch_axis={self.batch_axis})]"

    def __init__(
        self,
        dtype=Float32,
        shape=(
            [
                1,
                1,
                1,
                1,
            ],
            1,
            0,
        ),
        name=None,
        data=None,
    ):
        if not isinstance(shape, (tuple, list)):
            raise
        if not isinstance(shape[0], (tuple, list)):
            shape = (shape, 1, 0)
        if data is not None:
            ddd = np.ascontiguousarray(data, dtype=dtype)
            assert ddd.data.c_contiguous
            self.__dict__["true_obj"] = _nart.get_true_tensor(
                dtype, shape[0], shape[1], shape[2], ddd
            )
        else:
            self.__dict__["true_obj"] = _nart.get_true_tensor(
                dtype, shape[0], shape[1], shape[2]
            )
        # self.__dict__['shape'] = \
        #        FakeTensor.Shape(self.true_obj.shape_dims, self.true_obj.shape_channel_axis, self.true_obj.shape_batch_axis)
        if name is not None:
            self.name = name

    def __getattr__(self, k):
        if k == "dtype":
            return self.true_obj.dtype()
        if k == "shape":
            return FakeTensor.Shape(
                self.true_obj.shape_dims,
                self.true_obj.shape_channel_axis,
                self.true_obj.shape_batch_axis,
            )
        if k == "data":
            return self.true_obj.data()
        if k == "name":
            return self.true_obj.name
        return self.__dict__[k]

    def __setattr__(self, k, v):
        if k in ["true_obj", "shape", "data"]:
            raise
        if k == "name":
            self.true_obj.name = v

    def bind_true_tensor(self, true_tensor):
        if isinstance(true_tensor, _nart.FakeTensor):
            self.__dict__["true_obj"] = true_tensor

    def __repr__(self):
        return f"[FakeTensor(name={self.name}, dtype={self.dtype}, shape={self.shape}, data.size={self.data.size if self.data is not None else 0})]"
