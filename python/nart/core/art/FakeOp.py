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

from ... import _nart
from . import FakeTensor
from inspect import signature
import numpy as np


class FakeOp(object):
    """Op in switch"""

    def __init__(self, op_tp: str, inputs, params=[]):
        assert isinstance(op_tp, str)
        self.op_tp = op_tp
        self.inputs = list(inputs)
        self.params = list(params)

    def __repr__(self):
        return str(self.__dict__)

    class FakeOpInterface(_nart.FakeOpInterface):
        class FakeSetting:
            def __init__(self, name, dtype):
                self.name = name
                self.dtype = dtype

        class FakeSettingSingle(FakeSetting):
            def __init__(self, name, dtype, v):
                super(FakeSettingSingle, self).__init__(name, dtype)
                self.v = v

        class FakeSettingRepeated(FakeSetting):
            def __init__(self, name, dtype, v):
                super(FakeSettingSingle, self).__init__(name, dtype)
                self.v = v

        def __init__(self, inputs, params=None):
            if params is None:
                super(FakeOp.FakeOpInterface, self).__init__(
                    self, list(map(lambda x: x.true_obj, inputs))
                )
            else:
                super(FakeOp.FakeOpInterface, self).__init__(
                    self,
                    list(map(lambda x: x.true_obj, inputs)),
                    list(map(lambda x: x.true_obj, params)),
                )
            self._inputs = inputs
            self._params = params

        def _op_tp_code(self):
            if not hasattr(self, "op_tp_code"):
                raise NotImplementedError(
                    f"class {self.__class__} should implement function op_tp_code."
                )
            return self.op_tp_code()

        def _infer_shape(self):
            if not hasattr(self, "infer_shape"):
                raise NotImplementedError(
                    f"class {self.__class__} should implement function infer_shape."
                )
            if 2 == len(signature(self.infer_shape).parameters):
                return self.infer_shape(self._inputs, self._params)
            if 1 == len(signature(self.infer_shape).parameters):
                return self.infer_shape(self._inputs)
            raise

        def add_setting(self, item, dtype, obj):
            if not hasattr(self, "_settings"):
                self._settings = []
            if dtype == "string":
                dtype = "str"
            repeat = False
            if isinstance(obj, list):
                obj = np.array(obj, dtype=dtype)
                repeat = True
            if isinstance(obj, bytes):
                obj = np.frombuffer(obj, dtype=dtype)
                repeat = True
            if isinstance(obj, np.ndarray):
                obj = np.ascontiguousarray(obj, dtype=dtype)
                repeat = True
            if dtype == "str":
                dtype = "string"
            self._settings.append((item, dtype, repeat, obj))

        def set_group_code(self, group):
            op_code = self.op_tp_code()
            if (op_code >> 32) & 0xFFFFFFFF == 0:
                op_code = group << 32 | op_code

                def true_op_code():
                    return op_code

                self.op_tp_code = true_op_code
            else:
                raise
