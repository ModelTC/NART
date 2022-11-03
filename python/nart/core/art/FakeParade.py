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

from . import FakeOp
from ... import _nart
from . import FakeTensor
import logging

logger = logging.getLogger("nart.core.parade")


class FakeParade(object):
    """parade in switch"""

    def __init__(self):
        self.ops = []
        self.tensors = []
        self.marked_output_tensors = []
        self.true_obj = _nart.FakeParade()

    def append(self, op):
        """append an op into FakeParade.ops.

        Args:
            op: FakeOp or _nart.FakeOpInterface.

        Returns:
            list of FakeTensor, which is new Op's output tensors.
        """

        res_m = list(map(lambda x: x.true_obj, self.tensors))
        for ipt in op.inputs:
            if ipt not in res_m:
                tensor = FakeTensor()
                tensor.bind_true_tensor(ipt)
                self.tensors.append(tensor)

        assert isinstance(op, (FakeOp, _nart.FakeOpInterface))
        assert op not in self.ops
        outputs = self.true_obj.append(op)
        # outputs = nart._nart.infer_shape(op)
        res = []
        for item in outputs:
            tensor = FakeTensor()
            tensor.bind_true_tensor(item)
            res.append(tensor)
            self.tensors.append(tensor)
        self.ops.append(op)
        return res

    @property
    def infered_output(self):
        """parade's infere output(without marked output)

        Returns:
            list of FakeTensor.
        """
        res = list(self.tensors)
        res_m = list(map(lambda x: x.true_obj, self.tensors))
        for op in self.ops:
            rm_lst = []
            for t, tt in zip(res, res_m):
                if tt in op.inputs:
                    rm_lst.append((t, tt))
            for t, tt in rm_lst:
                res.remove(t)
                res_m.remove(tt)
        return res

    @property
    def outputs(self):
        """all outputs include marked outputs"""
        res = list(self.marked_output_tensors)
        for item in self.infered_output:
            if item not in res:
                res.append(item)
        return res

    def mark_as_output(self, tensors):
        if not isinstance(tensors, (list, tuple)):
            tensors = [
                tensors,
            ]
        tensors = list(
            map(lambda x: x.true_obj if hasattr(x, "true_obj") else x, tensors)
        )
        for tensor in tensors:
            if (
                tensor in map(lambda x: x.true_obj, self.tensors)
                and tensor not in self.marked_output_tensors
            ):
                self.true_obj.mark_as_output(tensor)
                self.marked_output_tensors.append(tensor)

    def mark_as_output_by_name(self, tensor_names):
        """mark tensor as output by name

        Args:
            tensor_names: str.
        """
        if not isinstance(tensor_names, (list, tuple)):
            tensor_names = [
                tensor_names,
            ]

        # change name to tensor
        tensors = []
        for name in tensor_names:
            for t in reversed(self.tensors):
                if t.name == name:
                    tensors.append(t)
                    break

        tensors = list(
            map(lambda x: x.true_obj if hasattr(x, "true_obj") else x, tensors)
        )
        for tensor in tensors:
            if (
                tensor in map(lambda x: x.true_obj, self.tensors)
                and tensor not in self.marked_output_tensors
            ):
                self.true_obj.mark_as_output(tensor)
                self.marked_output_tensors.append(tensor)
                logger.debug(f"mark `{tensor.name}` as an output of this parade")

    def serialize(self):
        """serialize parade

        Returns:
            a  bin model which can run in nart-case
        """
        return self.true_obj.serialize()

    def bind_transformer(self, tensor, transformer: _nart.FakeTransformer):
        assert isinstance(tensor, (str, FakeTensor, _nart.FakeTensor))
        if isinstance(tensor, str):
            return self.true_obj.bind_transformer(tensor, transformer)
        return self.true_obj.bind_transformer(tensor.name, transformer)
