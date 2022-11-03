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

from .op import Op
from .op import GroupType
from .op import PassThroughOp, BroadcastOp, Constant, Reshape
from .op import DataType
from .op import Mul
from .op import QuantDequant

from . import post_proc


class OpSetRegistry(dict):
    """A class that servers a common container for op-set handler registry,
    it is basically a dict from the composite key (op_type, domain, version) to any value.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def find(self, op_type, domain="", version=9, default=None):
        key = self.make_key(op_type, domain, version)
        return self.get(key, default)

    def insert(self, value, op_type: str = None, domain=None, version=None):
        if op_type is None:
            assert hasattr(
                value, "op_type"
            ), "op_type not given, and value doesn't have op_type attr"
            op_type = value.op_type
        # TODO: Is it safe to access _domain & _version?
        if domain is None:
            domain = value._domain
        if version is None:
            version = value._version
        key = (op_type, domain, version)
        self[key] = value

    def __getitem__(self, op_type):
        return self.find(op_type)

    @staticmethod
    def make_key(op_type, domain: str, version: int):
        return (op_type, domain, version)
