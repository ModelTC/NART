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

from queue import LifoQueue

_ctx = []


class Context:
    def __init__(self):
        self._container = {}

    def __enter__(self):
        _ctx.append(self)
        return self

    def __exit__(self):
        _ctx.remove(self)

    @staticmethod
    def current():
        return _ctx[-1]

    def set(self, k, v):
        self._container[k] = v

    def get(self, k):
        return self._container.get(k, None)
