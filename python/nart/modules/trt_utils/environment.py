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

""" Some misc environment related classes.
"""


def tensorrt_version():
    import tensorrt
    from packaging import version

    return version.parse(tensorrt.__version__)


class ParserContext:
    current = None

    def __init__(self, network, graph):
        # the tensorrt network being constructed.
        self.network = network
        # the Graph
        self.graph = graph
        # usually the node being parsed.
        self.cur_node = None
        # a map from tensor name to corresponding itensor.
        self.itensor_by_name = dict()
        self.old = None

    def __enter__(self):
        self.old = ParserContext.current
        ParserContext.current = self

    def __exit__(self, type, value, trace):
        ParserContext.current = self.old

    @staticmethod
    def get_current():
        return ParserContext.current
