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

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from nart.modules import CaffeParser
from nart import serialize_v1
from nart.utils.onnx_utils import OnnxDecoder
import onnx

CaffeParser.register_defaults()

model = onnx.load("whatever.onnx")
net = OnnxDecoder().decode(model.graph)
print(net)
print(net._tensor_shape)
net.update_tensor_shape()
print(net._tensor_shape)

parade = CaffeParser(net).parse()
ser = serialize_v1(parade)

with open("engine.bin", "wb") as f:
    f.write(ser)
