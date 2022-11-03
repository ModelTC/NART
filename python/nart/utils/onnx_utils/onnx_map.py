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

from onnx import TensorProto


map_onnx_to_art_dtype = {
    TensorProto.FLOAT: 10,  # float32
    TensorProto.UINT8: 5,  # uint8_t
    TensorProto.INT8: 1,  # int8_t
    TensorProto.UINT16: 6,  # uint16_t
    TensorProto.INT16: 2,  # int16_t
    TensorProto.INT32: 3,  # int32_t
    TensorProto.INT64: 4,  # int64_t
    TensorProto.STRING: 12,  # string
    TensorProto.BOOL: 15,  # bool
    TensorProto.FLOAT16: 9,  # float16
    TensorProto.DOUBLE: 11,  # double
    TensorProto.UINT32: 7,  # uint32_t
    TensorProto.UINT64: 8,  # uint64_t
    #  Future extensions go here.
}
