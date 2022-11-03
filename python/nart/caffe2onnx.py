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

from nart.utils.caffe_utils.caffe_builder import CaffeDecoder
from nart.core import Model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert caffe to onnx")
    parser.add_argument("prototxt", help="caffe prototxt path")
    parser.add_argument("caffemodel", help="caffe modelpath")
    parser.add_argument(
        "-o", "--output", type=str, default="model.onnx", help="output onnx model path"
    )
    arg = parser.parse_args()

    import logging

    logger = logging.getLogger("nart")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    x = CaffeDecoder()
    graph = x.load(arg.prototxt, arg.caffemodel)
    graph.update_topology()
    graph.update_tensor_shape()

    from .passes import *

    passes = get_standardize_op_passes()
    passes.extend([ConstantToInitializer(), DeadCodeElimination()])
    for item in passes:
        item.run(graph)

    m = Model.make_model(graph)
    onnx = m.dump_to_onnx()
    with open(arg.output, "wb") as f:
        byte = f.write(onnx.SerializeToString())
        print("save onnx model to (%s) with size %d" % (arg.output, byte))
