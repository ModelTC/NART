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

from nart.utils.onnx_utils.onnx_builder import OnnxDecoder
from nart.utils.alter import CaffeAlter
from nart.core import Model
from nart.passes import ConstantToInitializer, DeadCodeElimination, ExtractEltwise
import onnx
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert onnx to caffe")
    parser.add_argument("onnx", help="onnx file path")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="caffe",
        help="output caffe model path, [name].prototxt and [name].caffemodel",
    )

    arg = parser.parse_args()
    onf = None
    with open(arg.onnx, "rb") as f:
        onf = onnx.load(f)
    x = OnnxDecoder()
    graph = x.decode(onf)

    ConstantToInitializer().run(graph)
    ExtractEltwise().run(graph)
    DeadCodeElimination().run(graph)

    alter = CaffeAlter(graph)
    caffemodel = alter.parse()

    from copy import deepcopy

    netdef = deepcopy(caffemodel)
    for layer in netdef.layer:
        del layer.blobs[:]
    from .utils import write_netdef, write_model

    write_netdef(netdef, f"{arg.output}.prototxt")
    # write_model(caffemodel, f"{arg.output}.caffemodel")
    with open(f"{arg.output}.caffemodel", "wb") as f:
        byte = f.write(caffemodel.SerializeToString())
        print(
            "save caffe model to (%s) with size %d" % (f"{arg.output}.caffemodel", byte)
        )
