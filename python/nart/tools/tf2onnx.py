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
from nart.core import Model
from nart.passes import Pass
import argparse
import onnx


class Rename(Pass):
    """
    rename DNN model layer's name
    """

    def __init__(self):
        pass

    def run(self, graph):
        for node in list(graph.nodes):
            # modify layer name
            name = node.name
            if "/" in name:
                new_name = name.replace("/", "")
                new_node = []
                new_op = Op.make_op(
                    node.op_type, new_name, node.input, node.output, node.attributes
                )
                new_node.append(new_op)
                pos = graph.nodes.index(node)
                graph.del_node_purely(node)
                graph.insert_nodes_purely(new_node, pos=pos)
        graph.update_topology()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert tensorflow to onnx")
    parser.add_argument("--opset", type=str, help="optimization mode setting")
    parser.add_argument("--tflite", type=str, help="tensorflow model path")
    parser.add_argument(
        "--output", type=str, default="model.onnx", help="output onnx model path"
    )
    arg = parser.parse_args()

    try:
        import tf2onnx
    except Exception as e:
        logger.fatal(str(e))
        logger.fatal("Can not import tf2onnx, try install using pip install tf2onnx")
    tflite = arg.tflite
    output = arg.output
    opset = arg.opset
    cmd = (
        "python3 -m tf2onnx.convert "
        f" --opset={opset}"
        f" --tflite={tflite}"
        f" --output={output}"
    )
    from subprocess import PIPE, Popen

    p = Popen(cmd, shell=True)
    p.communicate()

    onnx_model = None
    with open(output, "rb") as f:
        onnx_model = onnx.load(f)

    x = OnnxDecoder()
    graph = x.decode(onnx_model)

    from ..passes import *

    passes = [
        RemoveInitializerDequant(),
        RemoveBackTOBackQuantDequant(),
        RemoveInputDequantOutputQuant(),
        RemoveExtraQuantDequant(),
        Rename(),
    ]
    for item in passes:
        item.run(graph)

    m = Model.make_model(graph)
    onnx = m.dump_to_onnx()
    with open(output, "wb") as f:
        byte = f.write(onnx.SerializeToString())
        print("save onnx model to (%s) with size %d" % (output, byte))
