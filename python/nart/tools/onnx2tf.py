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

import onnx
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert tensorflow to onnx")
    parser.add_argument("--onnx", type=str, help="onnx model path")
    parser.add_argument(
        "--output", type=str, default="model.pb", help="output tensorflow model path"
    )
    arg = parser.parse_args()

    try:
        from onnx_tf.backend import prepare
    except Exception as e:
        logger.fatal(str(e))
        logger.fatal("Can not import onnx2tf, try install using pip install onnx_tf")
    onnx_model = arg.onnx
    output = arg.output
    model = onnx.load(onnx_model)
    tf_rep = prepare(model, strict=False)
    tf_rep.export_graph(output)
    print("save tensorflow model to %s" % output)
