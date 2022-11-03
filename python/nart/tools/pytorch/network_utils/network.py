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

import google.protobuf.text_format as text_format
from ....proto import caffe_pb2
from ..onnx_utils import to_array
from ..layer_utils import make_layers
import json
import onnx


class Network:
    def __init__(self, model, name):
        self.params = caffe_pb2.NetParameter(name=name)
        self.model = model
        self.parse_blobs()
        self.parse_input()
        self.pytorch_map = self.parse_layers()

    def parse_blobs(self):
        self.blobs = dict()
        for blob in self.model.graph.initializer:
            self.blobs[blob.name] = to_array(blob)

        self.blobshape = dict()
        for inp in self.model.graph.input:
            if inp.name not in self.blobs.keys():
                self.blobshape[inp.name] = [
                    d.dim_value for d in inp.type.tensor_type.shape.dim
                ]

    def parse_input(self):
        inps = [
            inp for inp in self.model.graph.input if inp.name not in self.blobs.keys()
        ]
        self.params.input.extend([inp.name for inp in inps])
        for inp in inps:
            inp_shape = caffe_pb2.BlobShape()
            inp_shape.dim.extend([d.dim_value for d in inp.type.tensor_type.shape.dim])
            self.params.input_shape.extend([inp_shape])

    def parse_layers(self):
        pytorch_map = {}
        for index, node in enumerate(self.model.graph.node):
            layers = make_layers(node, str(index), self)
            for layer in layers:
                self.params.layer.extend([layer.params])
                attributes = dict(
                    zip([attr.name for attr in node.attribute], node.attribute)
                )
                if "module_name" in attributes:
                    pytorch_map[layer.params.name] = attributes["module_name"].s.decode(
                        "utf-8"
                    )
        return pytorch_map

    def save(self, verbose=False, output_weights=True):
        if verbose:
            print("Blobs dataflow of network {}:".format(self.params.name))
            for inp, inp_shape in zip(self.params.input, self.params.input_shape):
                shape = [int(dim[5:]) for dim in str(inp_shape).strip().split("\n")]
                print("Input blob '{}' with shape {}".format(inp, shape))
            output_names = set([output.name for output in self.model.graph.output])
            for layer in self.params.layer:
                for top in layer.top:
                    if top in output_names:
                        print(
                            "Output blob '{}' with shape {} as top of layer {}".format(
                                top, self.blobshape[top], layer.name
                            )
                        )
                    else:
                        print(
                            "Blob '{}' with shape {} as top of layer {}".format(
                                top, self.blobshape[top], layer.name
                            )
                        )

        proto = caffe_pb2.NetParameter()
        proto.CopyFrom(self.params)
        for layer in proto.layer:
            del layer.blobs[:]

        with open(self.params.name + ".prototxt", "w") as fout:
            fout.write(text_format.MessageToString(proto))

        if output_weights:
            with open(self.params.name + ".caffemodel", "wb") as fout:
                fout.write(self.params.SerializeToString())

        if verbose:
            # save model with module_name attribute
            onnx.save(self.model, self.params.name + "_mod.onnx")
            # save mapping from caffe layer name to pytorch module name
            with open(self.params.name + "_map.json", "w") as map_file:
                json.dump(self.pytorch_map, map_file, indent=4)
