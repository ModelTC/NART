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


def update_input_names(model, input_names):
    """
    Substitute the original blob names in the model
    with assigned names for input
    """
    initial_blobnames = [blob.name for blob in model.graph.initializer]
    inps = [inp for inp in model.graph.input if inp.name not in initial_blobnames]
    assert len(inps) == len(
        input_names
    ), "{} required but {} given: each input blob should have exactly one name.".format(
        len(inps), len(input_names)
    )
    input_to_name = dict(zip([inp.name for inp in inps], input_names))

    for blob in model.graph.initializer:
        if blob.name in input_to_name.keys():
            blob.name = input_to_name[blob.name]

    for inp in model.graph.input:
        if inp.name in input_to_name.keys():
            inp.name = input_to_name[inp.name]

    for node in model.graph.node:
        for idx, bottom in enumerate(node.input):
            if bottom in input_to_name.keys():
                node.input[idx] = input_to_name[bottom]

        for idx, top in enumerate(node.output):
            if top in input_to_name.keys():
                node.output[idx] = input_to_name[top]


def update_output_names(model, output_names):
    """
    Substitute the original blob names in the model
    with assigned names for output
    """
    outputs = [output for output in model.graph.output]
    assert len(outputs) == len(
        output_names
    ), "{} required but {} given: each output blob should have exactly one name.".format(
        len(outputs), len(output_names)
    )
    output_to_name = dict(zip([output.name for output in outputs], output_names))

    for blob in model.graph.initializer:
        if blob.name in output_to_name.keys():
            blob.name = output_to_name[blob.name]

    for node in model.graph.node:
        for idx, bottom in enumerate(node.input):
            if bottom in output_to_name.keys():
                node.input[idx] = output_to_name[bottom]

        for idx, top in enumerate(node.output):
            if top in output_to_name.keys():
                node.output[idx] = output_to_name[top]

    for output in model.graph.output:
        if output.name in output_to_name.keys():
            output.name = output_to_name[output.name]
