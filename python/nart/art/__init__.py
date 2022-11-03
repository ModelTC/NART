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

from typing import List, Tuple
from .libmodules import get_libmodules_dir
from . import _art
from ._art import RuntimeContext, Parade

"""
for example:
    ctx = create_context_from_json(/some/path/to/xxx.bin.json)
    parade = load_parade(/some/path/to/xxx.bin)
    inputs, outpus = get_empty_io(parade)
    inputs[xxx][:] = yourdata
    parade.run(inputs, outputs)
"""
_art.set_package_dir(get_libmodules_dir())


def create_context_from_json(jsonPath: str) -> RuntimeContext:
    import json

    ws_data = None
    with open(jsonPath, "r") as fp:
        ws_data = json.loads(fp.read())
    ctx = RuntimeContext(list(ws_data["workspaces"].keys()), ws_data["input_workspace"])
    return ctx


def create_context_for_target(target: str) -> RuntimeContext:
    MODULES_DICT = {
        "default": ["default"],
        "cuda": ["default", "cuda"],
        "tensorrt": ["default", "cuda", "tensorrt"],
    }
    ctx = RuntimeContext(MODULES_DICT[target], target)
    return ctx


def load_parade(engine_fp: str, ctx: RuntimeContext) -> Parade:
    with open(engine_fp, "rb") as fp:
        data = fp.read()
        parade = Parade(data, ctx)
        return parade


def get_empty_io(parade: Parade) -> Tuple[dict, dict]:
    import numpy

    input_shapes = parade.input_shapes()
    inputs = {}
    for k, v in input_shapes.items():
        dtype = parade.get_tensor_dtype(k)
        dtype = numpy.dtype(dtype)
        inputs[k] = numpy.zeros(v, dtype=dtype)
    output_shapes = parade.output_shapes()
    outputs = {}
    for k, v in output_shapes.items():
        dtype = parade.get_tensor_dtype(k)
        dtype = numpy.dtype(dtype)
        outputs[k] = numpy.zeros(v, dtype=dtype)
    return inputs, outputs
