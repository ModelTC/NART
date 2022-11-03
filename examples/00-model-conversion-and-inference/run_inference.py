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

# %% [markdown]
# # Run inference
#
# This file runs inference of the NART model converted by `nart.switch`.

# %%
engine_bin_path = "tmp/engine.bin"
engine_config_path = "tmp/engine.bin.json"

# %%
image_path = "data/drum.jpg"
labels_path = "data/imagenet_classes.txt"
expected_label = "drum"

# %%
from PIL import Image
import numpy as np

from nart.art import load_parade, create_context_from_json, get_empty_io

# %%
# read labels
with open(labels_path) as f:
    labels = list(map(str.strip, f.readlines()))

# %%
# prepare the input data
img = Image.open(image_path).resize((256, 256)).crop((16, 16, 240, 240))
input_data = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
input_data -= np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
input_data /= np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
input_data = input_data.reshape((1, 3, 224, 224))

# %%
# load the NART model (parade)
# may take some time if the target is cuda
ctx = create_context_from_json(engine_config_path)
parade = load_parade(engine_bin_path, ctx)

# %%
# create placeholder for inputs and outputs of `parade`
inputs, outputs = get_empty_io(parade)
assert len(inputs) == 1 and len(outputs) == 1

input_array = inputs[next(iter(inputs.keys()))]
output_array = outputs[next(iter(outputs.keys()))]

# %%
# setup the input data
input_array[:] = input_data

# %%
# run inference
res = parade.run(inputs, outputs)
print(res)
assert res

# %%
# verify the predicted result
predict_res = np.argmax(output_array, axis=1)
predict_labels = [labels[i] for i in predict_res.tolist()]
print(predict_labels)
assert predict_labels[0] == expected_label
