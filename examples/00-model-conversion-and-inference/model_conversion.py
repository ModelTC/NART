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
# # Model Conversion
#
# **To run this file requires PyTorch LTS 1.8 and TorchVision**
#
# This file converts the pretrained ResNet50 PyTorch model
# into either a Caffe model or an ONNX model.

# %%
target_model_type = "onnx"
# or convert the model to caffe
# target_model_type = "caffe"

# %%
image_path = "data/drum.jpg"
labels_path = "data/imagenet_classes.txt"
expected_label = "drum"

tmp_model_base_name = "tmp/model"

# %% [markdown]
# Test image:
# ![drum](data/drum.jpg)

# %%
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.models import resnet50

from nart.tools.pytorch import export_onnx, convert_v2
from nart.tools.pytorch.module_utils import convert_mode

# %%
torch.__version__, torchvision.__version__

# %%
Path(tmp_model_base_name).parent.mkdir(parents=True, exist_ok=True)

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
input_data = torch.from_numpy(input_data)
print(input_data.shape)

# %%
# use the pretrained model, download it if it does not exists
model = resnet50(pretrained=True)
model.eval()

# %%
# verify the predicted result
output_data = model(input_data)
predict_res = output_data.argmax(axis=1)
predict_labels = [labels[i] for i in predict_res.tolist()]
print(predict_labels)
assert predict_labels[0] == expected_label

# %%
if target_model_type.lower() == "onnx":
    convert_func = export_onnx
else:
    convert_func = convert_v2

# %%
# convert the model
with convert_mode():
    convert_func(
        model,
        [tuple(input_data.shape)],
        filename=tmp_model_base_name,
        log=True,
        verbose=True,
        cloze=False,
    )
