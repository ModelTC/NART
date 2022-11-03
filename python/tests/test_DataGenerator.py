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

import net_bed
import numpy as np

data_num = 10

net_def = net_bed.read_net_def("./model/test.prototxt")
caffe_net = net_bed.CaffeNet("test", net_def)
caffe_net.load({"weight_path": "./model/test.model"})

data_generator = net_bed.DataGenerator(caffe_net)
data_generator.set_config(data_num=data_num, rand_input=True)
data_generator.prepare()
print(data_generator.fetch_data("data"))

for path in data_generator.fetch_data("data"):
    data = np.fromfile(path)
    print(path)
    print(data)

print(data_generator.fetch_data("184"))

for path in data_generator.fetch_data("184"):
    data = np.fromfile(path)
    print(path)
    print(data)


print(data_generator.fetch_data("184"))

for path in data_generator.fetch_data("184"):
    data = np.fromfile(path)
    print(path)
    print(data)


print(data_generator.fetch_data("333"))

for path in data_generator.fetch_data("333"):
    data = np.fromfile(path)
    print(path)
    print(data)
