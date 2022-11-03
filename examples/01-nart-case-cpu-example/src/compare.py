#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import numpy as np


def run():
    a = np.fromfile("data/input/input1.bin", dtype="float32")
    b = np.fromfile("data/input/input2.bin", dtype="float32")
    return a + b


def cos_dis(a, b):
    a = a.flatten().astype("float32")
    b = b.flatten().astype("float32")
    u = np.sum(a * b)
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return u / d


c = np.fromfile("data/output/add_out.bin", dtype="float32")
res = cos_dis(c, run())
print(res)
assert res >= 0.999
