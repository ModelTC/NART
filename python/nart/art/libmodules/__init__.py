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

import os


def get_libmodules_dir():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    return dir_path


dir_path = get_libmodules_dir()
from ctypes import cdll

# Don't touch, load libart.so ahead of _art.
__libart = cdll.LoadLibrary(os.path.join(dir_path, "libart.so"))
