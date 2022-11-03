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

import torch
from importlib import import_module

major, minor, patch = torch.__version__[:5].split(".")
package = "nart.tools.pytorch.module_utils"
if major == "0" and minor == "4":
    module = import_module(
        ".pytorch" + "_".join((major, minor, patch)), package=package
    )
elif major == "0" and minor == "3":
    module = import_module(".pytorch" + "_".join((major, minor)), package=package)
elif major == "1" and (minor == "0" or minor == "1"):
    module = import_module(".pytorch" + "_".join((major, minor)), package=package)
elif major == "1" and (minor == "2" or minor == "3" or minor == "5" or minor == "8"):
    module = import_module(".pytorch" + "_".join((major, minor)), package=package)
else:
    raise NotImplementedError("Unsupported torch version " + torch.__version__)
convert_mode = module.convert_mode

__all__ = ["convert_mode"]
