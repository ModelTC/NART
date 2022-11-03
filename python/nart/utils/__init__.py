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

from .data_generator import DataGenerator, PerFrameDataGenerator
from .splitter import Splitter
from .utils import write_netdef
from .utils import write_model
from .utils import read_net_def
from .utils import generate_support_table, dump_support_info_to_csv

import logging

logger = logging.getLogger("nart")


def deprecated(new_fn=None):
    def inner_wrapper(fn):
        def wrapper(*args, **kwargs):
            if new_fn is not None:
                logger.warning(
                    f'`{fn.__module__ + "." + fn.__name__}` is deprecated, please use `{new_fn.__module__ + "." + new_fn.__name__}` instead.'
                )
            else:
                logger.warning(f'`{fn.__module__ + "." + fn.__name__}` is deprecated.')
            return fn(*args, **kwargs)

        return wrapper

    return inner_wrapper
