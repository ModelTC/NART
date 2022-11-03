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

from .tanh import *
from .eltwise import *
from .softmax import *
from .concat import *
from .reshape import *
from .deconvolution import *
from .argmax import *
from .inner_product import *
from .prelu import *
from .slice import *
from .convolution import *
from .sigmoid import *
from .relu import *
from .slgrnn import *
from .lstm_unit import *
from .sllstm import *

# FIXME: some tests are disabled because of using internal CaffeInfer
# FIXME: please update them to support official Caffe
# from .batchnorm import *
# from .transpose import *
# from .relu6 import *
# from .scale import *
# from .nninterp import *
# from .pixel_shuffle import *
# from .pooling import *
# from .interp import *
# from .dropout import *
# from .groupnorm import *
# from .holeconvolution import *
# from .shuffle_channel import *
