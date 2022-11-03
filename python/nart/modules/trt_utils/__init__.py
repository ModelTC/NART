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

from .passes import CollectExtraOutput
from .environment import ParserContext, tensorrt_version
from .parse_utils import to_const_tensor
from .parse_utils import (
    get_nb_dims,
    ShapeTensor,
    reduce_mul,
    concat_shape,
    concat_shapes,
    get_shape,
    add_shuffle,
    flatten_tensor,
)
from .parse_utils import dtype_from_art_dtype, art_dtype_from_dtype
from .calibrator import create_calibrator


def init_parsers():
    from . import conv
    from . import pooling
    from . import gemm
    from . import activation
    from . import softmax
    from . import constant
    from . import reshape
    from . import slice_
    from . import element_wise
    from . import transpose
    from . import concat
    from . import batch_norm
    from . import global_pooling
    from . import prelu
    from . import upsample
    from . import conv_transpose
    from . import unary
    from . import shuffle_channel
    from . import cast
    from . import split
    from . import reduce
    from . import matmul
    from . import gather
    from . import topk
    from . import depth_to_space
    from . import expand
    from . import shape
    from . import resize
    from . import argmax
