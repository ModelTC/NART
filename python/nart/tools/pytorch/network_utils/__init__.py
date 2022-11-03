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
from .network import Network
from .batchnorm import merge_batchnorm_nodes
from .common import update_input_names, update_output_names
from .convolution import merge_convolution_nodes
from .deconvolution import (
    merge_biased_deconvolution_nodes,
    merge_nonbiased_deconvolution_nodes,
)
from .eltwise import merge_eltwise_nodes
from .reshape import merge_reshape_nodes, delete_reshape_nodes
from .shuffle_channel import merge_shuffle_channel_nodes
from .slice import merge_slice_nodes, remove_slice_nodes, complement_slice_nodes
from .lstm_unit import merge_lstmcell_nodes
from .sllstm import merge_lstm_nodes
from .slgrnn import merge_gru_nodes
from .linear import remove_constant_nodes, merge_gemm_nodes
from .axpy import merge_axpy_nodes
from .unknown import remove_cast_nodes
from .dropout import delete_dropout_nodes
from .concat import delete_concat_nodes
from .expand import delete_expand_nodes

__all__ = ["make_network"]


def make_network(model, name, input_names, output_names):
    if input_names is not None:
        update_input_names(model, input_names)
    if output_names is not None:
        update_output_names(model, output_names)
    remove_cast_nodes(model)
    if torch.__version__[:3] == "0.3":
        merge_biased_deconvolution_nodes(model)
        merge_nonbiased_deconvolution_nodes(model)
        merge_batchnorm_nodes(model)
        merge_convolution_nodes(model)
        remove_constant_nodes(model)
    elif torch.__version__[:5] == "0.4.0":
        merge_gemm_nodes(model)
    elif torch.__version__[:5] == "0.4.1":
        merge_reshape_nodes(model)
        merge_gemm_nodes(model)
    elif torch.__version__[0] == "1":
        # remove_slice_nodes(model)
        merge_reshape_nodes(model)
        merge_batchnorm_nodes(model)
        merge_lstmcell_nodes(model)
        merge_lstm_nodes(model)
        merge_gru_nodes(model)
        delete_expand_nodes(model)
        # complement_slice_nodes(model)

        merge_gemm_nodes(model)
    # merge_slice_nodes(model)
    merge_shuffle_channel_nodes(model)
    merge_eltwise_nodes(model)
    #  merge_axpy_nodes(model)

    delete_concat_nodes(model)
    delete_dropout_nodes(model)
    delete_reshape_nodes(model)
    return Network(model, name)
