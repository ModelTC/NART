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

from contextlib import contextmanager
from importlib import import_module
import torch
import warnings

# get rid of the annoying fragmented reshape nodes
def size(self, *args, **kwargs):
    state = torch._C._get_tracing_state()
    if state:
        torch._C._tracer_abandon()
    result = torch.Tensor.ori_size(self, *args, **kwargs)
    if state:
        torch._C._set_tracing_state(state)
    return result


@property
def shape(self):
    return self.data.ori_shape


class convert_mode:
    def __init__(self):
        self.package = "nart.tools.pytorch.module_utils.pytorch1_1"

    def __enter__(self, *args, **kwargs):
        self.replace_symbolic()
        self.replace_jit()
        self.replace_size()
        self.replace_flip()
        self.replace_split()
        self.replace_lstmcell()
        self.replace_lstm()
        self.replace_gru()
        self.replace_interpolate()
        self.replace_group_norm()
        self.replace_deform()

    def __exit__(self, *args, **kwargs):
        self.restore_symbolic()
        self.restore_jit()
        self.restore_size()
        self.restore_flip()
        self.restore_split()
        self.restore_lstmcell()
        self.restore_lstm()
        self.restore_gru()
        self.restore_interpolate()
        self.restore_group_norm()
        self.restore_deform()

    def replace_symbolic(self):
        # for less redundancy, the re-written symbolic.py
        # only contains the part that needs to be overwritten
        # thus only that part will be substituted
        from torch.onnx import symbolic

        self.ori_symbolic_dict = symbolic.__dict__.copy()
        symbolic = import_module(".symbolic", package=self.package)
        for k, v in symbolic.__dict__.items():
            torch.onnx.symbolic.__dict__[k] = v

    def restore_symbolic(self):
        torch.onnx.symbolic.__dict__.clear()
        for k, v in self.ori_symbolic_dict.items():
            torch.onnx.symbolic.__dict__[k] = v

    def replace_jit(self):
        from torch.jit import LegacyTracedModule

        self.LegacyTracedModule = LegacyTracedModule
        self._iter_tensors = torch.autograd.function._iter_tensors
        jit = import_module(".jit", package=self.package)
        torch.jit.LegacyTracedModule = jit.LegacyTracedModule
        torch.autograd.function._iter_tensors = jit._iter_filter(
            lambda x: isinstance(x, torch.Tensor),
            condition_msg="Tensors",
            keyword="image",
        )

    def restore_jit(self):
        torch.jit.LegacyTracedModule = self.LegacyTracedModule
        torch.autograd.function._iter_tensors = self._iter_tensors

    def replace_size(self):
        torch.Tensor.ori_size = torch.Tensor.size
        torch.Tensor.size = size
        torch.Tensor.ori_shape = torch.Tensor.shape
        torch.Tensor.shape = shape

    def restore_size(self):
        torch.Tensor.size = torch.Tensor.ori_size
        delattr(torch.Tensor, "ori_size")
        torch.Tensor.shape = torch.Tensor.ori_shape
        delattr(torch.Tensor, "ori_shape")

    def replace_flip(self):
        self.ori_flip = torch.flip
        flip = import_module(".flip", package=self.package)
        torch.Tensor.ori_flip = torch.Tensor.flip
        torch.flip = flip.forward
        torch.Tensor.flip = flip.forward

    def restore_flip(self):
        torch.Tensor.flip = torch.Tensor.ori_flip
        torch.flip = self.ori_flip
        delattr(torch.Tensor, "ori_flip")

    def replace_split(self):
        self.ori_split = torch.split
        split = import_module(".split", package=self.package)
        torch.Tensor.ori_split = torch.Tensor.split
        torch.split = split.forward
        torch.Tensor.split = split.forward

    def restore_split(self):
        torch.Tensor.split = torch.Tensor.ori_split
        torch.split = self.ori_split
        delattr(torch.Tensor, "ori_split")

    def replace_lstmcell(self):
        self.ori_lstmcell_forward = torch.nn.LSTMCell.forward
        lstmcell = import_module(".lstmcell", package=self.package)
        torch.nn.LSTMCell.forward = lstmcell.forward

    def restore_lstmcell(self):
        torch.nn.LSTMCell.forward = self.ori_lstmcell_forward

    def replace_lstm(self):
        self.ori_lstm_forward = torch.nn.LSTM.forward
        lstm = import_module(".lstm", package=self.package)
        torch.nn.LSTM.forward = lstm.forward

    def restore_lstm(self):
        torch.nn.LSTM.forward = self.ori_lstm_forward

    def replace_gru(self):
        self.ori_gru_forward = torch.nn.GRU.forward
        gru = import_module(".gru", package=self.package)
        torch.nn.GRU.forward = gru.forward

    def restore_gru(self):
        torch.nn.GRU.forward = self.ori_gru_forward

    def replace_deform(self):
        try:
            from pod import extensions

            self.ori_deform_forward = extensions.DeformableConv.forward
            deform = import_module(".deform", package=self.package)
            extensions.DeformableConv.forward = deform.forward
        except ImportError:
            pass

    def restore_deform(self):
        try:
            from pod import extensions

            extensions.DeformableConv.forward = self.ori_deform_forward
        except ImportError:
            pass

    def replace_interpolate(self):
        interpolate = import_module(".interpolate", package=self.package)
        torch.nn.functional.ori_interpolate = torch.nn.functional.interpolate
        torch.nn.functional.interpolate = interpolate.forward

    def restore_interpolate(self):
        torch.nn.functional.interpolate = torch.nn.functional.ori_interpolate
        delattr(torch.nn.functional, "ori_interpolate")

    def replace_group_norm(self):
        group_norm = import_module(".group_norm", package=self.package)
        torch.nn.functional.ori_group_norm = torch.nn.functional.group_norm
        torch.nn.functional.group_norm = group_norm.forward

    def restore_group_norm(self):
        torch.nn.functional.group_norm = torch.nn.functional.ori_group_norm
        delattr(torch.nn.functional, "ori_group_norm")
