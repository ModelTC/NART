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
import types
import os


class convert_mode:
    def __init__(self):
        self.package = "nart.tools.pytorch.module_utils.pytorch0_3"

    def __enter__(self, *args, **kwargs):
        self.replace_symbolic()
        self.replace_jit()
        self.replace_flip()
        self.replace_batch_norm()
        self.replace_upsample()
        self.replace_prelu()
        self.replace_lstmcell()
        self.replace_lstm()
        self.replace_gru()

    def __exit__(self, *args, **kwargs):
        self.restore_symbolic()
        self.restore_jit()
        self.restore_flip()
        self.restore_batch_norm()
        self.restore_upsample()
        self.restore_prelu()
        self.restore_lstmcell()
        self.restore_lstm()
        self.restore_gru()

    def replace_symbolic(self):
        # for less redundancy, the re-written symbolic.py
        # only contains the part that needs to be overwritten
        # thus only that part will be substituted
        from torch.onnx import symbolic

        self.symbolic_dict = symbolic.__dict__.copy()
        symbolic = import_module(".symbolic", package=self.package)
        for k, v in symbolic.__dict__.items():
            torch.onnx.symbolic.__dict__[k] = v

    def restore_symbolic(self):
        torch.onnx.symbolic.__dict__.clear()
        for k, v in self.symbolic_dict.items():
            torch.onnx.symbolic.__dict__[k] = v

    def replace_jit(self):
        from torch.jit import TracedModule

        self.TracedModule = TracedModule
        jit = import_module(".jit", package=self.package)
        torch.jit.TracedModule = jit.TracedModule

    def restore_jit(self):
        torch.jit.TracedModule = self.TracedModule

    def replace_flip(self):
        flip = import_module(".flip", package=self.package)
        torch.flip = flip.forward
        torch.Tensor.flip = flip.forward
        torch.autograd.Variable.flip = flip.forward

    def restore_flip(self):
        delattr(torch, "flip")
        delattr(torch.Tensor, "flip")
        delattr(torch.autograd.Variable, "flip")

    def replace_batch_norm(self):
        from torch.nn.functional import batch_norm

        self.batch_norm = batch_norm
        functional = import_module(".functional", package=self.package)
        torch.nn.functional.batch_norm = functional.batch_norm

    def restore_batch_norm(self):
        torch.nn.functional.batch_norm = self.batch_norm

    def replace_upsample(self):
        upsample = import_module(".upsample", package=self.package)
        torch.nn.functional.ori_upsample = torch.nn.functional.upsample
        torch.nn.functional.upsample = upsample.forward

    def restore_upsample(self):
        torch.nn.functional.upsample = torch.nn.functional.ori_upsample
        delattr(torch.nn.functional, "ori_upsample")

    def replace_prelu(self):
        # the re-written activation.py has relative import,
        # thus it has to be dynamically compiled and substituted
        # to avoid the error triggered by relative path tracking
        from torch.nn._functions.thnn import PReLU

        self.PReLU = PReLU

        dir_path = os.path.dirname(os.path.realpath(__file__))
        activation = types.ModuleType("torch.nn._functions.thnn.PReLU")
        with open(dir_path + "/activation.py", "r") as fin:
            code = compile(fin.read(), "activation.py", "exec")
            exec(code, activation.__dict__)
        # skip the alias in torch.nn.functional.thnn.upsampling
        # directly replace the original module in _functions
        torch.nn.functional._functions.thnn.PReLU = activation.PReLU

    def restore_prelu(self):
        torch.nn.functional._functions.thnn.PReLU = self.PReLU

    def replace_lstmcell(self):
        torch.nn.backends.thnn.backend._LSTMCell = (
            torch.nn.backends.thnn.backend.LSTMCell
        )
        lstmcell = import_module(".lstmcell", package=self.package)
        torch.nn.backends.thnn.backend.LSTMCell = lstmcell.lstm_cell

    def restore_lstmcell(self):
        torch.nn.backends.thnn.backend.LSTMCell = (
            torch.nn.backends.thnn.backend._LSTMCell
        )

    def replace_deform(self):
        from pod import extensions

        self.ori_deform_forward = extensions.DeformableConv.forward
        deform = import_module(".deform", package=self.package)
        extensions.DeformableConv.forward = deform.forward

    def restore_deform(self):
        from pod import extensions

        extensions.DeformableConv.forward = self.ori_deform_forward

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
