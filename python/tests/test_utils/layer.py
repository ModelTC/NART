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
import caffe
import unittest
import numpy as np
from torch.autograd import Variable
from nart.tools.pytorch.convert import convert_v2, convert_mode, Convert
import logging

logger = logging.getLogger("nart.test")


def flatten(l):
    if isinstance(l, tuple):
        flattened = []
        for each in l:
            if isinstance(each, tuple):
                flattened += flatten(each)
            else:
                flattened.append(each)
        return flattened
    else:
        return l


def skip_test(fn):
    def skipper(*args, **kwargs):
        op_name = fn.__name__.replace("test_", "")
        if op_name is not None:
            logger.fatal(f"skip: {op_name} not supported yet.")
        return True

    return skipper


class LayerTestUnit(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LayerTestUnit, self).__init__(*args, **kwargs)
        self.epsilon = 5e-5

    def compare(
        self, model, input_shapes, filename, input_names, output_names, cloze=True
    ):
        """
        Convert the pytorch model to caffe model
        With the same input, compare outputs from pytorch/caffe model
        @params:
        model: the pytorch model to be converted
        input_shapes: list of ints of the shape of model input
        filename: the file name (without postfix) to be saved for the caffe model
        input_names: list of strs of the name of model input
        output_names: list of strs of the name of model output
        cloze: whether a [1] is added at the beginning of shapes
        """
        input_variables = Convert.substitute_shape(input_shapes, cloze)
        outputs = (model(*input_variables),)
        torch_outputs = []
        for torch_output in flatten(outputs):
            torch_outputs.append(torch_output.data.numpy())

        with convert_mode():
            convert_v2(
                model,
                input_shapes,
                filename,
                input_names=input_names,
                output_names=output_names,
                verbose=False,
                output_weights=True,
                cloze=cloze,
            )
        caffe_model = caffe.Net(
            filename + ".prototxt", filename + ".caffemodel", caffe.TEST
        )
        for input_name, input_variable in zip(input_names, flatten(input_variables)):
            caffe_model.blobs[input_name].data[:] = input_variable.data.numpy()
        outputs = caffe_model.forward()
        caffe_outputs = []
        for output_name in output_names:
            if output_name in outputs.keys():
                caffe_output = outputs[output_name]
                caffe_outputs.append(caffe_output)
            else:
                print(
                    "{}: output blob {} does not exist for caffe version".format(
                        filename, output_name
                    )
                )

        for torch_output, caffe_output in zip(torch_outputs, caffe_outputs):
            diff = np.abs(torch_output - caffe_output)
            self.assertTrue(np.all(diff < self.epsilon), np.max(diff))
