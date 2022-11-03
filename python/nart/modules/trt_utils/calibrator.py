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

import logging

LOGGER = logging.getLogger("nart.modules.tensorrt")
import numpy as np

DEBUG_DUMP_CALIBRATION_CACHE = False
DEBUG_LOAD_CALIBRATION_CACHE = False

calibrator_classes = None


def register_calibrators():
    """generate calibrator classes, and register it into calibrator_classes dict."""
    global calibrator_classes
    if calibrator_classes is not None:
        LOGGER.warning("calibrator has been registered, do not repeatly register")
        return

    def def_calibrator_class(CalibratorBase):
        """defines a calibrator which derivates from CalibratorBase"""

        class Calibrator(CalibratorBase):
            """the calibrator to use. It reads inputs from DataGenerator."""

            CALIBRATION_CACHE_FILE_PATH = "calib-cache-{0}.bytes"

            def __init__(self, data_num, input_fp_dict_eval, input2shape, net_name):
                """create a calibrator
                Args:
                    data_num (int):
                    input_fp_dict_eval (function): a function to get calibration data,
                        it should return a dict from tensor name to list of file path (`dict<str, list<str>>`).
                    input2shape (list<int>): the tensor shapes same as calibration profile.
                """
                CalibratorBase.__init__(self)

                import pycuda.autoinit
                import pycuda.driver as cuda

                self.data_num = data_num
                self.input_fp_dict_eval = input_fp_dict_eval
                self.input2shape = input2shape
                net_name = net_name.replace("/", "_")
                self.cache_file_path = self.CALIBRATION_CACHE_FILE_PATH.format(net_name)

                self.calibration_cache = None

                from functools import reduce
                from operator import mul

                # allocate buffers
                input_buffers = dict()
                for name, shape in input2shape.items():
                    # modify the batch size in shape.
                    # Eassume that the axis 0 is batch axis.
                    # shape[0] = batch_size
                    # NOTE: float32 dtype is assumed
                    size = reduce(mul, shape, 1) * 4
                    input_buffers[name] = cuda.mem_alloc(size)

                self.input_buffers = input_buffers

                def data_provider(name):
                    input_fp_dict = input_fp_dict_eval()
                    shape = input2shape[name]
                    # the data count.
                    data_count = reduce(mul, shape, 1)
                    for idx in range(data_num):
                        data = np.fromfile(input_fp_dict[name][idx], dtype=np.float32)
                        assert data.size == data_count, (
                            "bad calibration data for tensor `{name}`",
                            f"the expected data shape is {shape}, but the given data's size is {data.size}",
                        )
                        data = np.reshape(data, shape)
                        yield data

                self.generators = {x: data_provider(x) for x in input2shape}

            def get_batch(self, names):
                """overrides the Base::get_batch, get a batch of input for calibration."""
                import pycuda.autoinit
                import pycuda.driver as cuda

                try:
                    # copy to device
                    for name, generator in self.generators.items():
                        data = next(generator)
                        cuda.memcpy_htod(self.input_buffers[name], data)
                    return [int(self.input_buffers[x]) for x in names]
                    # return map(lambda x: int(self.input_buffers[x]), names)
                except StopIteration:
                    return None

            def get_batch_size(self):
                # return self.batch_size
                return 1

            def read_calibration_cache(self):
                if DEBUG_LOAD_CALIBRATION_CACHE:
                    with open(self.cache_file_path, "rb") as file:
                        self.calibration_cache = file.read()
                    LOGGER.info(f"loaded calibration cache from {self.cache_file_path}")
                return self.calibration_cache

            def write_calibration_cache(self, cache):
                # simply store the cache as an attribute.
                self.calibration_cache = cache
                if DEBUG_DUMP_CALIBRATION_CACHE:
                    with open(self.cache_file_path, "wb") as file:
                        file.write(cache)
                    LOGGER.info(f"calibration cache dumped to {self.cache_file_path}")

        return Calibrator

    import tensorrt.tensorrt as trt

    calibrator_classes = dict()
    calibrator_classes["IInt8EntropyCalibrator2"] = def_calibrator_class(
        trt.IInt8EntropyCalibrator2
    )


def create_calibrator(
    data_num,
    input_fp_dict_eval,
    shape_by_name,
    net_name="",
    type="IInt8EntropyCalibrator2",
):
    global calibrator_classes
    if calibrator_classes is None:
        register_calibrators()
    Calibrator = calibrator_classes[type]
    return Calibrator(data_num, input_fp_dict_eval, shape_by_name, net_name)
