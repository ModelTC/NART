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

import argparse
import os
import logging
import numpy as np
import warnings
import tempfile

logger = logging.getLogger("nart.utils.data_generator")


class DataGenerator(object):
    """The base class of data generator. A data generator is used to provide data for calibration."""

    # the data generator which produces one frame (image) per run, so every file of it's input/output data represents one image.
    PerFrameGenerator = "PerFrameGenerator"

    def __init__(self, graph):
        self.input_path_map = {}
        self.temp_dir_handle = tempfile.TemporaryDirectory()
        self.output_root_dir = self.temp_dir_handle.name
        self.output_path_map = {}
        self.input_shape_map = {}
        # self.output_shape_map = {}
        self.rand_input = False
        self.data_num = 1
        self.rand_input_data_range = {}
        self.graph = graph

    def set_config(
        self,
        temp_dir=None,
        input_path_map={},
        data_num=1,
        rand_input=False,
        rand_input_data_range={},
    ):
        """Set the config of the data generater.
            Functions only when self.prepare is False.

        Args:
            output_layer_name_list[Optional]: Traversable container of str elements which are referred to the name of layers needed to be dumped.
            batch_sizep[Optional]: Set the run batch of the net.

        Returns:
            True if succeeded.
            False if faied.
        """
        if not temp_dir is None:
            self.output_root_dir = temp_dir
        self.input_path_map = input_path_map
        self.data_num = data_num
        self.rand_input = rand_input
        self.rand_input_data_range = rand_input_data_range

    def fetch_data(self, name):
        """ """
        raise NotImplementedError(
            f"fetch_data not implemented in {self.__class__.__name__}"
        )

    def pre_fetch(self, names):
        """ """
        raise NotImplementedError(
            f"pre_fetch not implemented in {self.__class__.__name__}"
        )

    @property
    def type(self):
        """`str`: The data generator type"""
        raise NotImplementedError(f"type not implemented in {self.__class__.__name__}")

    def prepare(self):
        # prepare run_environment
        graph = self.graph
        input_shape_dict = {
            name: graph.get_tensor_shape(name) for name in graph.network_inputs
        }
        if not self.rand_input:
            self._check_input_path()
        else:
            rand_input_path_map = {}
            for item in graph.network_inputs:
                input_path = f"{self.output_root_dir}/input/{item}"
                try:
                    os.makedirs(input_path)
                except:
                    if not os.path.isdir(input_path):
                        raise RuntimeError(f"{input_path} is not a dir.")
                    else:
                        warnings.warn(f"[dir]:{input_path} has existed.")
                rand_input_path_map[item] = input_path
            self._gen_rand_data(input_shape_dict, rand_input_path_map)
        for name in input_shape_dict:
            self.input_shape_map[name] = list(input_shape_dict[name])

    def _check_input_path(self):
        """Check input path, if rand_input == False."""
        inputs = self.graph.network_inputs
        for item in inputs:
            if item not in self.input_path_map:
                logger.warning(
                    f"[Data generation] Missing input tensor:[{item}] data, "
                    "which may cause errors when this tensor is acquired"
                )
                continue
            input_path_list = self.input_path_map[item]
            assert self.data_num == len(input_path_list)
            for path in input_path_list:
                if not os.path.isfile(path):
                    raise RuntimeError(
                        f"[Data generation] Data of {item}: {path} is not a file."
                    )

    def _gen_rand_data(self, input_shape_dict, work_path_map):
        """DataGenerator implementation should override this method to generate random input data."""
        raise NotImplementedError(
            f"_gen_rand_data not implemented in {self.__class__.__name__}"
        )


class PerFrameDataGenerator(DataGenerator):
    """A data generator which produces one frame (image) per run, so every file of it's input/output data represents one image.

    Attributes:
        net: origin core.Net.
        input_path_map: dict, name to input_path.

    """

    def __init__(self, graph):
        from ..core import Graph

        assert isinstance(graph, Graph)
        # delay creation of network.
        self._net = None
        super(PerFrameDataGenerator, self).__init__(graph)

    @property
    def net(self):
        if self._net is None:
            logger.info("creating net for generating calibration data")
            from ..core import Net

            self._net = Net.from_graph(self.graph, backend="cuda")
        return self._net

    @property
    def type(self):
        return DataGenerator.PerFrameGenerator

    def __del__(self):
        self.free_net()
        if not self.temp_dir_handle is None:
            self.temp_dir_handle.cleanup()

    def _make_dirs(self):
        if self.rand_input:
            for item in self.graph.network_inputs:
                output_path = f"{self.output_root_dir}/input{item}"
                try:
                    os.makedirs(output_path)
                except:
                    if not os.path.isdir(output_path):
                        raise RuntimeError(
                            f"{output_path} has existed but is not a dir."
                        )
                    else:
                        warnings.warn(f"[dir]:{output_path} has existed.")

    def _gen_rand_data(self, input_shape_dict, work_path_map):
        """Generate rand data of tensors."""
        assert isinstance(work_path_map, dict)
        assert isinstance(input_shape_dict, dict)
        logger.debug(
            f"[Data generation] input_path_map: {list(input_shape_dict.keys())}"
        )
        for item in input_shape_dict:
            self.input_path_map[item] = []
            tensor_name = item
            tensor_shape = input_shape_dict[item]
            logger.debug(
                "[Data generation] tensor_name:%16s, tensor_shape:%s",
                tensor_name,
                str(tensor_shape),
            )
            for i in range(self.data_num):
                i_data = np.random.rand(*tensor_shape)
                if tensor_name in self.rand_input_data_range:
                    i_data_range = self.rand_input_data_range[tensor_name]
                    if (
                        not isinstance(i_data_range, tuple)
                        or len(i_data_range) != 2
                        or i_data_range[0] >= i_data_range[1]
                    ):
                        warnings.warn(
                            f"[Data generation] Range of input tensor[{tensor_name}]'s data is wrong: {i_data_range}. Use default range: [0,1]."
                        )
                    else:
                        i_data = (
                            i_data * (i_data_range[1] - i_data_range[0])
                            + i_data_range[0]
                        )
                assert tensor_name in work_path_map
                i_data_path = os.path.join(work_path_map[tensor_name], "%d.bin" % (i))
                i_data = i_data.astype("float32")
                i_data.tofile(i_data_path)
                self.input_path_map[item].append(i_data_path)
        logger.debug(f"[Data generation] end _gen_rand_data")

    def fetch_data(self, name):
        """Forward the net and then save the data."""
        if name in self.input_path_map:
            return self.input_path_map[name][:]

        if name in self.output_path_map:
            return self.output_path_map[name][:]
        self.pre_fetch([name])
        return self.output_path_map[name]

    def pre_fetch(self, names):
        """Pre-generate data for tensor in ``names``, and save their path in self.output_path_map.
        By doing this avoids running the net for many timies.
        """
        names = set(names)
        for name in list(names):
            if name in self.input_path_map or name in self.output_path_map:
                # if tensor ``name`` is an input or already in output_path_map, no need to generate it twice
                names.remove(name)
            elif self.net.get_binding(name) is None:
                names.remove(name)
                warnings.warn(f"can not fetch tensor {name}")
        if len(names) == 0:
            return
        logger.info("generating data for tensors: {0}".format(",".join(names)))
        named_output_path = dict()
        for name in names:
            named_output_path[name] = f"{self.output_root_dir}/output/{name}"
            try:
                os.makedirs(named_output_path[name])
            except:
                if not os.path.isdir(named_output_path[name]):
                    raise RuntimeError(f"{named_output_path[name]} is not a dir.")
                else:
                    warnings.warn(f"{named_output_path[name]} has existed.")

        input_binding_map = self.net.get_input_binding()

        for name in names:
            self.output_path_map[name] = []
        for i in range(self.data_num):
            for input_name in input_binding_map:
                assert (
                    input_name in self.input_path_map
                ), "[Data generation] Missing input tensor:[{item}] data"
                input_data_path = self.input_path_map[input_name][i]
                input_data = np.fromfile(input_data_path, dtype="float32")
                input_data_len = len(input_data)
                input_binding_map_name_size = (
                    len(input_binding_map[input_name])
                    * len(input_binding_map[input_name][0])
                    * len(input_binding_map[input_name][0][0])
                    * len(input_binding_map[input_name][0][0][0])
                )
                input_binding_map[input_name][:] = input_data.reshape(
                    input_binding_map[input_name].shape
                )
            assert self.net.forward()
            for name in names:
                output_data_path = os.path.join(named_output_path[name], "%d.bin" % (i))
                data = self.net.get_binding(name)[:]
                data.astype(dtype="float32")
                data.tofile(output_data_path)
                self.output_path_map[name].append(output_data_path)

    def free_net(self):
        if not self._net is None:
            del self._net
        self.prepared = False
