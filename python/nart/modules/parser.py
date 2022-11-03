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

# -*- coding: utf-8 -*-
from ..core.art import FakeParade
from ..core.art import FakeOp
from ..core.art import FakeTensor
from ..core.art import Fakes
from ..core.art import Dtype
from onnx import numpy_helper
import tempfile
import os

from abc import ABC, abstractmethod

MODULE_DICT = {}

import logging

logger = logging.getLogger("nart.modules.parser")

from ..ops import OpSetRegistry


class Parser:
    """base class of module parser

    Attributes:
        net: origin net
        temp_dir: temp_dir
        config: a dict of module config
        extend_outputs_ : extend outputs
    """

    # use -1 to represent a dynamic dimension. NOTE: DO NOT USE -1 directly! since it may change later.
    DYNAMIC_SHAPE_SIZE = -1

    @classmethod
    def __init_subclass__(cls, module_name=None):
        """register module in MODULE_DICT.

        Args:
            module_name: str.
        """
        if module_name is None:
            module_name = cls.__name__
        MODULE_DICT[module_name] = cls
        cls._registered_ops = OpSetRegistry()

    @classmethod
    def register_op(cls, op_name, op):
        """register op directly"""
        # domain and version will be infered from the op_desc
        cls._registered_ops.insert(op, op_type=op_name)

    @classmethod
    def is_registered(cls, op_name: str, domain: str = "", version: int = 9):
        """check if `op_name` was registerd"""
        key = OpSetRegistry.make_key(op_name, domain=domain, version=version)
        return key in cls._registered_ops

    @classmethod
    def get_registered_op(cls, name, domain="", version=9):
        """fetch registered op by name"""
        return cls._registered_ops.find(
            name, domain=domain, version=version, default=None
        )

    @classmethod
    def get_all_ops(cls):
        """fetch all registered ops"""
        return cls._registered_ops

    @classmethod
    def if_supported(cls, op):
        """check if `op` was supported"""
        if not cls.is_registered(op.op_type, op.domain, op.version):
            return False
        return cls.get_registered_op(op.op_type, op.domain, op.version).support(op)

    def __init__(self, net, config={}, data_generator=None, net_id=0, *args, **kwargs):
        # run custom passes
        for item in self.get_passes():
            item.run(net)
        self.net = net
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = {"output_names": []}
        self.config.update(config)
        self.extend_outputs_ = self.config["output_names"]
        self.data_generator = data_generator

        self.min_shapes = None
        self.max_shapes = None
        self.build_shapes = net.tensor_shapes
        # check layer type
        self.before_parse()

    def before_parse(self):
        """check layer type"""
        for node in self.net.nodes:
            if not self.__class__.if_supported(node):
                raise RuntimeError(
                    f"{self.__class__.__name__} for type '{node.op_type}' not registered"
                )

    def __del__(self):
        if "temp_dir" in self.__dict__:
            self.temp_dir.cleanup()

    def check_no_dynamic_shape(self, backend_name):
        """For those backends that doesn't support dynamic shape (except batch dimension),
        this method do the check to make sure all inputs/outputs has no dynamic shape (except batch dimension).

        Args:
            backend_name (str): a backend name displayed to user when requirement not satisfied.
        """
        inputs = [x for x in self.net.input if x not in self.net.initializer]
        outputs = self.net.output.copy()
        for name in inputs + outputs:
            if any(
                self.is_dynamic_shape_size(x)
                for x in self.get_determined_shape(name)[1:]
            ):
                raise RuntimeError(
                    f"tensor `{name}` has dynamic shape (besides batch dimension), which {backend_name} cannot support"
                )

    def detect_max_batch_size(self, include_output=True, excluded=set()):
        """Detects max_batch_size

        Args:
            include_output (bool): Should output tensors be included to do checking when detecting maximum batch size.
            excluded (set): tensors that are excluded when checking.

        Returns:
            int: the detected max batch size, or None if all inputs are excluded.
        """
        inputs = [x for x in self.net.input if x not in self.net.initializer]
        outputs = self.net.output.copy()
        max_batch_size = None
        if include_output:
            tensor_names = inputs + outputs
        else:
            tensor_names = inputs
        # remove excluded tensors
        tensor_names = {x for x in tensor_names if x not in excluded}
        for name in tensor_names:
            shape = self.get_max_shape(name)
            if max_batch_size is None:
                max_batch_size = shape[0]
            elif max_batch_size != shape[0]:
                raise RuntimeError(
                    f"can not determine max batch size, the Graph `{self.net.name}`'s "
                    f"input/output tensors have different `N` size: "
                    f"{max_batch_size} vs {shape[0]}"
                )
        return max_batch_size

    def detect_min_batch_size(self, include_output=True, excluded=set()):
        """Detects minimum batch size of the Graph.

        Args:
            include_output: Should output tensors be included to do checking when detecting maximum batch size.
            excluded (set): tensors that are excluded when checking.

        Returns:
            int: the detected max batch size, or None if all inputs are excluded.
        """
        inputs = [x for x in self.net.input if x not in self.net.initializer]
        outputs = self.net.output.copy()
        min_batch_size = None
        if include_output:
            tensor_names = inputs + outputs
        else:
            tensor_names = inputs
        tensor_names = {x for x in tensor_names if x not in excluded}
        for name in tensor_names:
            shape = self.get_min_shape(name)
            if min_batch_size is None:
                min_batch_size = shape[0]
            else:
                assert min_batch_size == shape[0]
        return min_batch_size

    def set_shape_range(self, min_shapes, max_shapes, calib_shapes):
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.calib_shapes = calib_shapes
        return self

    def get_min_shape(self, name):
        """Get the minimum shape of a tensor.

        Args:
            name (str): name of tensor.
        Returns:
            (list<int>): the minimum shape of the tensor.
        """
        return self.min_shapes[name]

    def get_max_shape(self, name):
        """Get the maximum shape of a tensor.

        Args:
            name (str): name of tensor.
        Returns:
            (list<int>): the maximum shape of the tensor.
        """
        return self.max_shapes[name]

    def get_build_shape(self, name):
        """Get the default build shape of a tensor. That is, the shape of tensors, when the model is now being built.

        Args:
            name (str): name of tensor.
        Returns:
            (list<int>): the build shape of the tensor.
        """
        return self.build_shapes[name]

    def get_determined_shape(self, name):
        """Get the determined shape of a tensor. If the size on one dimension can't be determined,
        the shape size on that dimension is dynamic (and the value itself has not sense),
        you can use self.is_dynamic_shape_size(shape[i]) to check it.
        """
        min_shape = self.get_min_shape(name)
        max_shape = self.get_max_shape(name)
        assert len(min_shape) == len(
            max_shape
        ), f"the number of dimensions of min/max shape does not match, {len(min_shape)} vs {len(max_shape)}"
        return [
            a if a == b else self.DYNAMIC_SHAPE_SIZE
            for a, b in zip(min_shape, max_shape)
        ]

    def is_dynamic_shape_size(self, shape_size):
        return shape_size == self.DYNAMIC_SHAPE_SIZE

    @classmethod
    def get_passes(cls):
        logger.warning(
            f"{cls.__name__} does not override Parser.get_passes method, returning default pass list"
        )

        from ..passes import (
            ConstantToInitializer,
            ConvertGlobalPool,
            ExtractEltwise,
            DeadCodeElimination,
            SubToAdd,
            SoftmaxToCaffeSoftmax,
            EliminateNodes,
        )
        from ..ops.op import Dropout, Cast

        def should_remove(node, graph):
            return isinstance(node, (Dropout, Cast))

        passes = [
            EliminateNodes(should_remove),
            ConstantToInitializer(),
            ConvertGlobalPool(),
            SubToAdd(),
            ExtractEltwise(),
            SoftmaxToCaffeSoftmax(),
            ConstantToInitializer(),
            DeadCodeElimination(),
        ]
        return passes

    @abstractmethod
    def parse(self, into_parade=None):
        """parse net"""
        # example
        # if into_parade is None:
        #     into_parade = FakeParade()
        # into_parade_outputs = {}
        # for t in into_parade.tensors:
        #     into_parade_outputs[t.name] = t
        # dct_name_tensor = self.parse_input_dict()
        # dct_name_tensor.update({x:into_parade_outputs[x] for x in dct_name_tensor if x in into_parade_outputs})
        # output_names = self.parse_output_names()
        # outputs = list(map(lambda x:FakeTensor(Dtype.Float32, shape= (apply_batch_size(list(self.net.get_shape()[x])),1,0), name = x), output_names))
        # x = into_parade.append(
        #         Fakes.XXNet(
        #             list(map(lambda x:dct_name_tensor[x], dct_name_tensor)),
        #             self.net,
        #             outputs
        #             )
        #         )
        # for t, name in zip(x, output_names):
        #     t.name = name
        # return into_parade

        raise NotImplementedError

    def parse_input_dict(self):
        """parse input dict

        Returns:
            a dict, name to FakeTensor.

        """
        res = {}
        for t in self.net.input:
            sp = self.net.get_tensor_shape(t)
            if (
                "dtype" in self.config
                and self.net.name in self.config["dtype"]
                and t in self.config["dtype"][self.net.name]
            ):
                dtype = self.config["dtype"][self.net.name][t]
            else:
                dtype = self.net.get_tensor_dtype(t)
            logger.info("set input: %s to dtype: %s" % (t, dtype))

            if not sp:
                logger.warning('convert shape of op "%s" from [] to [1]' % t)
                sp = [1]

            if t not in self.net.initializer:
                tensor = FakeTensor(dtype=dtype, shape=((list(sp)), 1, 0), name=t)
            else:
                tensor = FakeTensor(
                    data=numpy_helper.to_array(self.net.initializer[t]),
                    dtype=dtype,
                    shape=(sp),
                )
            res[t] = tensor
        return res

    def parse_output_names(self):
        """get net output_names in order

        Returns:
            a list, output name list.
        """
        inputs = set(self.net.input)
        outputs = set(self.net.output)

        for o in self.extend_outputs_:
            if o not in outputs and o not in inputs:
                outputs.add(o)

        res = []
        for node in reversed(self.net.nodes):
            for t in node.output:
                if t in outputs and t not in res:
                    res.append(t)
        res.reverse()
        return res

    @classmethod
    def gen_workspace(cls):
        return {}

    @classmethod
    def gen_input_workspace(cls):
        return "default"

    def dump_file_if_required(self, data, filename=None):
        import numpy as np

        if self.config.get("dump_info", False):
            output_path = self.config.get("dump_filename")
            if filename is not None:
                output_path = os.path.join(os.path.dirname(output_path), filename)
            if isinstance(data, str):
                with open(output_path, "wt") as f:
                    f.write(data)
                logger.debug(f"Dump to {output_path} successfully.")
            elif isinstance(data, bytes):
                with open(output_path, "wb") as f:
                    f.write(data)
                logger.debug(f"Dump to {output_path} successfully.")
            elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
                data = data.tobytes()
                with open(output_path, "wb") as f:
                    f.write(data)
                logger.debug(f"Dump to {output_path} successfully.")
            else:
                logger.warn(
                    f"Dump file failed, unsupported object format {type(data)}."
                )
