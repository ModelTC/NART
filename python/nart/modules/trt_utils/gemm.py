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

""" Contains parser(s) for a ONNX op, different class may be defined based on the tensorrt version.
    NOTE: This file will not be imported by default, call trt_utils.init_parsers() to import.
          This also means, when adding a new parser file, you should add `import xxx` to trt_utils.init_parsers
"""
from ..tensorrt import ONNXNodeParser, TensorrtParser
from ...ops.op import OpDesc
from .parse_utils import (
    get_nb_dims,
    ShapeTensor,
    reduce_mul,
    concat_shape,
    concat_shapes,
    get_shape,
    add_shuffle,
    flatten_tensor,
    to_const_tensor,
)
from tensorrt import tensorrt as trt

real_parser = TensorrtParser.get_class()


class Gemm(ONNXNodeParser, OpDesc, layertype="Gemm", parser=real_parser):
    @classmethod
    def _parse_(cls, node, network, itensor_by_name):
        A = itensor_by_name[node.input[0]]
        B = cls.get_itensor_by_name(node.input[1])
        bias = (
            cls.get_itensor_by_name(node.input[2])
            if len(node.input) == 3 and node.input[2]
            else None
        )
        get_attribute = cls.def_get_attr(node)

        opA = (
            trt.MatrixOperation.NONE
            if get_attribute("transA", 0) == 0
            else trt.MatrixOperation.TRANSPOSE
        )
        opB = (
            trt.MatrixOperation.NONE
            if get_attribute("transB", 0) == 0
            else trt.MatrixOperation.TRANSPOSE
        )

        # TODO: on which condition can we use fullyconnected instead.
        matmul = network.add_matrix_multiply(A, opA, B, opB)
        matmul.name = f"{node.name}::matmul"
        matmul_res = matmul.get_output(0)

        if bias is not None:
            # add bias tensor
            bias = to_const_tensor(bias) if not isinstance(bias, trt.ITensor) else bias
            # expand dimension of bias to 2
            expand_dim = add_shuffle(
                bias, concat_shape(ShapeTensor([1]), get_shape(bias))
            )
            bias = expand_dim.get_output(0)
            # then use elementwise to add them up
            add = network.add_elementwise(
                matmul_res, bias, trt.ElementWiseOperation.SUM
            )
            add.name = f"{node.name}::add"
            outputs = {node.output[0]: add.get_output(0)}
            return add, outputs
        else:
            outputs = {node.output[0]: matmul_res}
            return matmul, outputs

        # not used
        num_outputs = get_shape(B)[0]
        gemm = network.add_fully_connected(
            input=A, num_outputs=num_outputs, kernel=B, bias=bias
        )
        gemm.name = node.name

        outputs = {node.output[0]: gemm.get_output(0)}
        return gemm, outputs

    @OpDesc.attr(OpDesc.AttrType.FLOAT, 1.0)
    def alpha():
        return math.isclose(alpha, 1.0)

    @OpDesc.attr(OpDesc.AttrType.FLOAT, 1.0)
    def beta():
        return math.isclose(beta, 1.0)
