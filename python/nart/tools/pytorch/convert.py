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
import sys
from torch.autograd import Variable
from os.path import splitext, getsize
from .onnx_utils import load
from .network_utils import make_network
from .module_utils import convert_mode
import threading
from ..io import send
from .match_chain import _scope_module_name
from .trace_graph import trace_graph
import logging

from nart.utils import deprecated


def export(*args, **kwargs):
    minor = torch.__version__.split(".")[1]

    if int(minor) < 5 and "enable_onnx_checker" in kwargs:
        del kwargs["enable_onnx_checker"]
        del kwargs["use_external_data_format"]
    if int(minor) >= 5 and "use_external_data_format" not in kwargs:
        kwargs["use_external_data_format"] = False
    torch.onnx.export(*args, **kwargs)


def convert_v2(
    model,
    input_shapes,
    filename="test",
    log=True,
    input_names=None,
    output_names=None,
    input_dict=None,
    verbose=False,
    output_weights=True,
    cloze=True,
    use_external_data_format=False,
):
    """
    Convert a pytorch model into a caffe model
    @params:
    model: the pytorch model to be converted
    input_shapes: list of ints of the shape of model input
    filename: the file name (without postfix) to be saved for the caffe model
    log: indicate whether to log this call
    input_names: list of strs of the name of model input
    output_names: list of strs of the name of model output
    input_dict: dict of model configs except those of 'image' key
    verbose: indicate whether the detailed structure of model is displayed
    output_weights: indicate whether the weight file of caffe model is generated
    cloze: indicate whether to add a "1" at the head of each input shape
    """
    custom_info = {"Method": "convert", "Module": "nart.tools"}
    try:
        module_names = export_onnx(
            model,
            input_shapes,
            filename,
            input_names=input_names,
            output_names=output_names,
            log=False,
            input_dict=input_dict,
            verbose=verbose,
            cloze=cloze,
            use_external_data_format=use_external_data_format,
        )
        convert_onnx_v2(
            filename + ".onnx",
            log=False,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
            output_weights=output_weights,
            module_names=module_names,
        )
        custom_info["Status"] = "Success"
        custom_info["Caffe Model Size"] = "{} Bytes".format(
            getsize(filename + ".caffemodel")
        )
        custom_info["ONNX Model Size"] = "{} Bytes".format(getsize(filename + ".onnx"))
    except Exception as e:
        custom_info["Status"] = str(e)
        raise e
    finally:
        if log:
            log_thread = threading.Thread(target=send, args=(custom_info,))
            log_thread.start()
    return filename + ".prototxt", filename + ".caffemodel"


from nart.utils.alter.caffe import NonParamLayer


class Hswish(NonParamLayer):
    def layer_type(self):
        return "HSwish"


class Hsigmoid(NonParamLayer):
    def layer_type(self):
        return "Hsigmoid"


def convert_onnx_v2(
    filename,
    log=True,
    input_names=None,
    output_names=None,
    verbose=False,
    output_weights=True,
    module_names=None,
):
    if verbose:
        print("====convert: start of onnx-to-caffe export")

    custom_info = {"Method": "convert_onnx", "Module": "nart.tools"}
    try:
        from .network_utils import merge_shuffle_channel_nodes, merge_reshape_nodes

        model = load(filename)
        merge_reshape_nodes(model)
        merge_shuffle_channel_nodes(model)
        if module_names is not None:
            if len(module_names) != len(model.graph.node):
                print("[WARNING] len of traced module_names is not equal to nodes")
            else:
                for idx, (node, name) in enumerate(zip(model.graph.node, module_names)):
                    node.name = name + "_" + str(idx)

        from nart.utils.alter import CaffeAlter, CaffeAlterContext
        from nart.utils.alter.caffe import register_caffe_layer, Layer
        from nart.core import Graph, Node
        from nart.utils.onnx_utils import OnnxDecoder
        from nart.passes import (
            ConstantToInitializer,
            FoldConstantToAttr,
            DeadCodeElimination,
            ExtractEltwise,
            EliminateNodes,
        )
        from nart.utils.passes import (
            ExtractHswish,
            ExtractHsigmoid,
            ExtractGemm,
            EliminateReshapeNode,
        )
        from nart.proto import nart_caffe_pb2 as caffe_pb2
        from nart.ops.op import Dropout, Cast

        name = splitext(filename)[0]

        graph = OnnxDecoder().decode(model)
        graph.update_tensor_shape()
        graph.update_topology()

        def should_remove(node, graph):
            return isinstance(node, (Dropout, Cast))

        FoldConstantToAttr().run(graph)
        ExtractHswish().run(graph)
        ExtractHsigmoid().run(graph)
        ExtractGemm().run(graph)
        ExtractEltwise().run(graph)
        EliminateReshapeNode().run(graph)
        EliminateNodes(should_remove).run(graph)
        ConstantToInitializer().run(graph)
        DeadCodeElimination().run(graph)
        graph.update_topology()

        with CaffeAlterContext() as ctx:
            register_caffe_layer("Hswish", Hswish)
            register_caffe_layer("Hsigmoid", Hsigmoid)
            alter = CaffeAlter(graph)
            cnet = alter.parse()

        with open(name + ".caffemodel", "wb") as f:
            f.write(cnet.SerializeToString())

        for l in cnet.layer:
            l.ClearField("blobs")

        with open(name + ".prototxt", "wt") as f:
            f.write(str(cnet))

        custom_info["Status"] = "Success"
        custom_info["Caffe Model Size"] = "{} Bytes".format(
            getsize(name + ".caffemodel")
        )
    except Exception as e:
        custom_info["Status"] = str(e)
        raise e
    finally:
        if log:
            log_thread = threading.Thread(target=send, args=(custom_info,))
            log_thread.start()

    if verbose:
        print("====convert: end of onnx-to-caffe export")


@deprecated(new_fn=convert_v2)
def convert_v1(
    model,
    input_shapes,
    filename="test",
    log=True,
    input_names=None,
    output_names=None,
    input_dict=None,
    verbose=False,
    output_weights=True,
    cloze=True,
    use_external_data_format=False,
):
    """
    Convert a pytorch model into a caffe model
    @params:
    model: the pytorch model to be converted
    input_shapes: list of ints of the shape of model input
    filename: the file name (without postfix) to be saved for the caffe model
    log: indicate whether to log this call
    input_names: list of strs of the name of model input
    output_names: list of strs of the name of model output
    input_dict: dict of model configs except those of 'image' key
    verbose: indicate whether the detailed structure of model is displayed
    output_weights: indicate whether the weight file of caffe model is generated
    cloze: indicate whether to add a "1" at the head of each input shape
    """
    custom_info = {"Method": "convert", "Module": "nart.tools"}
    try:
        module_names = export_onnx(
            model,
            input_shapes,
            filename,
            log=False,
            input_dict=input_dict,
            verbose=verbose,
            cloze=cloze,
            use_external_data_format=use_external_data_format,
        )
        convert_onnx_v1(
            filename + ".onnx",
            log=False,
            input_names=input_names,
            output_names=output_names,
            verbose=verbose,
            output_weights=output_weights,
            module_names=module_names,
        )
        custom_info["Status"] = "Success"
        custom_info["Caffe Model Size"] = "{} Bytes".format(
            getsize(filename + ".caffemodel")
        )
        custom_info["ONNX Model Size"] = "{} Bytes".format(getsize(filename + ".onnx"))
    except Exception as e:
        custom_info["Status"] = str(e)
        raise e
    finally:
        if log:
            log_thread = threading.Thread(target=send, args=(custom_info,))
            log_thread.start()
    return filename + ".prototxt", filename + ".caffemodel"


convert = convert_v1
convert_caffe_by_return = convert


def convert_onnx_by_return(
    model,
    input_shapes,
    filename="test",
    log=True,
    input_dict=None,
    verbose=False,
    cloze=True,
    use_external_data_format=False,
):
    """
    Convert a pytorch model into an onnx model
    @params:
    model: the pytorch model to be converted
    input_shapes: list of ints of the shape of model input
    filename: the file name (without postfix) to be saved for the onnx model
    log: indicate whether to log this call
    input_dict: dict of model configs except those of 'image' key
    verbose: indicate whether the detailed structure of model is displayed
    cloze: indicate whether to add a "1" at the head of each input shape
    """
    if verbose:
        print("====convert: start of pytorch-to-onnx export")

    custom_info = {"Method": "export_onnx", "Module": "nart.tools"}
    try:
        model.cpu()
        model.eval()
        if input_dict is not None:
            print("Input dict is not None. Automatically setting cloze to be False.")
            cloze = False
        input_variables = Convert.substitute_shape(input_shapes, cloze)
        if input_dict is not None:
            input_dict["image"] = input_variables
            export(
                model,
                (input_dict,),
                filename + ".onnx",
                verbose=verbose,
                enable_onnx_checker=False,
                use_external_data_format=use_external_data_format,
            )
        else:
            export(
                model,
                input_variables,
                filename + ".onnx",
                verbose=verbose,
                enable_onnx_checker=False,
                use_external_data_format=use_external_data_format,
            )
        custom_info["Status"] = "Success"
        custom_info["ONNX Model Size"] = "{} Bytes".format(getsize(filename + ".onnx"))
    except Exception as e:
        custom_info["Status"] = str(e)
        raise e
    finally:
        if log:
            log_thread = threading.Thread(target=send, args=(custom_info,))
            log_thread.start()

    if verbose:
        print("====convert: end of pytorch-to-onnx export")
    return filename + ".onnx"


def export_onnx(
    model,
    input_shapes,
    filename,
    input_names=None,
    output_names=None,
    log=True,
    input_dict=None,
    verbose=False,
    cloze=True,
    use_external_data_format=False,
):
    """
    Convert a pytorch model into an onnx model
    @params:
    model: the pytorch model to be converted
    input_shapes: list of ints of the shape of model input
    filename: the file name (without postfix) to be saved for the onnx model
    log: indicate whether to log this call
    input_dict: dict of model configs except those of 'image' key
    verbose: indicate whether the detailed structure of model is displayed
    cloze: indicate whether to add a "1" at the head of each input shape
    """
    if verbose:
        print("====convert: start of pytorch-to-onnx export")

    custom_info = {"Method": "export_onnx", "Module": "nart.tools"}
    module_names = None
    try:
        model.cpu()
        model.eval()
        if input_dict is not None:
            print("Input dict is not None. Automatically setting cloze to be False.")
            cloze = False
        input_variables = Convert.substitute_shape(input_shapes, cloze)
        if input_dict is not None:
            input_dict["image"] = input_variables
            export(
                model,
                (input_dict,),
                filename + ".onnx",
                verbose=verbose,
                enable_onnx_checker=False,
                use_external_data_format=use_external_data_format,
            )

            # reset input_variables cause `torch.onnx.export` side-effect
            input_dict["image"] = input_variables
            graph = trace_graph(
                model, (input_dict,), torch.onnx.OperatorExportTypes.ONNX
            )
        else:
            export(
                model,
                input_variables,
                filename + ".onnx",
                verbose=verbose,
                enable_onnx_checker=False,
                use_external_data_format=use_external_data_format,
            )
            graph = trace_graph(
                model, input_variables, torch.onnx.OperatorExportTypes.ONNX
            )

        import onnx

        onnx_model = load(filename + ".onnx")
        from .network_utils.common import update_input_names, update_output_names

        if input_names is not None:
            update_input_names(onnx_model, input_names)
        if output_names is not None:
            update_output_names(onnx_model, output_names)
        onnx.save(onnx_model, filename + ".onnx")

        module_names = list(
            map(lambda x: _scope_module_name(x.scopeName()), graph.nodes())
        )

        custom_info["Status"] = "Success"
        custom_info["ONNX Model Size"] = "{} Bytes".format(getsize(filename + ".onnx"))
    except Exception as e:
        custom_info["Status"] = str(e)
        raise e
    finally:
        if log:
            log_thread = threading.Thread(target=send, args=(custom_info,))
            log_thread.start()

    if verbose:
        print("====convert: end of pytorch-to-onnx export")
    return module_names


@deprecated(new_fn=convert_onnx_v2)
def convert_onnx_v1(
    filename,
    log=True,
    input_names=None,
    output_names=None,
    verbose=False,
    output_weights=True,
    module_names=None,
):
    """
    Convert an onnx model into a caffe model
    @params:
    filename: the file name (with postfix) of the onnx model
    log: indicate whether to log this call
    input_names: list of strs of the name of model input
    output_names: list of strs of the name of model output
    verbose: indicate whether the detailed structure of model is displayed
    output_weights: indicate whether the weight file of caffe model is generated
    """
    if verbose:
        print("====convert: start of onnx-to-caffe export")

    custom_info = {"Method": "convert_onnx", "Module": "nart.tools"}
    try:
        model = load(filename)
        if module_names is not None:
            if len(module_names) != len(model.graph.node):
                print("[WARNING] len of traced module_names is not equal to nodes")
            else:
                for idx, (node, name) in enumerate(zip(model.graph.node, module_names)):
                    attr = node.attribute.add()
                    attr.name = "module_name"
                    attr.s = (name + "_" + str(idx)).encode("utf-8")

        name = splitext(filename)[0]
        network = make_network(model, name, input_names, output_names)
        network.save(verbose=verbose, output_weights=output_weights)
        custom_info["Status"] = "Success"
        custom_info["Caffe Model Size"] = "{} Bytes".format(
            getsize(name + ".caffemodel")
        )
    except Exception as e:
        custom_info["Status"] = str(e)
        raise e
    finally:
        if log:
            log_thread = threading.Thread(target=send, args=(custom_info,))
            log_thread.start()

    if verbose:
        print("====convert: end of onnx-to-caffe export")


convert_onnx = convert_onnx_v1


@deprecated()
def rearrange_channel(filename, indices, log=True):
    """
    ----毒瘤函数，无关人员速速退散----
    Rearrange the channel every 2 channels
    @params:
    filename: the name (without postfix) of the caffe model
    indices: the list of indices of the layers to be rearranged
    log: indicate whether to log this call
    ----版本旧约，对齐协议早早解脱----
    """
    import numpy as np
    from ..proto import caffe_pb2

    if log:
        custom_info = {"Method": "rearrange_channel", "Module": "nart.tools"}
        log_thread = threading.Thread(target=send, args=(custom_info,))
        log_thread.start()

    net = caffe_pb2.NetParameter()
    with open(filename + ".caffemodel", "rb") as fin:
        net.ParseFromString(fin.read())
    for layer in net.layer:
        layer_index = int(layer.name.split("_")[-1])
        if layer_index not in indices:
            continue
        weight = np.array(layer.blobs[0].data[:])
        span = weight.shape[0] // 2
        duliu = []
        for i in range(span):
            duliu.append(weight[2 * i])
        for i in range(span):
            duliu.append(weight[2 * i + 1])
        layer.blobs[0].data[:] = np.array(duliu)
    with open(filename + ".caffemodel", "wb") as fout:
        fout.write(net.SerializeToString())


class Convert(sys.modules[__name__].__class__):
    @staticmethod
    def __call__(
        model,
        input_shapes,
        filename,
        log=True,
        input_names=None,
        output_names=None,
        input_dict=None,
        verbose=False,
        output_weights=True,
        cloze=True,
    ):
        convert(
            model,
            input_shapes,
            filename,
            log=log,
            input_names=input_names,
            output_names=output_names,
            input_dict=input_dict,
            verbose=verbose,
            output_weights=output_weights,
            cloze=cloze,
        )

    @staticmethod
    def substitute_shape(input_shapes, cloze):
        input_variables = []
        for input_shape in input_shapes:
            if all(map(lambda shape: isinstance(shape, int), input_shape)):
                if cloze:
                    input_shape = [1] + list(input_shape)
                tensor = torch.randn(input_shape)
                variable = Variable(tensor, requires_grad=True)
                input_variables.append(variable)
            else:
                input_variables.append(Convert.substitute_shape(input_shape, cloze))
        return tuple(input_variables)


sys.modules[__name__].__class__ = Convert
