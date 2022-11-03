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

import warnings
import sys
import os


def write_netdef(netdef, file_path=""):
    """write caffe prototxt.

    Args:
        netdef(caffe_pb2.NetParameter): caffe netdef
        file_path(str): write path.
    """
    with open(file_path, "w") as f:
        f.write(str(netdef))


def write_model(model, file_path=""):
    """write cafffe model.

    Args:
        model: caffe_pb2.NetParameter, caffe weight
        file_path: str, write path.
    """
    model = model.SerializeToString()
    with open(file_path, "wb") as f:
        f.write(model)


def read_net_def(file_path=""):
    """read caffe netdef.

    Args:
        file_path: str, netdef path.
    """
    try:
        from ..proto import caffe_pb2
    except ImportError:
        warnings.warn(f"can not import caffe_pb2")
        sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}")

    net_def = caffe_pb2.NetParameter()
    with open(file_path) as f:
        from google.protobuf.text_format import Merge

        Merge(f.read(), net_def)
    return net_def


def generate_support_table(parser_cls, csv_like=False):
    """Generate a module(backend) Op support table.

    Args:
        parser_cls (modules.Parser): the class definition of a backend parser.
        csv_like (bool): whether to flatten the information into a csv-like form.

    Returns:
        (list<dict>): the Op supporting information of the backend.
    """
    op_descs = parser_cls.get_all_ops()
    if len(op_descs) == 0:
        parser_cls.register_defaults()
        op_descs = parser_cls.get_all_ops()
    ret = []
    for name, op_desc in op_descs.items():
        item = {"name": name, "supported": True}
        restriction = {}
        descs = op_desc._desc
        for attr, desc in descs.items():
            restriction[attr] = desc[1]
        if csv_like:
            # flatten the restriction information into a string.
            restriction_str = []
            for attr, restr in restriction.items():
                restriction_str.append(f"{attr}: {restr.strip()}")
            restriction = "\n".join(restriction_str)
        item["restriction"] = restriction
        ret.append(item)
    # now add all unsupported Ops
    supported = {x["name"] for x in ret}
    from ..ops import Op

    unsupported = [op for op in Op.ONNX_OP if op not in supported]
    ret.extend([{"name": x, "supported": False} for x in unsupported])
    return ret


def dump_support_info_to_csv(parser_cls, file):
    """Generate a module(backend) Op support table, and write it to a file (or file like stream).

    Args:
        parser_cls (modules.Parser): the class definition of a backend parser.
        file: the destination file.
    """
    info = generate_support_table(parser_cls, True)
    if len(info) == 0:
        return
    import csv

    writer = csv.DictWriter(file, info[0].keys())
    writer.writeheader()
    writer.writerows(info)


def clean_caffe_model(model):
    """Create an cleaned caffe model from on existing one, only layer name, type and blobs will be kept.

    Args:
        model (caffe_pb2.NetParameter): The original caffe model.

    Returns:
        caffe_pb2.NetParameter: The cleaned model.
    """
    from copy import deepcopy

    net_param = type(model)()
    net_param.name = model.name
    net_param.input.extend(model.input)
    net_param.input_shape.extend(deepcopy(model.input_shape))
    net_param.input_dim.extend(model.input_dim)

    # merge all layers
    for src in model.layer:
        dst = net_param.layer.add()
        dst.name = src.name
        dst.type = src.type
        dst.bottom.extend(src.bottom)
        dst.top.extend(src.top)
        dst.blobs.extend(deepcopy(src.blobs))
    # write_netdef(net_param, '2.pbtxt')
    return net_param
