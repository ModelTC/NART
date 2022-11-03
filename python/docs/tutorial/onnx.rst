..  Copyright 2022 SenseTime Group Limited

..  Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

..  http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

.. _tutorial-onnx:

ONNX
=============

caffe2onnx
--------------------------

**command line**

.. code-block:: bash

    python -m nart.caffe2onnx caffe.prototxt caffe.caffemodel

**api**

.. code-block:: python

	from nart.utils.caffe_utils.caffe_builder import CaffeDecoder
	from nart.core import Model
	
	...
	... get prototxt, caffemodel

	x = CaffeDecoder()
	graph = x.load(prototxt, caffemodel)
	m = Model.make_model(graph)
	onnx = m.dump_to_onnx()
	with open(arg.output, "wb") as f:
	    f.write(onnx.SerializeToString())

	...
	...

onnx2caffe
--------------------------

**command line**

.. code-block:: bash

    python -m nart.onnx2caffe model.onnx

**api**

.. code-block:: python

	from nart.utils.onnx_utils.onnx_builder import OnnxDecoder
	from nart.utils.alter import CaffeAlter
	from nart.core import Model
	import onnx
	import argparse

	...
	... onnx_model = onnx.load('xxx')
	
	decoder = OnnxDecoder()
	graph = decoder.decode(onnx_model)
	
	alter = CaffeAlter(graph)
	caffemodel = alter.parse()
	
	from copy import deepcopy
	netdef = deepcopy(caffemodel)
	for layer in netdef.layer:
	    del layer.blobs[:]

	from nart.utils import write_netdef, write_model
	write_netdef(netdef, f"{arg.output}.prototxt")
	with open(f"{arg.output}.caffemodel", "wb") as f:
	    f.write(caffemodel.SerializeToString())

	...
	...
