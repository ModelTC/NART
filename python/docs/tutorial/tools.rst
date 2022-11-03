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

.. _tutorial-tools:

NART Tools
=============

caffe.convert
--------------------------

caffe.convert模块主要用于对caffe模型做等价变换，常用于merge bn, inpalce relu等可以减少计算量/显存占用的场景

**example**

.. code-block:: bash

	$ python -m nart.tools.caffe.convert -a whatever.prototxt whatever.caffemodel


**frequently used option**

-a 对模型做常用的几个变换包括merge bn, inplace relu, merge slice等，等价于(-b -B -s -S --bns --BNs --inrelu --concsli)

-b 如果Batchnorm前面有卷积或者innerproduct，则merge进去，去除掉Batchnorm层的计算

-B 如果BN前面有卷积或者innerproduct，则merge进去，去除掉BN层的计算

-s 如果Scale前面有卷积或者innerproduct，则merge进去，去除掉Scale层的计算

-S 如果Slice前面是卷积，且Slice针对channel，则拆分前面的卷积，使得不需要Slice操作

--bns 把batchnorm和scale  merge成BN

--BNs 把BN和scale合并，成BN

--concsli 若出现连续的concat和slice，则去掉slice

--concbn 拆分bn和concat

--pure_output 输出的 caffemodel只包含layer的blob

-r 拆分所有group 卷积成小卷积

**detailed usage**

.. code-block:: bash

	$python -m nart.tools.caffe.convert --help
	usage: convert.py [-h] [-b] [-B] [-s] [-S] [--bns] [--BNs] [--scaleconv]
	                  [--bnconv] [--BNconv] [--scaleip] [--bnip] [--BNip]
	                  [--scale2BN] [-n] [--name [NAME [NAME ...]]]
	                  [--mean [MEAN [MEAN ...]]] [--std [STD [STD ...]]]
	                  [--concsli] [--concbn] [--splitbn] [--inrelu] [--prelu] [-a]
	                  [--cs] [--pure_output] [-v] [-r]
	                  modelpath1 [modelpath2]
	
	merge specific layers into conv layer
	
	positional arguments:
	  modelpath1            path to the caffe model
	  modelpath2            path to the caffe model
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -b, --batchnorm       conv + batchnorm -> conv
	  -B, --bn              conv + BN -> conv
	  -s, --scale           conv + scale -> conv
	  -S, --slice           merge slice layer
	  --bns                 batchnorm + scale -> BN
	  --BNs                 BN + scale -> BN
	  --scaleconv           scale + conv -> conv
	  --bnconv              batchnorm + conv -> conv
	  --BNconv              BN + conv -> conv
	  --scaleip             scale + ip -> ip
	  --bnip                batchnorm + ip -> ip
	  --BNip                BN + ip -> ip
	  --scale2BN            scale -> BN
	  -n, --normalize       normalize the input image
	  --name [NAME [NAME ...]]
	                        name for input blob
	  --mean [MEAN [MEAN ...]]
	                        mean value for normalization
	  --std [STD [STD ...]]
	                        standard deviation for normalization
	  --concsli             merge concat and slice
	  --concbn              reverse concat and bn
	  --splitbn             split bn into batch_norm and scale
	  --inrelu              inplace relu layer
	  --prelu               convert prelu to eltwise and relu
	  -a, --all             merge all supported layer
	  --cs                  concat layer support while merge slice layer
	  --pure_output         output pure caffemodel which only contains layer name
	                        and blobs
	  -v, --verbose         output detail information
	  -r, --resolve         resolve group convolution


caffe.count
-------------------------

caffe.count用于对caffe模型进行计算量统计，使用时输入caffe的prototxt文件，输出模型的计算量。


**example**

.. code-block:: bash

	$ python -m nart.tools.caffe.count whatever.prototxt
	comp: 15.265602, add: 38.567361, div: 23.498024, macc: 7234.477539, exp: 0.000954
	param: 170.142487, activation: 269.324036

输出解释：

输出单位是M。comp表示比较次数，add表示加法次数，div表示除法次数，macc表示乘加（一般通过macc算flops，一个macc等于2个flops），exp表示求指数次数。

param表示参数大小，activation表示访存次数。

pytorch.convert
-------------------------

pytorch.convert是基于onnx的模型转换工具，可将pytorch的模型转换为caffe的模型，在1.2.0之前主要使用内部的Network及Layer来完成图变换及onnx->caffe的转换，自1.2.0版本起已改为使用CaffeAlter完成onnx->caffe的转换来获得更大的灵活性

**example**

.. code-block:: bash

	import nart.tools.pytorch as pytorch
	from resnet import resnet18
	 
	model = resnet18()
	with pytorch.convert_mode():
	    pytorch.convert_v2(model,[(3,224,224)],"testfile",input_names=["data"],output_names=["out"])
	#input_names和output_names是可选参数,务必与实际输入输出对应

