/*
 * Copyright 2022 SenseTime Group Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include "art/op.h"
#include "art/op_tp.h"

DECLARE_OP_TP(cuda, add)
DECLARE_OP_TP(cuda, abs)
DECLARE_OP_TP(cuda, conv_2d)
DECLARE_OP_TP(cuda, conv_nd)
DECLARE_OP_TP(cuda, deconv_2d)
DECLARE_OP_TP(cuda, relu)
DECLARE_OP_TP(cuda, relu6)
DECLARE_OP_TP(cuda, prelu)
DECLARE_OP_TP(cuda, pool)
DECLARE_OP_TP(cuda, ip)
DECLARE_OP_TP(cuda, eltwise)
DECLARE_OP_TP(cuda, softmax)
DECLARE_OP_TP(cuda, concat)
DECLARE_OP_TP(cuda, interp)
DECLARE_OP_TP(cuda, sigmoid)
DECLARE_OP_TP(cuda, bn)
DECLARE_OP_TP(cuda, batchnorm)
DECLARE_OP_TP(cuda, deform_conv_2d)
DECLARE_OP_TP(cuda, roialignpooling)
DECLARE_OP_TP(cuda, roipooling)
DECLARE_OP_TP(cuda, psroipooling)
DECLARE_OP_TP(cuda, psroialignpooling)
DECLARE_OP_TP(cuda, pad)
DECLARE_OP_TP(cuda, mul)
DECLARE_OP_TP(cuda, podroialignpooling)
DECLARE_OP_TP(cuda, correlation)
DECLARE_OP_TP(cuda, quant_dequant)
DECLARE_OP_TP(cuda, div)
DECLARE_OP_TP(cuda, sub)
DECLARE_OP_TP(cuda, pow)
DECLARE_OP_TP(cuda, log)
DECLARE_OP_TP(cuda, exp)
DECLARE_OP_TP(cuda, tanh)
DECLARE_OP_TP(cuda, instancenorm)
DECLARE_OP_TP(cuda, matmul)
DECLARE_OP_TP(cuda, podroialignpooling)
DECLARE_OP_TP(cuda, correlation)
DECLARE_OP_TP(cuda, reducemin)
DECLARE_OP_TP(cuda, reducemax)
DECLARE_OP_TP(cuda, reducemean)
DECLARE_OP_TP(cuda, reduceprod)
DECLARE_OP_TP(cuda, reducesum)
DECLARE_OP_TP(cuda, reducel2)
DECLARE_OP_TP(cuda, transpose)
DECLARE_OP_TP(cuda, lpnormalization)
DECLARE_OP_TP(cuda, slice)
DECLARE_OP_TP(cuda, reshape)
DECLARE_OP_TP(cuda, gridsample)
DECLARE_OP_TP(cuda, unfold)
DECLARE_OP_TP(cuda, subpixel)
DECLARE_OP_TP(cuda, lstm)
DECLARE_OP_TP(cuda, hardsigmoid)
DECLARE_OP_TP(cuda, heatmap2coord)
DECLARE_OP_TP(cuda, clip)
DECLARE_OP_TP(cuda, cast)
DECLARE_OP_TP(cuda, erf)
DECLARE_OP_TP(cuda, psroimaskpooling)
DECLARE_OP_TP(cuda, hswish)
DECLARE_OP_TP(cuda, scatternd)
DECLARE_OP_TP(cuda, elu)
DECLARE_OP_TP(cuda, clip_cast)
DECLARE_OP_TP(cuda, add_div_clip_cast)
#endif // CUDA_OPS_H
