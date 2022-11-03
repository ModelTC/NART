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

#ifndef DEFAULT_OPS_H
#define DEFAULT_OPS_H

#include "art/op.h"
#include "art/op_tp.h"

DECLARE_OP_TP(default, add)
DECLARE_OP_TP(default, sub)
DECLARE_OP_TP(default, sqrt)
DECLARE_OP_TP(default, relu)
DECLARE_OP_TP(default, tanh)
DECLARE_OP_TP(default, prelu)
DECLARE_OP_TP(default, conv_2d)
DECLARE_OP_TP(default, conv_nd)
DECLARE_OP_TP(default, deform_conv_2d)
DECLARE_OP_TP(default, deconv_2d)
DECLARE_OP_TP(default, pool)
DECLARE_OP_TP(default, lrn)
DECLARE_OP_TP(default, ip)
DECLARE_OP_TP(default, bn)
DECLARE_OP_TP(default, batchnorm)
DECLARE_OP_TP(default, concat)
DECLARE_OP_TP(default, slice)
DECLARE_OP_TP(default, interp)
DECLARE_OP_TP(default, sigmoid)
DECLARE_OP_TP(default, softmax)
DECLARE_OP_TP(default, eltwise)
DECLARE_OP_TP(default, scale)
DECLARE_OP_TP(default, reshape)
DECLARE_OP_TP(default, transpose)
DECLARE_OP_TP(default, subpixel)
DECLARE_OP_TP(default, heatmap2coord)
DECLARE_OP_TP(default, exchange)
DECLARE_OP_TP(default, roipooling)
DECLARE_OP_TP(default, roialignpooling)
DECLARE_OP_TP(default, psroipooling)
DECLARE_OP_TP(default, correlation)
DECLARE_OP_TP(default, psroimaskpooling)
DECLARE_OP_TP(default, bilateralslice)
DECLARE_OP_TP(default, relu6)
DECLARE_OP_TP(default, shufflechannel)

DECLARE_OP_TP(default, write_input)
DECLARE_OP_TP(default, read_output)

DECLARE_OP_TP(default, psroialignpooling)

DECLARE_OP_TP(default, pad)
DECLARE_OP_TP(default, quant_dequant)
DECLARE_OP_TP(default, podroialignpooling)

DECLARE_OP_TP(default, mul)
DECLARE_OP_TP(default, div)
DECLARE_OP_TP(default, pow)
DECLARE_OP_TP(default, exp)
DECLARE_OP_TP(default, log)
DECLARE_OP_TP(default, instancenorm)

DECLARE_OP_TP(default, matmul)

DECLARE_OP_TP(default, reducemin)
DECLARE_OP_TP(default, reducemax)
DECLARE_OP_TP(default, reducemean)
DECLARE_OP_TP(default, reduceprod)
DECLARE_OP_TP(default, reducesum)
DECLARE_OP_TP(default, reducel2)

DECLARE_OP_TP(default, correlation1d)
DECLARE_OP_TP(default, lpnormalization)
DECLARE_OP_TP(default, gather)
DECLARE_OP_TP(default, argmax)
DECLARE_OP_TP(default, gridsample)
DECLARE_OP_TP(default, unfold)
DECLARE_OP_TP(default, topk)
DECLARE_OP_TP(default, abs)
DECLARE_OP_TP(default, floor)
DECLARE_OP_TP(default, lstm)
DECLARE_OP_TP(default, hardsigmoid)
DECLARE_OP_TP(default, erf)
DECLARE_OP_TP(default, clip)
DECLARE_OP_TP(default, cast)
DECLARE_OP_TP(default, hswish)
DECLARE_OP_TP(default, scatternd)
DECLARE_OP_TP(default, min)
DECLARE_OP_TP(default, sign)
DECLARE_OP_TP(default, roundto0)
DECLARE_OP_TP(default, elu)
DECLARE_OP_TP(default, clip_cast)
DECLARE_OP_TP(default, add_div_clip_cast)

#endif // DEFAULT_OPS_H
