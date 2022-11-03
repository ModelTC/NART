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

#ifndef QUANT_OPS_H
#define QUANT_OPS_H

#include "art/op.h"
#include "art/op_tp.h"

DECLARE_OP_TP(quant, quantize)
DECLARE_OP_TP(quant, dequantize)
DECLARE_OP_TP(quant, add)
DECLARE_OP_TP(quant, eltwise)
// DECLARE_OP_TP(cuda, sub)
// DECLARE_OP_TP(cuda, sqrt)
DECLARE_OP_TP(quant, relu)
DECLARE_OP_TP(quant, conv_2d)
DECLARE_OP_TP(quant, conv_2d_wino)
DECLARE_OP_TP(quant, deconv_2d)
DECLARE_OP_TP(quant, pool)
// DECLARE_OP_TP(cuda, lrn)
DECLARE_OP_TP(quant, ip)
// DECLARE_OP_TP(cuda, bn)
DECLARE_OP_TP(quant, concat)
DECLARE_OP_TP(quant, softmax)
DECLARE_OP_TP(quant, slice)
DECLARE_OP_TP(quant, interp)
DECLARE_OP_TP(quant, prelu)
DECLARE_OP_TP(quant, bilateralslice)

#endif // QUANT_OPS_H
