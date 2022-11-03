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

#ifndef QUANT_OP_TP_H
#define QUANT_OP_TP_H

#include "art/op.h"
#include "art/op_tp.h"

#define OP_QUANT_QUANTIZE   OP_DECLARE(OP_GROUP_CODE_QUANT, 0x80000000l)
#define OP_QUANT_DEQUANTIZE OP_DECLARE(OP_GROUP_CODE_QUANT, 0x80000001l)

#endif // QUANT_OP_TP_H
