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

#ifndef CONV_2D_WINO_H
#define CONV_2D_WINO_H

#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/quant/quant_helper.h"
#include "art/quant/quant_op_settings.h"
#include "art/quant/quant_op_tp.h"
#include "art/timer.h"

#include "../utils/winograd.h"

typedef struct {
    op_t o;
    uint32_t num_output;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_w;
    uint32_t kernel_h;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t group;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *walpha;
    uint8_t *wzero_point;
    uint8_t *wbits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;

    mem_t *aux_mem1; // winograd_pad_input
    mem_t *aux_mem2; // winograd_input
    mem_t *aux_mem3; // winograd_output
    mem_t *aux_mem4; // winograd_pad_output
    mem_t *tmp_output;
} op_conv_2d_wino_t;

size_t op_conv_2d_wino_i8xi8_alloc_aux_mem(op_t *op);
void op_conv_2d_wino_i8xi8_run(op_t *op);

size_t op_conv_2d_wino_u8xu8_alloc_aux_mem(op_t *op);

#endif // CONV_2D_WINO_H
