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

#ifndef CONV_2D_H
#define CONV_2D_H

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

#include "../gemm/gemm.h"
#include "../utils/im2col.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t num_output;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t group;

    float *walpha;
    uint8_t *wzero_point;
    uint8_t *wbits;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;

    void (*gemm)(
        const size_t, const size_t, const size_t, const int8_t *, const int8_t *, int32_t *,
        mem_t *);
    void (*ugemm)(
        const size_t, const size_t, const size_t, const int16_t *, const int16_t *, int32_t *,
        mem_t *);
    size_t (*gemm_auxmem_size)(int, int, int);

    mem_t *temp_data; // temp mem for col in im2col
    mem_t *aux_mem;
    mem_t *tmp_output;
} op_conv_2d_t;

size_t op_conv_2d_i8xi8_alloc_aux_mem(op_t *op);
void op_conv_2d_i8xi8_run(op_t *op);

size_t op_conv_2d_u8xu8_alloc_aux_mem(op_t *op);
void op_conv_2d_u8xu8_run(op_t *op);

#ifdef __cplusplus
}
#endif

#endif // CONV_2D_H
