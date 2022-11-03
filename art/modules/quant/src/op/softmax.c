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

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/quant/quant_helper.h"
#include "art/quant/quant_op_settings.h"

typedef struct {
    op_t o;
    uint32_t axis;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
    /* float */
    mem_t *tmp;
} op_softmax_t;

op_softmax_t *op_quant_softmax_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_softmax_t *res = (op_softmax_t *)malloc(sizeof(op_softmax_t));
    memset(res, 0, sizeof(op_softmax_t));
    return res;
}

void op_quant_softmax_tp_config(op_t *op)
{
    /* axis */
    CHECK(op_setting_single_get(op, SETTING_SOFTMAX_AXIS, dtUINT32, &((op_softmax_t *)op)->axis));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_softmax_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_softmax_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_softmax_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_softmax_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_softmax_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_softmax_t *)op)->obits));
}

void op_quant_softmax_tp_destroy(op_t *op)
{
    op_softmax_t *softmax_op = (op_softmax_t *)op;
    if (NULL != softmax_op->tmp)
        mem_delete(softmax_op->tmp);
}

void op_quant_softmax_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_softmax_run(op_t *op)
{
    const op_softmax_t *softmax_op = (op_softmax_t *)op;
    size_t axis = softmax_op->axis;
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_num = shape_part_count(&op->input_tensors[0]->shape, softmax_op->axis) / ch;
    size_t count = shape_count(&op->output_tensors[0].shape);
    size_t outer_num = count / inner_num / ch;

    float ialpha = softmax_op->ialpha[0];
    uint8_t izero_point = softmax_op->izero_point[0];
    float oalpha = softmax_op->oalpha[0];
    uint8_t ozero_point = softmax_op->ozero_point[0];

    size_t inner_idx, outer_idx, ch_idx;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    float *_tmp_out = (float *)mem_cpu_data(softmax_op->tmp);
    uint8_t bit = softmax_op->obits[0];
    for (outer_idx = 0; outer_idx < outer_num; ++outer_idx) {
        for (inner_idx = 0; inner_idx < inner_num; ++inner_idx) {
            float tmp = 0.;
            for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
                size_t idx = outer_idx * ch * inner_num + ch_idx * inner_num + inner_idx;
                tmp += (_tmp_out[idx] = exp(ialpha * (input_0[idx] - izero_point)));
            }
            for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
                size_t idx = outer_idx * ch * inner_num + ch_idx * inner_num + inner_idx;
                output_0[idx]
                    = saturate_float_by_bits((_tmp_out[idx] / tmp / oalpha) + ozero_point, bit);
            }
        }
    }
}

void op_quant_softmax_tp_prepare(op_t *op)
{
    int i;
    op_softmax_t *softmax_op = (op_softmax_t *)op;
    size_t axis = softmax_op->axis;
    if (NULL == softmax_op->tmp) {
        softmax_op->tmp = mem_new(cpu_mem_tp);
    }
    mem_alloc(softmax_op->tmp, sizeof(float) * shape_count(&op->output_tensors[0].shape));
    CHECK_GT((size_t)op->input_tensors[0]->shape.dim_size, axis);
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        op->run_func = op_quant_softmax_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
