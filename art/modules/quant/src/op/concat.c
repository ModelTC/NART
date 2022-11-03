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

#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/quant/quant_helper.h"
#include "art/quant/quant_op_settings.h"

#include "../quant_workspace.h"

typedef struct {
    op_t o;
    uint32_t axis;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_concat_t;

op_concat_t *op_quant_concat_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_concat_t *res = (op_concat_t *)malloc(sizeof(op_concat_t));
    memset(res, 0, sizeof(op_concat_t));
    return res;
}

void op_quant_concat_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_CONCAT_AXIS, dtUINT32, &((op_concat_t *)op)->axis));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_concat_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_concat_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_concat_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_concat_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_concat_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_concat_t *)op)->obits));
}

void op_quant_concat_tp_destroy(op_t *op) { (void)op; }

void op_quant_concat_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_concat_run(op_t *op)
{
    size_t i, j, k;
    // int batch_size = op->input_tensors[0]->shape.dim[op->input_tensors[0]->shape.batch_axis];
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t count = 0;
    const uint8_t *input = NULL;
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    const op_concat_t *concat_op = (op_concat_t *)op;

    float oalpha = concat_op->oalpha[0];
    uint8_t ozero_point = concat_op->ozero_point[0];

    uint8_t bit = concat_op->obits[0];
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < op->input_size; ++j) {
            input = mem_cpu_data(op->input_tensors[j]->mem);
            count = shape_count(&op->input_tensors[j]->shape) / batch_size;
            float ialpha = concat_op->ialpha[j];
            uint8_t izero_point = concat_op->izero_point[j];

#ifdef USE_FIXED_POINT_ONLY
            int8_t shift = 0;
            uint16_t mult = 0;
            get_quant_scale_int16(ialpha / oalpha, &mult, &shift);
#endif

            for (k = 0; k < count; k++) {
#ifdef USE_FIXED_POINT_ONLY
                output_0[k] = saturate_int_by_bits(
                    (rshift_rn(mult * ((int16_t) * (input + i * count + k) - izero_point), shift))
                        + ozero_point,
                    bit);
#else
                output_0[k] = saturate_float_by_bits(
                    ialpha / oalpha * (*(input + i * count + k) - izero_point) + ozero_point, bit);
#endif
            }
            output_0 += count;
        }
    }
}

void op_quant_concat_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        op->run_func = op_quant_concat_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
