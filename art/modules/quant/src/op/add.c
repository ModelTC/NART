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

typedef struct {
    op_t o;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;

    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;

} op_add_t;

op_add_t *op_quant_add_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_add_t *res = (op_add_t *)malloc(sizeof(op_add_t));
    memset(res, 0, sizeof(op_add_t));
    return res;
}

void op_quant_add_tp_config(op_t *op)
{
    size_t len_ialpha;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_ialpha, &((op_add_t *)op)->ialpha));

    size_t len_izero_point;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_izero_point, &((op_add_t *)op)->izero_point));

    size_t len_ibits;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_ibits, &((op_add_t *)op)->ibits));

    size_t len_oalpha;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_oalpha, &((op_add_t *)op)->oalpha));

    size_t len_ozero_point;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_ozero_point, &((op_add_t *)op)->ozero_point));

    size_t len_obits;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_obits, &((op_add_t *)op)->obits));
}

void op_quant_add_tp_destroy(op_t *op) { (void)op; }

void op_quant_add_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_add_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    size_t i;
    const uint8_t *input_0;
    const uint8_t *input_1;
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    const op_add_t *add_op = (op_add_t *)op;

    float i0alpha = add_op->ialpha[0];
    uint8_t i0zero_point = add_op->izero_point[0];

    float i1alpha = add_op->ialpha[1];
    uint8_t i1zero_point = add_op->izero_point[1];

    float oalpha = add_op->oalpha[0];
    uint8_t ozero_point = add_op->ozero_point[0];
    input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    input_1 = mem_cpu_data(op->input_tensors[1]->mem);

    if (i0alpha > i1alpha) {
        {
            float tmp = i0alpha;
            i0alpha = i1alpha;
            i1alpha = tmp;
            uint8_t tmpp = i0zero_point;
            i0zero_point = i1zero_point;
            i1zero_point = tmpp;
        }
        {
            const uint8_t *tmp = input_0;
            input_0 = input_1;
            input_1 = tmp;
        }
    }

#ifdef USE_FIXED_POINT_ONLY
    int8_t shift_0 = 0;
    uint32_t multer_0 = 0;
    int8_t shift_1 = 0;
    uint32_t multer_1 = 0;

    get_quant_scale(i0alpha / oalpha, &multer_0, &shift_0, 23);
    get_quant_scale(i1alpha / oalpha, &multer_1, &shift_1, 23);

    // TBD
    int8_t shift = (shift_0 > shift_1 ? shift_1 : shift_0) - 1;
    uint8_t bit = add_op->obits[0];
#endif

    /* output_0 = (i1alhpa / oalpha) ((i0alpha / i1alpha) * (input_0 - i0zero) + (input_1 -
     * i1zero))*/
    for (i = 0; i < count; ++i) {
        int32_t i0 = (int16_t)input_0[i] - i0zero_point;
        int32_t i1 = (int16_t)input_1[i] - i1zero_point;
#ifdef USE_FIXED_POINT_ONLY
        int32_t tmp = rshift_rn((int32_t)(multer_0 * (int32_t)i0), shift_0 - shift)
            + rshift_rn((int32_t)(multer_1 * (int32_t)i1), shift_1 - shift);
        output_0[i] = saturate_int_by_bits(rshift_rn(tmp, shift) + ozero_point, bit);
#else
        output_0[i]
            = saturate_float_by_bits((i0alpha * i0 + i1alpha * i1) / oalpha + ozero_point, bit);
#endif
    }
}

void op_quant_add_tp_prepare(op_t *op)
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
        op->run_func = op_quant_add_run;
        break;
    default:
        printf("name: %s\n", op->input_tensors[0]->name);
        CHECK(false);
        break;
    }
}
