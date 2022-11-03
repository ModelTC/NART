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
#include "art/quant/quant_op_tp.h"

typedef struct {
    op_t o;
    bool share;

    float *walpha;
    uint8_t *wzero_point;
    uint8_t *wbits;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_prelu_t;

op_prelu_t *op_quant_prelu_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_prelu_t *res = (op_prelu_t *)malloc(sizeof(op_prelu_t));
    memset(res, 0, sizeof(op_prelu_t));
    return res;
}

void op_quant_prelu_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_PRELU_SHARE, dtBOOL, &((op_prelu_t *)op)->share));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_prelu_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point, &((op_prelu_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_prelu_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WALPHA, dtFLOAT32, &len_alpha, &((op_prelu_t *)op)->walpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WZERO_POINT, dtUINT8, &len_zero_point, &((op_prelu_t *)op)->wzero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_WBITS, dtUINT8, &len_bits, &((op_prelu_t *)op)->wbits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_prelu_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point, &((op_prelu_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_prelu_t *)op)->obits));
}

void op_quant_prelu_tp_destroy(op_t *op) { (void)op; }

void op_quant_prelu_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_prelu_run(op_t *op)
{
    size_t n, c, i;
    size_t batch_size = op->input_tensors[0]->shape.dim[op->input_tensors[0]->shape.batch_axis];
    size_t channel = op->input_tensors[0]->shape.dim[op->input_tensors[0]->shape.channel_axis];
    size_t hw = shape_count(&(op->input_tensors[0]->shape)) / batch_size / channel;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const uint8_t *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    op_prelu_t *prelu_op = (op_prelu_t *)op;
    float walpha = prelu_op->walpha[0];
    uint8_t wzero_point = prelu_op->wzero_point[0];

    float ialpha = prelu_op->ialpha[0];
    uint8_t izero_point = prelu_op->izero_point[0];

    float oalpha = prelu_op->oalpha[0];
    uint8_t ozero_point = prelu_op->ozero_point[0];

#ifdef USE_FIXED_POINT_ONLY
    int8_t shift_0 = 0;
    uint32_t multer_0 = 0;
    int8_t shift_w = 0;
    uint32_t multer_w = 0;
    get_quant_scale(ialpha / oalpha, &multer_0, &shift_0, 16);
    get_quant_scale(walpha * ialpha / oalpha, &multer_w, &shift_w, 16);
#endif

    uint8_t bit = prelu_op->obits[0];
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel; ++c) {
            // int32_t w_tmp = rshift_rn(((int32_t)input_1[c] - wzero_point) * (int32_t)multer_w,
            // shift_w);
            int32_t w_tmp = input_1[c] - wzero_point;
            for (i = 0; i < hw; ++i) {
                int32_t tmp = *input_0 - izero_point;

#ifdef USE_FIXED_POINT_ONLY
                int32_t o = rshift_rn((int32_t)(MAX(0, tmp) * (int32_t)multer_0), shift_0)
                    + rshift_rn((int32_t)(MIN(0, tmp) * w_tmp * (int32_t)multer_w), shift_w);
                *output_0 = saturate_int_by_bits(o + ozero_point, bit);
#else
                *output_0 = saturate_float_by_bits(
                    (ialpha / oalpha) * (MAX(0, tmp) + MIN(0, tmp) * (w_tmp * walpha))
                        + ozero_point,
                    bit);
#endif
                ++input_0;
                ++output_0;
            }
        }
    }
}

static void op_quant_prelu_share_run(op_t *op)
{
    size_t i;
    size_t count = shape_count(&(op->input_tensors[0]->shape));
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    // const float* input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    const uint8_t slope = ((uint8_t *)mem_cpu_data(op->input_tensors[1]->mem))[0];
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    op_prelu_t *prelu_op = (op_prelu_t *)op;
    float walpha = prelu_op->walpha[0];
    uint8_t wzero_point = prelu_op->wzero_point[0];

    float ialpha = prelu_op->ialpha[0];
    uint8_t izero_point = prelu_op->izero_point[0];

    float oalpha = prelu_op->oalpha[0];
    uint8_t ozero_point = prelu_op->ozero_point[0];

    int8_t shift_0 = 0;
    uint32_t multer_0 = 0;
    int8_t shift_w = 0;
    uint32_t multer_w = 0;
    get_quant_scale(ialpha / oalpha, &multer_0, &shift_0, 16);
    get_quant_scale(walpha, &multer_w, &shift_w, 16);

    uint8_t bit = prelu_op->obits[0];
    int32_t w_tmp = rshift_rn((int32_t)(slope - wzero_point) * multer_w, shift_w);
    for (i = 0; i < count; ++i) {
        int16_t tmp = *input_0 - izero_point;
        *output_0 = saturate_int_by_bits(
            rshift_rn((MAX(0, tmp) + MIN(0, tmp) * w_tmp) * multer_0, shift_0) + ozero_point, bit);
        ++input_0;
        ++output_0;
    }
}
void op_quant_prelu_tp_prepare(op_t *op)
{
    int i;
    bool share;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    share = ((op_prelu_t *)op)->share;
    switch (op->input_tensors[0]->dtype) {
    case dtUINT8:
        if (share) {
            op->run_func = op_quant_prelu_share_run;
        } else {
            op->run_func = op_quant_prelu_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
