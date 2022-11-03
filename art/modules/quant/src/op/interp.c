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
    uint32_t height;
    uint32_t width;
    uint32_t zoom_factor;
    uint32_t shrink_factor;
    uint32_t pad_beg;
    uint32_t pad_end;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_interp_t;

op_interp_t *op_quant_interp_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_interp_t *res = (op_interp_t *)malloc(sizeof(op_interp_t));
    memset(res, 0, sizeof(op_interp_t));
    return res;
}

void op_quant_interp_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_INTERP_HEIGHT, dtUINT32, &((op_interp_t *)op)->height));
    CHECK(op_setting_single_get(op, SETTING_INTERP_WIDTH, dtUINT32, &((op_interp_t *)op)->width));
    CHECK(op_setting_single_get(
        op, SETTING_INTERP_ZOOM_FACTOR, dtUINT32, &((op_interp_t *)op)->zoom_factor));
    CHECK(op_setting_single_get(
        op, SETTING_INTERP_SHRINK_FACTOR, dtUINT32, &((op_interp_t *)op)->shrink_factor));
    CHECK(
        op_setting_single_get(op, SETTING_INTERP_PAD_BEG, dtUINT32, &((op_interp_t *)op)->pad_beg));
    CHECK(
        op_setting_single_get(op, SETTING_INTERP_PAD_END, dtUINT32, &((op_interp_t *)op)->pad_end));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_interp_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_interp_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_interp_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_interp_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_interp_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_interp_t *)op)->obits));
}

void op_quant_interp_tp_destroy(op_t *op) { (void)op; }

void op_quant_interp_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_interp_run(op_t *op)
{
    const op_interp_t *interp_op = (op_interp_t *)op;
    int pad_beg = interp_op->pad_beg;
    int pad_end = interp_op->pad_end;
    int i_h = op->input_tensors[0]->shape.dim[2];
    int i_w = op->input_tensors[0]->shape.dim[3];
    int o_h = op->output_tensors[0].shape.dim[2];
    int o_w = op->output_tensors[0].shape.dim[3];
    int i_area = i_h * i_w;
    int o_area = o_h * o_w;
    int i_w_eff = i_w - pad_beg - pad_end;
    int i_h_eff = i_h - pad_beg - pad_end;
    int i, j, k;
    int nc = op->input_tensors[0]->shape.dim[0] * op->input_tensors[0]->shape.dim[1];
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    float ialpha = interp_op->ialpha[0];
    uint8_t izero_point = interp_op->izero_point[0];

    float oalpha = interp_op->oalpha[0];
    uint8_t ozero_point = interp_op->ozero_point[0];

    uint32_t result_mult;
    int8_t result_shift;

    result_mult = 0;
    result_shift = 0;
    get_quant_scale(ialpha / oalpha, &result_mult, &result_shift, 24);

    const uint8_t *pos1 = NULL;
    uint8_t *pos2 = NULL;
    uint8_t bit = interp_op->obits[0];
    // effect area shape == output shape
    if (i_h_eff == o_h && i_w_eff == o_w) {
        for (i = 0; i < o_h; ++i) {
            for (j = 0; j < o_w; ++j) {
                pos1 = &input_0[(pad_beg + i) * i_w + pad_beg + j];
                pos2 = &output_0[i * o_w + j];
                for (k = 0; k < nc; ++k) {
                    pos2[0] = saturate_int_by_bits(
                        rshift_rn(result_mult * (pos1[0] - izero_point), result_shift)
                            + ozero_point,
                        bit);
                    pos1 += i_area;
                    pos2 += o_area;
                }
            }
        }
        return;
    }

#if 0
int h1, h1p, w1, w1p;
float h1r, h1lambda, h0lambda;
float w1r, w1lambda, w0lambda;
const float rheight = (o_h > 1) ? (i_h_eff - 1.0) / (o_h - 1.0) : 0;
const float rwidth = (o_w > 1) ? (i_w_eff - 1.0) / (o_w - 1.0) : 0;
for (i = 0; i < o_h; ++i) {
h1r = rheight * i;
h1 = (int)h1r;
h1p = (h1 < i_h_eff - 1) ? 1 : 0;
h1lambda = h1r - h1;
h0lambda = 1.0 - h1lambda;
for (j = 0; j < o_w; ++j) {
w1r = rwidth * j;
w1 = (int)w1r;
w1p = (w1 < i_w_eff) ? 1 : 0;
w1lambda = w1r - w1;
w0lambda = 1.0 - w1lambda;
pos1 = &input_0[(pad_beg + h1) * i_w + w1];
pos2 = &output_0[i * o_w + j];
for (k = 0; k < nc; ++k) {
pos2[0] = saturate_float_by_bits(h0lambda * (w0lambda * (pos1[0] - izero_point) * ialpha + w1lambda * (pos1[w1p] - izero_point) * ialpha) + \
h1lambda * (w0lambda * (pos1[h1p * i_w] - izero_point) * ialpha + \
w1lambda * (pos1[h1p * i_w + w1p] - izero_point) * ialpha) / oalpha + ozero_point, interp_op->bits);
pos1 += i_area;
pos2 += o_area;
}
}
}
#else
    // use fixed-point only
    int h1, h1p, w1, w1p;
    int h1lambda, h0lambda;
    int w1lambda, w0lambda;
    const int rheight = (o_h > 1) ? ((i_h_eff - 1) << 16) / (o_h - 1) : 0;
    const int rwidth = (o_w > 1) ? ((i_w_eff - 1) << 16) / (o_w - 1) : 0;
    const int ibias = 255 * 255 * izero_point;

    for (i = 0; i < o_h; ++i) {
        h1 = (rheight * i) >> 16;
        h1p = (h1 < i_h_eff - 1) ? 1 : 0;
        h1lambda = ((rheight * i) >> 8) & 0xff;
        h0lambda = 255 - h1lambda;

        for (j = 0; j < o_w; ++j) {
            w1 = (rwidth * j) >> 16;
            w1p = (w1 < i_w_eff) ? 1 : 0;
            w1lambda = ((rwidth * j) >> 8) & 0xff;
            w0lambda = 255 - w1lambda;

            pos1 = &input_0[(pad_beg + h1) * i_w + w1];
            pos2 = &output_0[i * o_w + j];
            for (k = 0; k < nc; ++k) {

                /**** if ialpha == oalpha && izero_point == ozero_point
                uint16_t x0 = w0lambda * (pos1[0]) + w1lambda * (pos1[w1p]);
                uint16_t x1 = w0lambda * (pos1[h1p * i_w]) + w1lambda * (pos1[h1p * i_w + w1p]);
                pos2[0] = saturate_int_by_bits(rshift_rn(h0lambda * x0 + h1lambda * x1, 16),
                interp_op->bits);
                */

                uint16_t x0 = w0lambda * (pos1[0]) + w1lambda * (pos1[w1p]);
                uint16_t x1 = w0lambda * (pos1[h1p * i_w]) + w1lambda * (pos1[h1p * i_w + w1p]);
                pos2[0] = saturate_int_by_bits(
                    rshift_rn(
                        (uint64_t)result_mult
                            * ((h0lambda * x0 + h1lambda * x1 + (1 << 15) - ibias) >> 16),
                        result_shift)
                        + ozero_point,
                    bit);

                pos1 += i_area;
                pos2 += o_area;
            }
        }
    }

#endif
}

void op_quant_interp_tp_prepare(op_t *op)
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
        op->run_func = op_quant_interp_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
