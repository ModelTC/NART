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

#include <float.h>
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
    uint32_t pool_method; //目前默认使用MAX
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    bool ceil_mode;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_pool_t;

op_pool_t *op_quant_pool_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_pool_t *res = (op_pool_t *)malloc(sizeof(op_pool_t));
    memset(res, 0, sizeof(op_pool_t));
    return res;
}

void op_quant_pool_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_POOL_KERNEL_H, dtUINT32, &((op_pool_t *)op)->kernel_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_KERNEL_W, dtUINT32, &((op_pool_t *)op)->kernel_w));
    CHECK(
        op_setting_single_get(op, SETTING_POOL_METHOD, dtUINT32, &((op_pool_t *)op)->pool_method));
    CHECK(op_setting_single_get(op, SETTING_POOL_PAD_H, dtUINT32, &((op_pool_t *)op)->pad_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_PAD_W, dtUINT32, &((op_pool_t *)op)->pad_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_STRIDE_H, dtUINT32, &((op_pool_t *)op)->stride_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_STRIDE_W, dtUINT32, &((op_pool_t *)op)->stride_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_CEIL_MODE, dtBOOL, &((op_pool_t *)op)->ceil_mode));

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_pool_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IZERO_POINT, dtUINT8, &len_zero_point, &((op_pool_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_IBITS, dtUINT8, &len_bits, &((op_pool_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_pool_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OZERO_POINT, dtUINT8, &len_zero_point, &((op_pool_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_OBITS, dtUINT8, &len_bits, &((op_pool_t *)op)->obits));
}

void op_quant_pool_tp_destroy(op_t *op) { (void)op; }

void op_quant_pool_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_quant_pool_max_run(op_t *op)
{
    int i, j, n, c, h, w;
    int i_h, i_w, i_hw;
    int o_w, o_h, o_c;
    int i_offset, batch_size;
    int hstart, hend, wstart, wend;
    int stride_h, stride_w;
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    uint8_t max_value;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    const op_pool_t *pool_op = (op_pool_t *)op;
    i_h = op->input_tensors[0]->shape.dim[2];
    i_w = op->input_tensors[0]->shape.dim[3];
    i_hw = i_h * i_w;
    o_c = op->output_tensors[0].shape.dim[1];
    o_h = op->output_tensors[0].shape.dim[2];
    o_w = op->output_tensors[0].shape.dim[3];
    batch_size = op->input_tensors[0]->shape.dim[0];
    stride_h = pool_op->stride_h;
    stride_w = pool_op->stride_w;
    kernel_h = pool_op->kernel_h;
    kernel_w = pool_op->kernel_w;
    pad_h = pool_op->pad_h;
    pad_w = pool_op->pad_w;
    float ialpha = pool_op->ialpha[0];
    uint8_t izero_point = pool_op->izero_point[0];
    float oalpha = pool_op->oalpha[0];
    uint8_t ozero_point = pool_op->ozero_point[0];

#ifdef USE_FIXED_POINT_ONLY
    int8_t shift = 0;
    uint16_t mult = 0;
    get_quant_scale_int16(ialpha / oalpha, &mult, &shift);
#endif

    uint8_t bit = pool_op->obits[0];
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < o_c; ++c) {
            for (h = 0; h < o_h; ++h) {
                for (w = 0; w < o_w; ++w) {
                    hstart = h * stride_h - pad_h;
                    wstart = w * stride_w - pad_w;
                    hend = MIN(hstart + kernel_h, i_h);
                    wend = MIN(wstart + kernel_w, i_w);
                    hstart = MAX(hstart, 0);
                    wstart = MAX(wstart, 0);
                    max_value = 0;
                    for (i = hstart; i < hend; ++i) {
                        for (j = wstart; j < wend; ++j) {
                            i_offset = i * i_w + j;
                            if (input_0[i_offset] > max_value) {
                                max_value = input_0[i_offset];
                            }
                        }
                    }
#ifdef USE_FIXED_POINT_ONLY
                    *output_0++ = saturate_int_by_bits(
                        rshift_rn((int32_t)mult * ((int16_t)max_value - izero_point), shift)
                            + ozero_point,
                        bit);
#else
                    *output_0++ = saturate_float_by_bits(
                        ialpha / oalpha * ((int16_t)max_value - izero_point) + ozero_point, bit);
#endif
                }
            }
            input_0 += i_hw;
        }
    }
}

static void op_quant_pool_average_run(op_t *op)
{
    int i, j, n, c, h, w;
    int i_h, i_w, i_hw, i_chw;
    int o_w, o_h, o_c;
    int batch_size;
    int hstart, hend, wstart, wend;
    uint32_t pool_size;
    int stride_h, stride_w;
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    int32_t sum;
    const uint8_t *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    uint8_t *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    const op_pool_t *pool_op = (op_pool_t *)op;
    i_h = op->input_tensors[0]->shape.dim[2];
    i_w = op->input_tensors[0]->shape.dim[3];
    i_hw = i_h * i_w;
    i_chw = i_hw * op->input_tensors[0]->shape.dim[1];
    o_c = op->output_tensors[0].shape.dim[1];
    o_h = op->output_tensors[0].shape.dim[2];
    o_w = op->output_tensors[0].shape.dim[3];
    batch_size = op->input_tensors[0]->shape.dim[0];
    stride_h = pool_op->stride_h;
    stride_w = pool_op->stride_w;
    kernel_h = pool_op->kernel_h;
    kernel_w = pool_op->kernel_w;
    pad_h = pool_op->pad_h;
    pad_w = pool_op->pad_w;

    float ialpha = ((op_pool_t *)op)->ialpha[0];
    uint8_t izero_point = ((op_pool_t *)op)->izero_point[0];
    float oalpha = ((op_pool_t *)op)->oalpha[0];
    uint8_t ozero_point = ((op_pool_t *)op)->ozero_point[0];

#ifdef USE_FIXED_POINT_ONLY
    int8_t shift = 0;
    uint32_t mult = 0;
    get_quant_scale(ialpha / oalpha, &mult, &shift, 16);
#endif

    uint8_t bit = pool_op->obits[0];
    for (n = 0; n < batch_size; ++n) {
        for (h = 0; h < o_h; ++h) {
            for (w = 0; w < o_w; ++w) {
                hstart = h * stride_h - pad_h;
                wstart = w * stride_w - pad_w;
                hend = MIN(hstart + kernel_h, i_h + pad_h);
                wend = MIN(wstart + kernel_w, i_w + pad_w);
                pool_size = (hend - hstart) * (wend - wstart);
#ifdef USE_FIXED_POINT_ONLY
                get_quant_scale(ialpha / oalpha / pool_size, &mult, &shift, 16);
#endif

                hstart = MAX(hstart, 0);
                wstart = MAX(wstart, 0);
                hend = MIN(hend, i_h);
                wend = MIN(wend, i_w);
                for (c = 0; c < o_c; ++c) {
                    sum = 0;
                    const uint8_t *in0 = &input_0[n * i_chw + c * i_hw];
                    for (i = hstart; i < hend; ++i) {
                        for (j = wstart; j < wend; ++j) {
                            sum += (int16_t)in0[i * i_w + j] - izero_point;
                        }
                    }
                    uint8_t *out0 = &output_0[n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w];
#ifdef USE_FIXED_POINT_ONLY
                    *out0 = saturate_int_by_bits(
                        rshift_rn((int32_t)mult * sum, shift) + ozero_point, bit);
#else
                    *out0 = saturate_float_by_bits(
                        ialpha * sum / pool_size / oalpha + ozero_point, bit);
#endif
                }
            }
            // input_0 += i_hw;
        }
    }
}

void op_quant_pool_tp_prepare(op_t *op)
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
        if (((op_pool_t *)op)->pool_method == 0) {
            op->run_func = op_quant_pool_max_run;
        } else {
            op->run_func = op_quant_pool_average_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
