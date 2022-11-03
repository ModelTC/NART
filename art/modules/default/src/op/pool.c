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

#include "../utils/utils.h"

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
} op_pool_t;

op_pool_t *op_default_pool_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_pool_t *res = (op_pool_t *)malloc(sizeof(op_pool_t));
    memset(res, 0, sizeof(op_pool_t));
    return res;
}

void op_default_pool_tp_config(op_t *op)
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
}

void op_default_pool_tp_destroy(op_t *op) { (void)op; }

void op_default_pool_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_pool_max_run(op_t *op)
{
    int i, j, n, c, h, w;
    int i_h, i_w, i_hw;
    int o_w, o_h, o_c;
    int i_offset, batch_size;
    int hstart, hend, wstart, wend;
    int stride_h, stride_w;
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    float max_value;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
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
                    max_value = -FLT_MAX;
                    for (i = hstart; i < hend; ++i) {
                        for (j = wstart; j < wend; ++j) {
                            i_offset = i * i_w + j;
                            if (input_0[i_offset] > max_value) {
                                max_value = input_0[i_offset];
                            }
                        }
                    }
                    *output_0++ = max_value;
                }
            }
            input_0 += i_hw;
        }
    }
}

static void op_default_pool_average_run(op_t *op)
{
    int i, j, n, c, h, w;
    int i_h, i_w, i_hw;
    int o_w, o_h, o_c;
    int batch_size;
    int hstart, hend, wstart, wend, pool_size;
    int stride_h, stride_w;
    int kernel_h, kernel_w;
    int pad_h, pad_w;
    float sum;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
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
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < o_c; ++c) {
            for (h = 0; h < o_h; ++h) {
                for (w = 0; w < o_w; ++w) {
                    hstart = h * stride_h - pad_h;
                    wstart = w * stride_w - pad_w;
                    hend = MIN(hstart + kernel_h, i_h + pad_h);
                    wend = MIN(wstart + kernel_w, i_w + pad_w);
                    pool_size = (hend - hstart) * (wend - wstart);
                    hstart = MAX(hstart, 0);
                    wstart = MAX(wstart, 0);
                    hend = MIN(hend, i_h);
                    wend = MIN(wend, i_w);
                    sum = 0;
                    for (i = hstart; i < hend; ++i) {
                        for (j = wstart; j < wend; ++j) {
                            sum += input_0[i * i_w + j];
                        }
                    }
                    *output_0++ = sum / pool_size;
                }
            }
            input_0 += i_hw;
        }
    }
}

void op_default_pool_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        CHECK(
            ((op_pool_t *)op)->pool_method == SETTING_POOL_MAX
            || ((op_pool_t *)op)->pool_method == SETTING_POOL_AVE);
        if (((op_pool_t *)op)->pool_method == SETTING_POOL_MAX) {
            op->run_func = op_default_pool_max_run;
        } else {
            op->run_func = op_default_pool_average_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}
