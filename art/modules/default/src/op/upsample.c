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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"
#include "art/tensor.h"

typedef struct {
    op_t o;
    uint32_t mode;
} op_upsample_t;

op_upsample_t *op_default_upsample_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_upsample_t *res = (op_upsample_t *)malloc(sizeof(op_upsample_t));
    memset(res, 0, sizeof(op_upsample_t));
    return res;
}

void op_default_upsample_tp_config(op_t *op)
{
    size_t len;
    CHECK(op_setting_single_get(op, SETTING_PAD_MODE, dtUINT32, &((op_upsample_t *)op)->mode));
}

void op_default_upsample_tp_destroy(op_t *op) { (void)op; }

void op_default_upsample_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void UpsampleBilinearOp(
    const int batch_size, const int num_channels, const int input_height, const int input_width,
    const float *input0, const float *input1, float *output)
{

    float height_scale_ = input1[2];
    float width_scale_ = input1[3];
    int output_width = input_width * width_scale_;
    int output_height = input_height * height_scale_;
    int channels = num_channels * batch_size;
    const float rheight
        = (output_height > 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
    const float rwidth = (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
    int h2, w2, c;
    for (h2 = 0; h2 < output_height; ++h2) {
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = (float)1. - h1lambda;
        for (w2 = 0; w2 < output_width; ++w2) {
            const float w1r = rwidth * w2;
            const int w1 = w1r;
            const int w1p = (w1 < input_width - 1) ? 1 : 0;
            const float w1lambda = w1r - w1;
            const float w0lambda = (float)1. - w1lambda;
            const float *Xdata = &input1[h1 * input_width + w1];
            float *Ydata = &output[h2 * output_width + w2];
            for (c = 0; c < channels; ++c) {
                Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p])
                    + h1lambda
                        * (w0lambda * Xdata[h1p * input_width]
                           + w1lambda * Xdata[h1p * input_width + w1p]);
                Xdata += input_width * input_height;
                Ydata += output_width * output_height;
            }
        }
    }
}

static void op_default_upsample_run(op_t *op)
{
    uint32_t mode = ((op_upsample_t *)op)->mode;
    int batch_size, i_h, i_w, i_c;
    i_c = op->input_tensors[0]->shape.dim[1];
    i_h = op->input_tensors[0]->shape.dim[2];
    i_w = op->input_tensors[0]->shape.dim[3];
    batch_size = op->input_tensors[0]->shape.dim[0];
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
#define BILINEAR 1
    switch (mode) {
    case BILINEAR:
        UpsampleBilinearOp(batch_size, i_c, i_h, i_w, input_0, input_1, output_0);
        break;
    default:
        break;
    }
#undef BILINEAR
    (void)op;
}

void op_default_upsample_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtUNKNOWN:
        CHECK(false);
        break;
    default:
        op->run_func = op_default_upsample_run;
        break;
    }
}
