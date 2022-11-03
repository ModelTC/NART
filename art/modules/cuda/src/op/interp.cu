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

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t height;
    uint32_t width;
    uint32_t zoom_factor;
    uint32_t shrink_factor;
    uint32_t pad_beg;
    uint32_t pad_end;
    uint32_t type;
} op_interp_t;

op_interp_t *op_cuda_interp_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_interp_t *res = (op_interp_t *)malloc(sizeof(op_interp_t));
    memset(res, 0, sizeof(op_interp_t));
    return res;
}

void op_cuda_interp_tp_config(op_t *op)
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
    CHECK(op_setting_single_get(op, SETTING_INTERP_TYPE, dtUINT32, &((op_interp_t *)op)->type));
}

void op_cuda_interp_tp_destroy(op_t *op) { (void)op; }

void op_cuda_interp_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_interp_kernel(
    const int n, const float rheight, const float rwidth, const int channels, const float *data1,
    const int x1, const int y1, const int height1, const int width1, const int Height1,
    const int Width1, float *data2, const int x2, const int y2, const int height2, const int width2,
    const int Height2, const int Width2)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int w2 = index % width2; // 0:width2-1
        const int h2 = index / width2; // 0:height2-1
        // special case: just copy
        if (height1 == height2 && width1 == width2) {
            const int h1 = h2;
            const int w1 = w2;
            const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
            float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
            for (int c = 0; c < channels; ++c) {
                pos2[0] = pos1[0];
                pos1 += Width1 * Height1;
                pos2 += Width2 * Height2;
            }
            return;
        }
        //
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const float h1lambda = h1r - h1;
        const float h0lambda = float(1.) - h1lambda;
        //
        const float w1r = rwidth * w2;
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const float w1lambda = w1r - w1;
        const float w0lambda = float(1.) - w1lambda;
        //
        const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
            pos2[0] = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p])
                + h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
        }
    }
}

static void op_cuda_interp_run(op_t *op)
{
    const op_interp_t *interp_op = (op_interp_t *)op;
    int pad_beg = interp_op->pad_beg;
    int pad_end = interp_op->pad_end;
    int i_h = op->input_tensors[0]->shape.dim[2];
    int i_w = op->input_tensors[0]->shape.dim[3];
    int o_h = op->output_tensors[0].shape.dim[2];
    int o_w = op->output_tensors[0].shape.dim[3];
    int i_w_eff = i_w - pad_beg - pad_end;
    int i_h_eff = i_h - pad_beg - pad_end;

    int nc = op->input_tensors[0]->shape.dim[0] * op->input_tensors[0]->shape.dim[1];
    const float rheight = (o_h > 1) ? (i_h_eff - 1.0) / (o_h - 1.0) : 0;
    const float rwidth = (o_w > 1) ? (i_w_eff - 1.0) / (o_w - 1.0) : 0;

    const float *input = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    op_cuda_interp_kernel<<<
        (o_h * o_w + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        o_h * o_w, rheight, rwidth, nc, input, -pad_beg, -pad_beg, i_h_eff, i_w_eff, i_h, i_w,
        output, 0, 0, o_h, o_w, o_h, o_w);
}

__global__ void op_cuda_nninterp_kernel(
    const int n, const float rheight, const float rwidth, const int channels, const float *data1,
    const int x1, const int y1, const int height1, const int width1, const int Height1,
    const int Width1, float *data2, const int x2, const int y2, const int height2, const int width2,
    const int Height2, const int Width2)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int w2 = index % width2; // 0:width2-1
        const int h2 = index / width2; // 0:height2-1
        // special case: just copy
        if (height1 == height2 && width1 == width2) {
            const int h1 = h2;
            const int w1 = w2;
            const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
            float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
            for (int c = 0; c < channels; ++c) {
                pos2[0] = pos1[0];
                pos1 += Width1 * Height1;
                pos2 += Width2 * Height2;
            }
            return;
        }
        //
        const float h1r = rheight * h2;
        const int h1 = h1r;
        const int h1p = (h1 < height1 - 1) ? 1 : 0;
        const int h1lambda = (h1r <= h1 + 1) ? 0 : 1;
        const int h0lambda = 1 - h1lambda;
        //
        const float w1r = rwidth * w2;
        const int w1 = w1r;
        const int w1p = (w1 < width1 - 1) ? 1 : 0;
        const int w1lambda = (w1r <= w1 + 1) ? 0 : 1;
        const int w0lambda = 1 - w1lambda;
        //
        const float *pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
        float *pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
        for (int c = 0; c < channels; ++c) {
            pos2[0] = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p])
                + h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
            pos1 += Width1 * Height1;
            pos2 += Width2 * Height2;
        }
    }
}

static void op_cuda_nninterp_run(op_t *op)
{
    const op_interp_t *nninterp_op = (op_interp_t *)op;
    int pad_beg = nninterp_op->pad_beg;
    int pad_end = nninterp_op->pad_end;
    int i_h = op->input_tensors[0]->shape.dim[2];
    int i_w = op->input_tensors[0]->shape.dim[3];
    int o_h = op->output_tensors[0].shape.dim[2];
    int o_w = op->output_tensors[0].shape.dim[3];
    int i_w_eff = i_w - pad_beg - pad_end;
    int i_h_eff = i_h - pad_beg - pad_end;

    int nc = op->input_tensors[0]->shape.dim[0] * op->input_tensors[0]->shape.dim[1];
    const float rheight = (o_h > 1) ? (float)(i_h_eff) / (o_h) : 0;
    const float rwidth = (o_w > 1) ? (float)(i_w_eff) / (o_w) : 0;
    const float *input = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    op_cuda_nninterp_kernel<<<
        (o_h * o_w + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        o_h * o_w, rheight, rwidth, nc, input, -pad_beg, -pad_beg, i_h_eff, i_w_eff, i_h, i_w,
        output, 0, 0, o_h, o_w, o_h, o_w);
}

void op_cuda_interp_tp_prepare(op_t *op)
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
        break;
    default:
        CHECK(false);
        break;
    }
    switch (((op_interp_t *)op)->type) {
    case SETTING_INTERP_BILINEAR:
        op->run_func = op_cuda_interp_run;
        break;
    case SETTING_INTERP_NEAREST:
        op->run_func = op_cuda_nninterp_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
