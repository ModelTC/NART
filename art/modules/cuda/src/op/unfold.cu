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

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t hole_h;
    uint32_t hole_w;
} op_unfold_t;

op_unfold_t *op_cuda_unfold_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_unfold_t *res = (op_unfold_t *)malloc(sizeof(op_unfold_t));
    memset(res, 0, sizeof(op_unfold_t));
    return res;
}

void op_cuda_unfold_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_UNFOLD_KERNEL_H, dtUINT32, &((op_unfold_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_UNFOLD_KERNEL_W, dtUINT32, &((op_unfold_t *)op)->kernel_w));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_PAD_H, dtUINT32, &((op_unfold_t *)op)->pad_h));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_PAD_W, dtUINT32, &((op_unfold_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_UNFOLD_STRIDE_H, dtUINT32, &((op_unfold_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_UNFOLD_STRIDE_W, dtUINT32, &((op_unfold_t *)op)->stride_w));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_HOLE_H, dtUINT32, &((op_unfold_t *)op)->hole_h));
    CHECK(op_setting_single_get(op, SETTING_UNFOLD_HOLE_W, dtUINT32, &((op_unfold_t *)op)->hole_w));
}

void op_cuda_unfold_tp_destroy(op_t *op) { (void)op; }

void op_cuda_unfold_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_unfold_kernel(
    float *output, float *input, size_t size, int kernel_h, int kernel_w, int pad_h, int pad_w,
    int stride_h, int stride_w, int hole_h, int hole_w, size_t height, size_t width,
    size_t width_col, size_t i_c, size_t u_hw, size_t o_hw, size_t i_chw)
{
    CUDA_KERNEL_LOOP(index, size)
    {
        // size_t n = index / (i_c * u_hw * o_hw);
        size_t c = (index / (u_hw * o_hw)) % i_c;
        size_t k = (index / o_hw) % u_hw;
        size_t hw = index % o_hw;

        size_t h = hw / width_col;
        size_t w = hw % width_col;
        int h_padt = h * stride_h - pad_h;
        int w_padt = w * stride_w - pad_w;

        int h_pad = h_padt + k / kernel_w % kernel_h * hole_h;
        int w_pad = w_padt + k % kernel_w * hole_w;
        int a = h_pad >= 0 && h_pad < (int)height && w_pad >= 0 && w_pad < (int)width;

        float *output_offset = output + index;
        float *input_offset = input + ((index / (i_c * u_hw * o_hw)) * i_chw);

        if (a)
            *output_offset = input_offset[(c * height + h_pad) * width + w_pad];
        else
            *output_offset = 0;
    }
}

static void op_cuda_unfold_run(op_t *op)
{
    const op_unfold_t *unfold_op = (op_unfold_t *)op;
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t i_chw = op->input_tensors[0]->shape.dim[1] * op->input_tensors[0]->shape.dim[2]
        * op->input_tensors[0]->shape.dim[3];

    float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);

    int kernel_h = unfold_op->kernel_h;
    int kernel_w = unfold_op->kernel_w;
    int pad_h = unfold_op->pad_h;
    int pad_w = unfold_op->pad_w;
    int stride_h = unfold_op->stride_h;
    int stride_w = unfold_op->stride_w;
    int hole_h = unfold_op->hole_h;
    int hole_w = unfold_op->hole_w;
    size_t height = op->input_tensors[0]->shape.dim[2];
    size_t width = op->input_tensors[0]->shape.dim[3];
    // size_t kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
    size_t kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
    size_t width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
    size_t i_c = op->input_tensors[0]->shape.dim[1];
    size_t u_hw = unfold_op->kernel_h * unfold_op->kernel_w;
    size_t o_hw = op->output_tensors[0].shape.dim[2];

    op_cuda_unfold_kernel<<<
        (batch_size * i_c * u_hw * o_hw + 1024 - 1) / 1024, 1024, 0,
        CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output_0, input_0, batch_size * i_c * u_hw * o_hw, kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, hole_h, hole_w, height, width, width_col, i_c, u_hw, o_hw, i_chw);
}

void op_cuda_unfold_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_unfold_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
