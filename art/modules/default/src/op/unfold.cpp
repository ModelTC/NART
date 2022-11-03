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

#include "../utils/im2col.hpp"
#include "../utils/sgemm.hpp"

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
op_unfold_t *op_default_unfold_tp_alloc(workspace_t *ws);
void op_default_unfold_tp_config(op_t *op);
void op_default_unfold_tp_destroy(op_t *op);
void op_default_unfold_tp_dealloc(op_t *op);
void op_default_unfold_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif
op_unfold_t *op_default_unfold_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_unfold_t *res = (op_unfold_t *)malloc(sizeof(op_unfold_t));
    memset(res, 0, sizeof(op_unfold_t));
    return res;
}

void op_default_unfold_tp_config(op_t *op)
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

void op_default_unfold_tp_destroy(op_t *op) { (void)op; }

void op_default_unfold_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_unfold_run(op_t *op)
{
    const op_unfold_t *unfold_op = (op_unfold_t *)op;
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t i_chw = op->input_tensors[0]->shape.dim[1] * op->input_tensors[0]->shape.dim[2]
        * op->input_tensors[0]->shape.dim[3];
    size_t o_cl = op->output_tensors[0].shape.dim[1] * op->output_tensors[0].shape.dim[2];
    const float *input_0 = (const float *)mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_cpu_data(op->output_tensors[0].mem);

    // size_t i;
    // for (i = 0; i < batch_size; ++i) {
    //     im2col(input_0, output_0, op->input_tensors[0]->shape.dim[1],
    //         op->input_tensors[0]->shape.dim[2], op->input_tensors[0]->shape.dim[3],
    //         unfold_op->kernel_h, unfold_op->kernel_w,
    //         unfold_op->pad_h, unfold_op->pad_w,
    //         unfold_op->stride_h, unfold_op->stride_w,
    //         unfold_op->hole_h, unfold_op->hole_w);

    //    input_0 += i_chw;
    //    output_0 += o_cl;
    //}

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
    size_t kernel_h_eff = kernel_h + (kernel_h - 1) * (hole_h - 1);
    size_t kernel_w_eff = kernel_w + (kernel_w - 1) * (hole_w - 1);
    size_t height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
    size_t width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;

    size_t n, c, h, w, hw, k;
    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < op->input_tensors[0]->shape.dim[1]; ++c) {
            for (k = 0; k < unfold_op->kernel_h * unfold_op->kernel_w; ++k) {
                for (hw = 0; hw < op->output_tensors[0].shape.dim[2]; ++hw) {
                    h = hw / width_col;
                    w = hw % width_col;
                    int h_padt = h * stride_h - pad_h;
                    int w_padt = w * stride_w - pad_w;

                    int h_pad = h_padt + k / kernel_w % kernel_h * hole_h;
                    int w_pad = w_padt + k % kernel_w * hole_w;
                    int a = h_pad >= 0 && h_pad < (int)height && w_pad >= 0 && w_pad < (int)width;
                    if (a)
                        *output_0 = input_0[(c * height + h_pad) * width + w_pad];
                    else
                        *output_0 = 0;
                    output_0++;
                }
            }
        }

        // batch loop
        input_0 += i_chw;
    }
}

void op_default_unfold_tp_prepare(op_t *op)
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
        op->run_func = op_default_unfold_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
