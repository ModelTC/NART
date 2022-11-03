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
    uint32_t axis;
} op_softmax_t;

op_softmax_t *op_cuda_softmax_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_softmax_t *res = (op_softmax_t *)malloc(sizeof(op_softmax_t));
    memset(res, 0, sizeof(op_softmax_t));
    return res;
}

void op_cuda_softmax_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_SOFTMAX_AXIS, dtUINT32, &((op_softmax_t *)op)->axis));
}

void op_cuda_softmax_tp_destroy(op_t *op) { (void)op; }

void op_cuda_softmax_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_softmax_kernel(
    size_t n, float *dst, size_t outer_num, size_t ch, size_t inner_num, const float *src)
{
    CUDA_KERNEL_LOOP(i, n)
    {
        size_t outer_idx = i / inner_num;
        size_t inner_idx = i % inner_num;
        size_t ch_idx;
        float val = 0.f;
        float max = -FLT_MAX;
        float *dst_p = dst + outer_idx * ch * inner_num + inner_idx;
        const float *src_p = src + outer_idx * ch * inner_num + inner_idx;
        for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
            if (max < src_p[inner_num * ch_idx]) {
                max = src_p[inner_num * ch_idx];
            }
        }
        for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
            val += (*dst_p = __expf(*src_p - max));
            src_p += inner_num;
            dst_p += inner_num;
        }
        dst_p = dst + outer_idx * ch * inner_num + inner_idx;
        for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
            *dst_p /= val;
            dst_p += inner_num;
        }
    }
}

static void op_cuda_softmax_run(op_t *op)
{
    const op_softmax_t *softmax_op = (op_softmax_t *)op;
    size_t axis = softmax_op->axis;
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_num = shape_part_count(&op->input_tensors[0]->shape, softmax_op->axis) / ch;
    size_t count = shape_count(&op->output_tensors[0].shape);
    size_t outer_num = count / inner_num / ch;

    const float *input = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    op_cuda_softmax_kernel<<<
        (inner_num * outer_num + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        inner_num * outer_num, output, outer_num, ch, inner_num, input);
}

void op_cuda_softmax_tp_prepare(op_t *op)
{
    int i;
    const op_softmax_t *softmax_op = (op_softmax_t *)op;
    size_t axis = softmax_op->axis;
    CHECK_GT((size_t)op->input_tensors[0]->shape.dim_size, axis);
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_softmax_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
