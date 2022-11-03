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

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "cuda_runtime.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ThreadsPerBlock 1024

typedef struct {
    op_t o;
    int32_t p;
    int32_t axis;
} op_lpnormalization_t;

op_lpnormalization_t *op_cuda_lpnormalization_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_lpnormalization_t *res = (op_lpnormalization_t *)malloc(sizeof(op_lpnormalization_t));
    memset(res, 0, sizeof(op_lpnormalization_t));
    return res;
}

void op_cuda_lpnormalization_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_LPNORMALIZATION_P, dtINT32, &((op_lpnormalization_t *)op)->p));
    CHECK(op_setting_single_get(
        op, SETTING_LPNORMALIZATION_AXIS, dtINT32, &((op_lpnormalization_t *)op)->axis));
}

void op_cuda_lpnormalization_tp_destroy(op_t *op) { (void)op; }

void op_cuda_lpnormalization_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void reduce_l1norm(
    float *output, const float *input, size_t inner_num, size_t outer_num, size_t reduced_num)
{
    CUDA_KERNEL_LOOP(i, inner_num * outer_num)
    {
        size_t inner = i % inner_num;
        size_t outer = i / inner_num;
        const float *tmp = &input[outer * reduced_num * inner_num + inner];
        float sum = 0.;
        for (int k = 0; k < reduced_num; k++) {
            sum += abs(tmp[k * inner_num]);
        }
        float *out = &output[outer * reduced_num * inner_num + inner];
        for (int k = 0; k < reduced_num; k++) {
            out[k * inner_num] = tmp[k * inner_num] / sum;
        }
    }
}

__global__ void reduce_l2norm(
    float *output, const float *input, size_t inner_num, size_t outer_num, size_t reduced_num)
{
    CUDA_KERNEL_LOOP(i, inner_num * outer_num)
    {
        size_t inner = i % inner_num;
        size_t outer = i / inner_num;
        const float *tmp = &input[outer * reduced_num * inner_num + inner];
        float sum = 0.;
        for (int k = 0; k < reduced_num; k++) {
            sum += tmp[k * inner_num] * tmp[k * inner_num];
        }
        sum = sqrt(sum);
        float *out = &output[outer * reduced_num * inner_num + inner];
        for (int k = 0; k < reduced_num; k++) {
            out[k * inner_num] = tmp[k * inner_num] / sum;
        }
    }
}

static void op_cuda_l1norm_run(op_t *op)
{
    const op_lpnormalization_t *lpnormalization_op = (op_lpnormalization_t *)op;
    shape_t *shape = &op->input_tensors[0]->shape;
    int32_t axis = lpnormalization_op->axis;
    axis = axis > 0 ? axis : axis + shape->dim_size;
    size_t reduced_num = shape->dim[axis];
    size_t inner_num = shape_part_count(shape, axis) / reduced_num;
    size_t outer_num = shape_count(shape) / reduced_num / inner_num;

    const float *input = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    reduce_l1norm<<<
        (inner_num * outer_num + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output, input, inner_num, outer_num, reduced_num);
}

static void op_cuda_l2norm_run(op_t *op)
{
    const op_lpnormalization_t *lpnormalization_op = (op_lpnormalization_t *)op;
    shape_t *shape = &op->input_tensors[0]->shape;
    int32_t axis = lpnormalization_op->axis;
    axis = axis > 0 ? axis : axis + shape->dim_size;
    size_t reduced_num = shape->dim[axis];
    size_t inner_num = shape_part_count(shape, axis) / reduced_num;
    size_t outer_num = shape_count(shape) / reduced_num / inner_num;

    const float *input = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    reduce_l2norm<<<
        (inner_num * outer_num + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output, input, inner_num, outer_num, reduced_num);
}

void op_cuda_lpnormalization_tp_prepare(op_t *op)
{
    op_lpnormalization_t *lpnormalization_op = (op_lpnormalization_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (1 == lpnormalization_op->p) {
            op->run_func = op_cuda_l1norm_run;
        } else if (2 == lpnormalization_op->p) {
            op->run_func = op_cuda_l2norm_run;
        } else {
            CHECK(false);
        }
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
