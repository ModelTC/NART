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

#include <cuda_runtime.h>

#include "art/cuda/cuda_mem.h"
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
    uint32_t operation;
    float *coeff;
} op_eltwise_t;

op_eltwise_t *op_cuda_eltwise_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_eltwise_t *res = (op_eltwise_t *)malloc(sizeof(op_eltwise_t));
    memset(res, 0, sizeof(op_eltwise_t));
    return res;
}

void op_cuda_eltwise_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_ELTWISE_OPERATION, dtUINT32, &((op_eltwise_t *)op)->operation));
    if (op_setting_if_set(op, SETTING_ELTWISE_COEFF)) {
        size_t len;
        CHECK(op_setting_array_get(
            op, SETTING_ELTWISE_COEFF, dtFLOAT32, &len, &((op_eltwise_t *)op)->coeff));
        CHECK_EQ(op->input_size, len);
    }
}

void op_cuda_eltwise_tp_destroy(op_t *op) { (void)op; }

void op_cuda_eltwise_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void
op_cuda_eltwise_prod_kernel(size_t n, float *out, const float *in0, const float *in1)
{
    CUDA_KERNEL_LOOP(i, n) { out[i] = in0[i] * in1[i]; }
}

static void op_cuda_eltwise_accumulate_output_prod(op_t *op)
{
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    op_cuda_eltwise_prod_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        count, output, (float *)mem_data(op->input_tensors[0]->mem),
        (float *)mem_data(op->input_tensors[1]->mem));
}

__global__ void op_cuda_eltwise_sum_kernel(
    size_t n, float *out, const float *in0, float coeff0, const float *in1, float coeff1)
{
    CUDA_KERNEL_LOOP(i, n) { out[i] = coeff0 * in0[i] + coeff1 * in1[i]; }
}

static void op_cuda_eltwise_accumulate_output_sum(op_t *op)
{
    uint32_t i;
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    const float *coeff = ((op_eltwise_t *)op)->coeff;
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    const float **input = (const float **)malloc(op->input_size * sizeof(float *));
    CHECK(op->input_size == 2);
    for (i = 0; i < op->input_size; ++i) {
        input[i] = (float *)mem_data(op->input_tensors[i]->mem);
    }
    op_cuda_eltwise_sum_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        count, output, input[0], coeff[0], input[1], coeff[1]);
}

__global__ void op_cuda_eltwise_max_kernel(size_t n, float *out, const float *in0, const float *in1)
{
    CUDA_KERNEL_LOOP(i, n) { out[i] = fmaxf(in0[i], in1[i]); }
}

static void op_cuda_eltwise_accumulate_output_max(op_t *op)
{
    uint32_t count = shape_count(&op->output_tensors[0].shape);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    op_cuda_eltwise_max_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        count, output, (float *)mem_data(op->input_tensors[0]->mem),
        (float *)mem_data(op->input_tensors[1]->mem));
}

void op_cuda_eltwise_tp_prepare(op_t *op)
{
    CHECK(
        ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_PROD
        || ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_SUM
        || ((op_eltwise_t *)op)->operation == SETTING_ELTWISE_OP_MAX);

    int i;
    CHECK(op->input_size == 2);
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
    switch (((op_eltwise_t *)op)->operation) {
    case SETTING_ELTWISE_OP_PROD:
        op->run_func = op_cuda_eltwise_accumulate_output_prod;
        break;
    case SETTING_ELTWISE_OP_SUM:
        op->run_func = op_cuda_eltwise_accumulate_output_sum;
        break;
    case SETTING_ELTWISE_OP_MAX:
        op->run_func = op_cuda_eltwise_accumulate_output_max;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
