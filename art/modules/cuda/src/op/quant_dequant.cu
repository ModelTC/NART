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
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    int *q_min;
    int *q_max;
    float *scale;
    int *cuda_q_max;
    int *cuda_q_min;
    float *cuda_scale;
} op_quant_dequant_t;

op_quant_dequant_t *op_cuda_quant_dequant_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_quant_dequant_t *res = (op_quant_dequant_t *)malloc(sizeof(op_quant_dequant_t));
    memset(res, 0, sizeof(op_quant_dequant_t));
    return res;
}

void op_cuda_quant_dequant_tp_config(op_t *op)
{
    size_t len_q_min;
    size_t len_q_max;
    size_t len_scale;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_QMIN, dtINT32, &len_q_min, &((op_quant_dequant_t *)op)->q_min));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_QMAX, dtINT32, &len_q_max, &((op_quant_dequant_t *)op)->q_max));
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_SCALE, dtFLOAT32, &len_scale,
        &((op_quant_dequant_t *)op)->scale));
}

void op_cuda_quant_dequant_tp_destroy(op_t *op) { (void)op; }

void op_cuda_quant_dequant_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
    cudaFree(((op_quant_dequant_t *)op)->cuda_q_max);
    cudaFree(((op_quant_dequant_t *)op)->cuda_q_min);
    cudaFree(((op_quant_dequant_t *)op)->cuda_scale);
}

__global__ void op_cuda_quant_dequant_kernel(
    size_t features, int *q_max, int *q_min, float *scale, float *output, const float *input,
    size_t size)
{
    CUDA_KERNEL_LOOP(i, size)
    {
        size_t c = i / features;
        output[i] = (int)round(input[i] / scale[c]);
        output[i] = output[i] <= q_max[c] ? output[i] : q_max[c];
        output[i] = output[i] >= q_min[c] ? output[i] : q_min[c];
        output[i] *= scale[c];
    }
}

static void op_cuda_quant_dequant_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    op_quant_dequant_t *quant_dequant_op = (op_quant_dequant_t *)op;
    const float *input = (const float *)mem_data(op->input_tensors[1]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    size_t num_c;
    size_t features;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_SCALE, dtFLOAT32, &num_c, &quant_dequant_op->scale));
    features = count / num_c;

    op_cuda_quant_dequant_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        features, quant_dequant_op->cuda_q_max, quant_dequant_op->cuda_q_min,
        quant_dequant_op->cuda_scale, output, input, count);
}

void op_cuda_quant_dequant_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[1]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_quant_dequant_run;
        break;
    default:
        CHECK(false);
        break;
    }
    size_t num_c;
    op_quant_dequant_t *quant_dequant_op = (op_quant_dequant_t *)op;
    CHECK(op_setting_array_get(
        op, SETTING_QUANT_DEQUANT_SCALE, dtFLOAT32, &num_c, &quant_dequant_op->scale));
    cudaMalloc((void **)&(quant_dequant_op->cuda_q_min), num_c * sizeof(int));
    cudaMalloc((void **)&(quant_dequant_op->cuda_q_max), num_c * sizeof(int));
    cudaMalloc((void **)&(quant_dequant_op->cuda_scale), num_c * sizeof(float));
    cudaMemcpy(
        quant_dequant_op->cuda_q_max, quant_dequant_op->q_max, num_c * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        quant_dequant_op->cuda_q_min, quant_dequant_op->q_min, num_c * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        quant_dequant_op->cuda_scale, quant_dequant_op->scale, num_c * sizeof(float),
        cudaMemcpyHostToDevice);
}

#ifdef __cplusplus
}
#endif
