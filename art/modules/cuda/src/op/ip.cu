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
#include <cudnn.h>

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
    uint32_t num_output;
    int32_t axis;
} op_ip_t;

op_ip_t *op_cuda_ip_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_ip_t *res = (op_ip_t *)malloc(sizeof(op_ip_t));
    memset(res, 0, sizeof(op_ip_t));
    return res;
}

void op_cuda_ip_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_IP_NUM_OUTPUT, dtUINT32, &((op_ip_t *)op)->num_output));
    CHECK(op_setting_single_get(op, SETTING_IP_AXIS, dtINT32, &((op_ip_t *)op)->axis));
}

void op_cuda_ip_tp_destroy(op_t *op) { (void)op; }

void op_cuda_ip_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void apply_bias(size_t n, const float *bias, size_t plane_size, float *top)
{
    CUDA_KERNEL_LOOP(i, n) { top[i] += bias[i % plane_size]; }
}

static void op_cuda_ip_run(op_t *op)
{
    const op_ip_t *ip_op = (op_ip_t *)op;
    const float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (float *)mem_data(op->input_tensors[1]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    // size_t M = op->input_tensors[0]->shape.dim[0];

    size_t N = ip_op->num_output;
    // size_t K = shape_count(&op->input_tensors[0]->shape) / M;

    float alpha = 1.0f;
    float beta = 0.0f;
    int axis, i;
    size_t i_preceding_axes = 1;
    size_t i_succeeding_axes = 1;
    if (ip_op->axis < 0) {
        axis = ip_op->axis + op->input_tensors[0]->shape.dim_size;
    } else {
        axis = ip_op->axis;
    }
    for (i = 0; i < axis; ++i) {
        i_preceding_axes *= op->input_tensors[0]->shape.dim[i];
    }
    for (i = axis; i < op->input_tensors[0]->shape.dim_size; ++i) {
        i_succeeding_axes *= op->input_tensors[0]->shape.dim[i];
    }
    size_t M = i_preceding_axes;
    size_t K = i_succeeding_axes;
    CUBLAS_CHECK(cublasSgemm(
        CUDA_WORKSPACE_CUBLASHDL(op->workspace), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, input_1,
        K, input_0, K, &beta, output, N));
    if (op->input_size > 2) {
        apply_bias<<<(M * N + 1024) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            M * N, (const float *)mem_data(op->input_tensors[2]->mem), N,
            (float *)mem_data(op->output_tensors[0].mem));
    }
}

void op_cuda_ip_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_ip_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
