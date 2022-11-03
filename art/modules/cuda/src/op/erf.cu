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

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

typedef struct {
    op_t o;
} op_erf_t;

#ifdef __cplusplus
extern "C" {

op_erf_t *op_cuda_erf_tp_alloc(workspace_t *ws);
void op_cuda_erf_tp_config(op_t *op);
void op_cuda_erf_tp_destroy(op_t *op);
void op_cuda_erf_tp_dealloc(op_t *op);
void op_cuda_erf_tp_prepare(op_t *op);
}
#endif

op_erf_t *op_cuda_erf_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_erf_t *res = (op_erf_t *)malloc(sizeof(op_erf_t));
    memset(res, 0, sizeof(op_erf_t));
    return res;
}

void op_cuda_erf_tp_config(op_t *op) { (void)op; }

void op_cuda_erf_tp_destroy(op_t *op) { (void)op; }

void op_cuda_erf_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_erf_kernel(size_t n, float *out, const float *in)
{
    CUDA_KERNEL_LOOP(i, n) { out[i] = erff(in[i]); }
}

static void op_cuda_erf_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const float *input_0 = reinterpret_cast<float *>(mem_data(op->input_tensors[0]->mem));
    float *output_0 = reinterpret_cast<float *>(mem_data(op->output_tensors[0].mem));
    size_t block_size = 1024;
    // maximum 65536 blocks.
    size_t block_num = (count + block_size - 1) / block_size;
    op_cuda_erf_kernel<<<block_num, block_size, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        count, output_0, input_0);
}

void op_cuda_erf_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_erf_run;
        break;

    default:
        CHECK(false);
        break;
    }
}
