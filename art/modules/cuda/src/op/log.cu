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
} op_log_t;

op_log_t *op_cuda_log_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_log_t *res = (op_log_t *)malloc(sizeof(op_log_t));
    memset(res, 0, sizeof(op_log_t));
    return res;
}

void op_cuda_log_tp_config(op_t *op) { (void)op; }

void op_cuda_log_tp_destroy(op_t *op) { (void)op; }

void op_cuda_log_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_log_kernel(float *c, const float *a, size_t size)
{
    CUDA_KERNEL_LOOP(i, size) { c[i] = logf(a[i]); }
}

static void op_cuda_log_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    op_cuda_log_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output_0, input_0, count);
}

void op_cuda_log_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_log_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
