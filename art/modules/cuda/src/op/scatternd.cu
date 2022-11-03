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

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { NONE = 0, MUL = 1, ADD = 2 } reduction_type;

typedef struct {
    op_t o;
    mem_t *strides;
    reduction_type reduction;
} op_scatternd_t;

op_scatternd_t *op_cuda_scatternd_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_scatternd_t *res = (op_scatternd_t *)malloc(sizeof(op_scatternd_t));
    memset(res, 0, sizeof(op_scatternd_t));
    res->strides = mem_new(cuda_mem_tp);
    return res;
}

void op_cuda_scatternd_tp_config(op_t *op)
{
    char *reduction;
    CHECK(op_setting_single_get(op, SETTING_SCATTERND_REDUCTION, dtSTR, &reduction));
    if (!strcmp(reduction, "")) {
        ((op_scatternd_t *)op)->reduction = NONE;
    } else if (!strcmp(reduction, "add")) {
        ((op_scatternd_t *)op)->reduction = ADD;
    } else if (!strcmp(reduction, "mul")) {
        ((op_scatternd_t *)op)->reduction = MUL;
    }
}

void op_cuda_scatternd_tp_destroy(op_t *op) { (void)op; }

void op_cuda_scatternd_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_scatternd_kernel(
    float *output, const float *indices, const float *updates, const size_t *strides,
    size_t dim_size, size_t count)
{
    CUDA_KERNEL_LOOP(i, count)
    {
        size_t offset = 0;
        for (int j = dim_size - 1; j >= 0; j--) {
            offset += strides[j] * indices[i * dim_size + j];
        }
        output[offset] = updates[i];
    }
}

static void op_cuda_scatternd_run(op_t *op)
{
    const float *data = (const float *)mem_data(op->input_tensors[0]->mem);
    const float *indices = (const float *)mem_data(op->input_tensors[1]->mem);
    const float *updates = (const float *)mem_data(op->input_tensors[2]->mem);
    const size_t *strides = (const size_t *)mem_data(((op_scatternd_t *)op)->strides);
    float *output = (float *)mem_data(op->output_tensors[0].mem);
    size_t count = shape_count(&op->input_tensors[0]->shape);
    size_t indices_count = shape_count(&op->input_tensors[1]->shape);
    size_t dim_size = op->input_tensors[0]->shape.dim_size;

    size_t update_count = indices_count / dim_size;
    cudaMemcpyAsync(
        output, data, sizeof(float) * count, cudaMemcpyDeviceToDevice,
        CUDA_WORKSPACE_STREAM(op->workspace));
    op_cuda_scatternd_kernel<<<
        (update_count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output, indices, updates, strides, dim_size, update_count);
}

void op_cuda_scatternd_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    // prepare strides
    size_t dim_size = op->input_tensors[0]->shape.dim_size;
    mem_alloc(((op_scatternd_t *)op)->strides, sizeof(size_t) * dim_size);
    size_t *strides = (size_t *)mem_cpu_data(((op_scatternd_t *)op)->strides);
    strides[dim_size - 1] = 1;
    for (int j = dim_size - 2; j > 0; j--) {
        strides[j] = strides[j + 1] * op->input_tensors[0]->shape.dim[j + 1];
    }

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_scatternd_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
