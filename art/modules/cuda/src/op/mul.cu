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
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    mem_t *shape_dim;
    mem_t *shape_dim1;
    mem_t *shape_dim2;
} op_mul_t;

op_mul_t *op_cuda_mul_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_mul_t *res = (op_mul_t *)malloc(sizeof(op_mul_t));
    memset(res, 0, sizeof(op_mul_t));
    return res;
}

void op_cuda_mul_tp_config(op_t *op) { (void)op; }

void op_cuda_mul_tp_destroy(op_t *op)
{
    if (((op_mul_t *)op)->shape_dim != NULL) {
        mem_delete(((op_mul_t *)op)->shape_dim);
        ((op_mul_t *)op)->shape_dim = NULL;
    }
    if (((op_mul_t *)op)->shape_dim1 != NULL) {
        mem_delete(((op_mul_t *)op)->shape_dim1);
        ((op_mul_t *)op)->shape_dim1 = NULL;
    }
    if (((op_mul_t *)op)->shape_dim2 != NULL) {
        mem_delete(((op_mul_t *)op)->shape_dim2);
        ((op_mul_t *)op)->shape_dim2 = NULL;
    }
    (void)op;
}

void op_cuda_mul_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_mul_kernel(float *c, const float *a, const float *b, size_t size)
{
    CUDA_KERNEL_LOOP(i, size) { c[i] = a[i] * b[i]; }
}

__global__ void op_cuda_mul_broadcast_kernel(
    float *output_0, const float *input_0, const float *input_1, size_t size, uint32_t *shape_dim,
    uint32_t *shape_dim1, uint32_t *shape_dim2, uint32_t dim_size)
{
    CUDA_KERNEL_LOOP(i, size)
    {
        uint32_t p = 1, p1 = 1, p2 = 1;
        uint32_t id1 = 0, id2 = 0;
        int j;
        for (j = dim_size - 1; j >= 0; j--) {
            uint32_t tmp = (i / p) % (shape_dim[j]);
            id1 += (shape_dim1[j] == 1 ? 0 : tmp) * p1;
            id2 += (shape_dim2[j] == 1 ? 0 : tmp) * p2;
            p1 *= shape_dim1[j];
            p2 *= shape_dim2[j];
            p *= shape_dim[j];
        }
        output_0[i] = input_0[id1] * input_1[id2];
    }
}

static void op_cuda_mul_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    if (tensor_shape_eq(&op->input_tensors[0]->shape, &op->input_tensors[1]->shape)) {
        op_cuda_mul_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, input_1, count);
    } else {
        shape_t shape = op->output_tensors[0].shape;
        shape_t shape1 = op->input_tensors[0]->shape;
        shape_t shape2 = op->input_tensors[1]->shape;
        for (int i = 0; i < shape1.dim_size; ++i) {
            shape1.dim[shape.dim_size - 1 - i] = shape1.dim[shape1.dim_size - 1 - i];
        }
        for (int i = 0; i < shape.dim_size - shape1.dim_size; ++i) {
            shape1.dim[i] = 1;
        }
        for (int i = 0; i < shape2.dim_size; ++i) {
            shape2.dim[shape.dim_size - 1 - i] = shape2.dim[shape2.dim_size - 1 - i];
        }
        for (int i = 0; i < shape.dim_size - shape2.dim_size; ++i) {
            shape2.dim[i] = 1;
        }
        uint32_t *shape_dim = (uint32_t *)mem_data(((op_mul_t *)op)->shape_dim);
        CUDA_CHECK(cudaMemcpy(
            shape_dim, shape.dim, sizeof(uint32_t) * shape.dim_size, cudaMemcpyHostToDevice));
        uint32_t *shape_dim1 = (uint32_t *)mem_data(((op_mul_t *)op)->shape_dim1);
        CUDA_CHECK(cudaMemcpy(
            shape_dim1, shape1.dim, sizeof(uint32_t) * shape.dim_size, cudaMemcpyHostToDevice));
        uint32_t *shape_dim2 = (uint32_t *)mem_data(((op_mul_t *)op)->shape_dim2);
        CUDA_CHECK(cudaMemcpy(
            shape_dim2, shape2.dim, sizeof(uint32_t) * shape.dim_size, cudaMemcpyHostToDevice));
        op_cuda_mul_broadcast_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, input_1, count, shape_dim, shape_dim1, shape_dim2, shape.dim_size);
    }
}

void op_cuda_mul_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    size_t count = op->output_tensors[0].shape.dim_size;
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (((op_mul_t *)op)->shape_dim == NULL) {
            ((op_mul_t *)op)->shape_dim = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_mul_t *)op)->shape_dim, count * sizeof(uint32_t));
        if (((op_mul_t *)op)->shape_dim1 == NULL) {
            ((op_mul_t *)op)->shape_dim1 = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_mul_t *)op)->shape_dim1, count * sizeof(uint32_t));
        if (((op_mul_t *)op)->shape_dim2 == NULL) {
            ((op_mul_t *)op)->shape_dim2 = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_mul_t *)op)->shape_dim2, count * sizeof(uint32_t));
        op->run_func = op_cuda_mul_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
