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
} op_div_t;
op_div_t *op_cuda_div_tp_alloc(workspace_t *ws);
void op_cuda_div_tp_config(op_t *op);
void op_cuda_div_tp_destroy(op_t *op);
void op_cuda_div_tp_dealloc(op_t *op);
void op_cuda_div_tp_prepare(op_t *op);

#ifdef __cplusplus
}
#endif

op_div_t *op_cuda_div_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_div_t *res = (op_div_t *)malloc(sizeof(op_div_t));
    memset(res, 0, sizeof(op_div_t));
    return res;
}

void op_cuda_div_tp_config(op_t *op) { (void)op; }

void op_cuda_div_tp_destroy(op_t *op)
{
    if (((op_div_t *)op)->shape_dim != NULL) {
        mem_delete(((op_div_t *)op)->shape_dim);
        ((op_div_t *)op)->shape_dim = NULL;
    }
    if (((op_div_t *)op)->shape_dim1 != NULL) {
        mem_delete(((op_div_t *)op)->shape_dim1);
        ((op_div_t *)op)->shape_dim1 = NULL;
    }
    if (((op_div_t *)op)->shape_dim2 != NULL) {
        mem_delete(((op_div_t *)op)->shape_dim2);
        ((op_div_t *)op)->shape_dim2 = NULL;
    }
    (void)op;
}

void op_cuda_div_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

// Trunc Div
template <typename T> __device__ void div(const T *D, const T *d, T *q) { (*q) = (*D) / (*d); }

// Floor Div
template <> __device__ void div<int32_t>(const int32_t *D, const int32_t *d, int32_t *q)
{
    *q = (*D) / (*d);
    int32_t r = (*D) % (*d);
    if ((r > 0 && (*d) < 0) || (r < 0 && (*d) > 0))
        (*q) -= 1;
}

template <typename T> __global__ void op_cuda_div_kernel(T *c, const T *a, const T *b, size_t size)
{
    CUDA_KERNEL_LOOP(i, size) { div(a + i, b + i, c + i); }
}

template <typename T>
__global__ void op_cuda_div_broadcast_kernel(
    T *output_0, const T *input_0, const T *input_1, size_t size, uint32_t *shape_dim,
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
        div(input_0 + id1, input_1 + id2, output_0 + i);
    }
}

template <typename T> static void op_cuda_div_run(op_t *op)
{
    size_t count = shape_count(&op->output_tensors[0].shape);
    const T *input_0 = (const T *)mem_data(op->input_tensors[0]->mem);
    const T *input_1 = (const T *)mem_data(op->input_tensors[1]->mem);
    T *output_0 = (T *)mem_data(op->output_tensors[0].mem);
    if (tensor_shape_eq(&op->input_tensors[0]->shape, &op->input_tensors[1]->shape)) {
        op_cuda_div_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, input_1, count);
    } else {
        shape_t shape = op->output_tensors[0].shape;
        shape_t shape1 = op->input_tensors[0]->shape;
        shape_t shape2 = op->input_tensors[1]->shape;
        // broadcasting
        // i means the i-th item in the reverse order
        // copy the front dim to end,then pad 1 in the front
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

        uint32_t *pinned_shapes;
        CUDA_CHECK(cudaHostAlloc(
            (void **)&pinned_shapes, sizeof(uint32_t) * shape.dim_size * 3, cudaHostAllocDefault));
        memcpy(pinned_shapes, shape.dim, sizeof(uint32_t) * shape.dim_size);
        memcpy(pinned_shapes + shape.dim_size, shape1.dim, sizeof(uint32_t) * shape.dim_size);
        memcpy(pinned_shapes + shape.dim_size * 2, shape2.dim, sizeof(uint32_t) * shape.dim_size);

        uint32_t *shape_dim = (uint32_t *)mem_data(((op_div_t *)op)->shape_dim);
        CUDA_CHECK(cudaMemcpyAsync(
            shape_dim, pinned_shapes, sizeof(uint32_t) * shape.dim_size, cudaMemcpyHostToDevice,
            CUDA_WORKSPACE_STREAM(op->workspace)));
        uint32_t *shape_dim1 = (uint32_t *)mem_data(((op_div_t *)op)->shape_dim1);
        CUDA_CHECK(cudaMemcpyAsync(
            shape_dim1, pinned_shapes + shape.dim_size, sizeof(uint32_t) * shape.dim_size,
            cudaMemcpyHostToDevice, CUDA_WORKSPACE_STREAM(op->workspace)));
        uint32_t *shape_dim2 = (uint32_t *)mem_data(((op_div_t *)op)->shape_dim2);
        CUDA_CHECK(cudaMemcpyAsync(
            shape_dim2, pinned_shapes + shape.dim_size * 2, sizeof(uint32_t) * shape.dim_size,
            cudaMemcpyHostToDevice, CUDA_WORKSPACE_STREAM(op->workspace)));
        CUDA_CHECK(cudaFreeHost((void *)pinned_shapes));
        op_cuda_div_broadcast_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, input_1, count, shape_dim, shape_dim1, shape_dim2, shape.dim_size);
    }
}

void op_cuda_div_tp_prepare(op_t *op)
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
        if (((op_div_t *)op)->shape_dim == NULL) {
            ((op_div_t *)op)->shape_dim = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_div_t *)op)->shape_dim, count * sizeof(uint32_t));
        if (((op_div_t *)op)->shape_dim1 == NULL) {
            ((op_div_t *)op)->shape_dim1 = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_div_t *)op)->shape_dim1, count * sizeof(uint32_t));
        if (((op_div_t *)op)->shape_dim2 == NULL) {
            ((op_div_t *)op)->shape_dim2 = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_div_t *)op)->shape_dim2, count * sizeof(uint32_t));
        op->run_func = op_cuda_div_run<float>;
        break;
    case dtINT32:
        if (((op_div_t *)op)->shape_dim == NULL) {
            ((op_div_t *)op)->shape_dim = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_div_t *)op)->shape_dim, count * sizeof(uint32_t));
        if (((op_div_t *)op)->shape_dim1 == NULL) {
            ((op_div_t *)op)->shape_dim1 = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_div_t *)op)->shape_dim1, count * sizeof(uint32_t));
        if (((op_div_t *)op)->shape_dim2 == NULL) {
            ((op_div_t *)op)->shape_dim2 = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_div_t *)op)->shape_dim2, count * sizeof(uint32_t));
        op->run_func = op_cuda_div_run<int32_t>;
        break;
    default:
        CHECK(false);
        break;
    }
}
