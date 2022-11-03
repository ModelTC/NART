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
} op_clip_t;
op_clip_t *op_cuda_clip_tp_alloc(workspace_t *ws);
void op_cuda_clip_tp_config(op_t *op);
void op_cuda_clip_tp_prepare(op_t *op);
void op_cuda_clip_tp_destroy(op_t *op);
void op_cuda_clip_tp_dealloc(op_t *op);

#ifdef __cplusplus
} // extern "C"
#endif

op_clip_t *op_cuda_clip_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_clip_t *res = (op_clip_t *)malloc(sizeof(op_clip_t));
    memset(res, 0, sizeof(op_clip_t));
    return res;
}

void op_cuda_clip_tp_config(op_t *op) { (void)op; }

void op_cuda_clip_tp_destroy(op_t *op) { (void)op; }

void op_cuda_clip_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

template <typename T>
__global__ void op_cuda_clip_kernel(T *c, const T *a, const T *min, const T *max, size_t size)
{
    CUDA_KERNEL_LOOP(i, size)
    {
        if (a[i] < (*min)) {
            c[i] = (*min);
        } else if (a[i] > (*max)) {
            c[i] = (*max);
        } else {
            c[i] = a[i];
        }
    }
}
template <typename T> __global__ void op_cuda_clip_dummy_kernel(T *c, const T *a, size_t size)
{
    CUDA_KERNEL_LOOP(i, size) { c[i] = a[i]; }
}

template <typename T> static void op_cuda_clip_run(op_t *op)
{
    // op_clip_t *clip_op = (op_clip_t *)op;
    size_t count = shape_count(&op->output_tensors[0].shape);
    const T *input_0 = (const T *)mem_data(op->input_tensors[0]->mem);
    T *output_0 = (T *)mem_data(op->output_tensors[0].mem);
    if (op->input_size == 3) {
        const T *min = (const T *)mem_data(op->input_tensors[1]->mem);
        const T *max = (const T *)mem_data(op->input_tensors[2]->mem);
        op_cuda_clip_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, min, max, count);
    } else if (op->input_size == 1) {
        op_cuda_clip_dummy_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            output_0, input_0, count);
    } else {
        CHECK(false);
    }
}

void op_cuda_clip_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_clip_run<float>;
        break;
    case dtINT32:
        op->run_func = op_cuda_clip_run<int32_t>;
        break;
    default:
        CHECK(false);
        break;
    }
}
