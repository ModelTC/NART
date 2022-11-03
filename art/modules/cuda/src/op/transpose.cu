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

typedef struct {
    op_t o;
    mem_t *from_counts;
    mem_t *to_counts;
    mem_t *map;
    mem_t *buf;
    int32_t *dims;
} op_transpose_t;

op_transpose_t *op_cuda_transpose_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_transpose_t *res = (op_transpose_t *)malloc(sizeof(op_transpose_t));
    memset(res, 0, sizeof(op_transpose_t));

    res->from_counts = mem_new(cuda_mem_tp);
    res->to_counts = mem_new(cuda_mem_tp);
    res->map = mem_new(cuda_mem_tp);
    res->buf = mem_new(cuda_mem_tp);
    return res;
}

void op_cuda_transpose_tp_config(op_t *op)
{
    size_t ndim;
    CHECK(op_setting_array_get(
        op, SETTING_TRANSPOSE_DIMS, dtINT32, &ndim, &((op_transpose_t *)op)->dims));
}

void op_cuda_transpose_tp_destroy(op_t *op) { (void)op; }

void op_cuda_transpose_tp_dealloc(op_t *op)
{
    op_transpose_t *trans_op = (op_transpose_t *)op;

    if (NULL != trans_op->from_counts) {
        mem_delete(trans_op->from_counts);
    }
    if (NULL != trans_op->to_counts) {
        mem_delete(trans_op->to_counts);
    }
    if (NULL != trans_op->map) {
        mem_delete(trans_op->map);
    }
    if (NULL != trans_op->buf) {
        mem_delete(trans_op->buf);
    }
    free(op);
}

__global__ void op_cuda_transpose_kernel(
    const int nthreads, const float *from_data, float *to_data, const int *from_counts,
    const int *to_counts, const int *map, const int num_axes, int *buf)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int *from_inds = buf + index * num_axes;

        int from_index = index, to_index = 0;
        for (int i = 0; i < num_axes; i++) {
            from_inds[i] = from_index / from_counts[i];
            from_index = from_index % from_counts[i];
        }
        for (int i = 0; i < num_axes; i++) {
            to_index += from_inds[map[i]] * to_counts[i];
        }

        *(to_data + to_index) = *(from_data + index);
    }
}

static void op_cuda_transpose_run(op_t *op)
{
    op_transpose_t *trans_op = (op_transpose_t *)op;
    size_t count = shape_count(&op->output_tensors[0].shape);
    const float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);

    size_t num_axes = op->input_tensors[0]->shape.dim_size;
    int *from_counts = (int *)mem_data(trans_op->from_counts);
    int *to_counts = (int *)mem_data(trans_op->to_counts);
    int *map = (int *)mem_data(trans_op->map);
    int *buf = (int *)mem_data(trans_op->buf);
    op_cuda_transpose_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        count, input_0, output_0, from_counts, to_counts, map, num_axes, buf);
}

void op_cuda_transpose_tp_prepare(op_t *op)
{
    op_transpose_t *trans_op = (op_transpose_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    // configure aux memories
    size_t num_axes = op->input_tensors[0]->shape.dim_size;
    mem_alloc(trans_op->from_counts, num_axes * sizeof(int));
    mem_alloc(trans_op->to_counts, num_axes * sizeof(int));
    mem_alloc(trans_op->map, num_axes * sizeof(int));
    mem_alloc(trans_op->buf, num_axes * shape_count(&op->input_tensors[0]->shape) * sizeof(int));

    int *from_counts = (int *)mem_cpu_data(trans_op->from_counts);
    int *to_counts = (int *)mem_cpu_data(trans_op->to_counts);
    int *map = (int *)mem_cpu_data(trans_op->map);

    size_t c1 = shape_count(&op->input_tensors[0]->shape);
    size_t c2 = shape_count(&op->output_tensors[0].shape);
    shape_t shape_i = op->input_tensors[0]->shape;
    shape_t shape_o = op->output_tensors[0].shape;
    for (size_t idx = 0; idx < num_axes; idx++) {
        c1 /= shape_i.dim[idx];
        c2 /= shape_o.dim[idx];
        *from_counts++ = c1;
        *to_counts++ = c2;
        map[idx] = trans_op->dims[idx];
    }

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_transpose_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
