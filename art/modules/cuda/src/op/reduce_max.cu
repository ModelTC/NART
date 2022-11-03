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

#include <float.h>
#include <math.h>
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
    bool keepdims;
    int32_t *axes;
    mem_t *reduce;
} op_reducemax_t;

op_reducemax_t *op_cuda_reducemax_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_reducemax_t *res = (op_reducemax_t *)malloc(sizeof(op_reducemax_t));
    memset(res, 0, sizeof(op_reducemax_t));
    return res;
}

void op_cuda_reducemax_tp_config(op_t *op)
{
    size_t len;
    CHECK(op_setting_array_get(
        op, SETTING_REDUCE_AXES, dtINT32, &len, &((op_reducemax_t *)op)->axes));
    (void)op;
}

void op_cuda_reducemax_tp_destroy(op_t *op)
{
    if (((op_reducemax_t *)op)->reduce != NULL) {
        mem_delete(((op_reducemax_t *)op)->reduce);
        ((op_reducemax_t *)op)->reduce = NULL;
    }
    (void)op;
}

void op_cuda_reducemax_tp_dealloc(op_t *op)
{
    if (NULL != op) {
        free(op);
    }
}

__global__ void
op_cuda_reducemax_kernel(float *output_0, float *input_0, size_t shape, size_t lowDim)
{
    // CUDA_KERNEL_LOOP(i, size) {
    //     output_0[i] = fmax(output_0[i] , input_0[i]);
    // }
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int highId = bid / lowDim;
    unsigned int lowId = bid % lowDim;
    float *bInput = input_0 + highId * shape * lowDim + lowId;
    float *output = output_0 + highId * lowDim + lowId;
    extern __shared__ float cache[];
    cache[tid] = bInput[lowDim * tid];
    __syncthreads();
    if (shape > 1024) {
        int cycle = (shape + 1023) / 1024;
        for (int i = 1; i < cycle; i++) {
            int calShape = ((i == cycle - 1) && (shape % 1024)) ? (shape % 1024) : 1024;
            if (tid < calShape) {
                cache[tid] = fmax(cache[tid], bInput[lowDim * tid + i * 1024 * lowDim]);
            }
        }
    }
    __syncthreads();
    int count = blockDim.x;
    for (int stride = (blockDim.x + 1) >> 1; stride > 1; stride = (stride + 1) >> 1) {
        if ((tid + stride) < count) {
            cache[tid] = fmax(cache[tid], cache[stride + tid]);
        }
        count = stride;
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = fmax(cache[0], cache[1]);
    }
}

static void op_cuda_reducemax_run(op_t *op)
{
    size_t count = shape_count(&op->input_tensors[0]->shape);
    shape_t shape_i = op->input_tensors[0]->shape;
    size_t count_o = shape_count(&op->output_tensors[0].shape);
    int i;
    const float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    size_t len;
    int *axes;
    CHECK(op_setting_array_get(op, SETTING_REDUCE_AXES, dtINT32, &len, &axes));
    // TODO: when reduce->(1) bank conflict?
    //  if(len == 0){
    //      output_0[0] = FLT_MAX;
    //      for (i = 0; i < count; ++i) {
    //          output_0[0] = fmin(output_0[0],input_0[i]);
    //      }
    //      return;
    //  }
    int32_t flag[MAX_DIM] = { 0 };
    size_t lowDim = count;
    size_t highDim = 1;
    for (i = 0; i < (int)len; ++i) {
        if (axes[i] < 0)
            axes[i] = shape_i.dim_size + axes[i];
    }
    for (i = 0; i < (int)len; ++i) {
        flag[axes[i]] = 1;
    }
    CUDA_CHECK(cudaMemsetAsync(
        output_0, 0, sizeof(float) * count_o, CUDA_WORKSPACE_STREAM(op->workspace)));
    float *reduce = (float *)mem_data(((op_reducemax_t *)op)->reduce);
    CUDA_CHECK(cudaMemcpyAsync(
        reduce, input_0, sizeof(float) * count, cudaMemcpyDeviceToDevice,
        CUDA_WORKSPACE_STREAM(op->workspace)));
    for (i = 0; i < (int)shape_i.dim_size; ++i) {
        lowDim /= shape_i.dim[i];
        if (flag[i]) {
            CHECK(shape_i.dim[i] != 1);
            float *in = reduce;
            float *out = reduce + count;
            int blockSize = (shape_i.dim[i] >= 1024) ? 1024 : (shape_i.dim[i]);
            op_cuda_reducemax_kernel<<<
                highDim * lowDim, blockSize, sizeof(float) * min(1024, shape_i.dim[i]),
                CUDA_WORKSPACE_STREAM(op->workspace)>>>(out, in, shape_i.dim[i], lowDim);
            CUDA_CHECK(cudaMemcpyAsync(
                in, out, sizeof(float) * count, cudaMemcpyDeviceToDevice,
                CUDA_WORKSPACE_STREAM(op->workspace)));
        } else {
            highDim *= shape_i.dim[i];
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(
        output_0, reduce, sizeof(float) * count_o, cudaMemcpyDeviceToDevice,
        CUDA_WORKSPACE_STREAM(op->workspace)));
}
void op_cuda_reducemax_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    size_t count = shape_count(&op->input_tensors[0]->shape);
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (((op_reducemax_t *)op)->reduce == NULL) {
            ((op_reducemax_t *)op)->reduce = mem_new(cuda_mem_tp);
        }
        mem_alloc(((op_reducemax_t *)op)->reduce, count * sizeof(float) * 2);
        op->run_func = op_cuda_reducemax_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
