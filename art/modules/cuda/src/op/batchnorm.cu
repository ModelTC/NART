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
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "cuda_runtime.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ThreadsPerBlock 1024

typedef struct {
    op_t o;
    float eps;
    mem_t *mean;
    mem_t *variance;
    mem_t *reduce;
} op_batchnorm_t;

op_batchnorm_t *op_cuda_batchnorm_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_batchnorm_t *res = (op_batchnorm_t *)malloc(sizeof(op_batchnorm_t));
    memset(res, 0, sizeof(op_batchnorm_t));
    res->mean = NULL;
    res->variance = NULL;
    res->reduce = NULL;
    return res;
}

void op_cuda_batchnorm_tp_config(op_t *op)
{
    CHECK(
        op_setting_single_get(op, SETTING_BATCHNORM_EPS, dtFLOAT32, &((op_batchnorm_t *)op)->eps));
}

void op_cuda_batchnorm_tp_destroy(op_t *op)
{
    op_batchnorm_t *batchnorm_op = (op_batchnorm_t *)op;
    if (batchnorm_op->mean != NULL) {
        mem_delete(batchnorm_op->mean);
        batchnorm_op->mean = NULL;
    }

    if (batchnorm_op->variance != NULL) {
        mem_delete(batchnorm_op->variance);
        batchnorm_op->variance = NULL;
    }
    if (batchnorm_op->reduce != NULL) {
        mem_delete(batchnorm_op->reduce);
        batchnorm_op->reduce = NULL;
    }
}

void op_cuda_batchnorm_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static int32_t caculate_2_n_less(int hw)
{
    int32_t hw_less_near_2_n = 0;
    for (int i = 0; i < 30; i++) {
        if (hw >= (1 << i))
            hw_less_near_2_n = (1 << i);
    }
    return hw_less_near_2_n;
}

__global__ void
bn_sum_mean_2(float *output, const float *input, size_t bs, size_t ch, size_t hw, int hw_near)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= bs * ch * hw)
        return;
    int tid = threadIdx.x;
    int nci = global_idx / hw;
    int hwi = global_idx % hw;
    __shared__ float reduce[ThreadsPerBlock];
    reduce[tid] = input[global_idx];
    __syncthreads();
    if (hwi + 1024 >= hw) {
        atomicAdd(output + (nci % ch) * hw + hwi / 1024, reduce[tid]);
        return;
    }
    for (int i = ThreadsPerBlock / 2; i > 0; i /= 2) {
        if (tid < i) {
            reduce[tid] += reduce[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output + (nci % ch) * hw + hwi / 1024, reduce[0]);
    }
}

__global__ void bn_get_mean_2(float *input, size_t ch, size_t NHW)
{
    CUDA_KERNEL_LOOP(i, ch) { input[i] /= NHW; }
}

__global__ void bn_sum_variance_2(
    float *output, const float *input, float *mean, size_t bs, size_t ch, size_t hw, int hw_near)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= bs * ch * hw)
        return;
    int tid = threadIdx.x;
    int nci = global_idx / hw;
    int hwi = global_idx % hw;
    __shared__ float reduce[ThreadsPerBlock];
    reduce[tid] = pow(input[global_idx] - mean[nci], 2);
    __syncthreads();
    if (hwi + 1024 >= hw) {
        atomicAdd(output + (nci % ch) * hw + hwi / 1024, reduce[tid]);
        return;
    }
    for (int i = ThreadsPerBlock / 2; i > 0; i /= 2) {
        if (tid < i) {
            reduce[tid] += reduce[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output + (nci % ch) * hw + hwi / 1024, reduce[0]);
    }
}

__global__ void bn_get_variance_2(float *input, size_t ch, size_t NHW)
{
    CUDA_KERNEL_LOOP(i, ch) { input[i] /= NHW; }
}

__global__ void bn_reduece_2(float *output, float *input, size_t bs, size_t ch, size_t hw)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    atomicAdd(output + (bid % ch), input[(bid % ch) * hw + tid]);
}

__global__ void bn_update_output_2(
    float *output, const float *input_0, const float *mean, const float *variance, size_t bs,
    size_t ch, size_t hw, const float eps)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < bs * ch * hw) {
        int channel_id = (global_idx / hw) % ch;
        float std_ch = sqrt(variance[channel_id] + eps);
        float mean_ch = mean[channel_id];

        output[global_idx] = (input_0[global_idx] - mean_ch) / std_ch;
    }
}

static void op_cuda_batchnorm_run(op_t *op)
{
    const op_batchnorm_t *batchnorm_op = (op_batchnorm_t *)op;
    int bs = op->input_tensors[0]->shape.dim[0];
    int ch = op->input_tensors[0]->shape.dim[1];
    int hw = 1;
    shape_t const *shape = &(op->input_tensors[0]->shape);
    for (int idx = 2; idx < shape->dim_size; ++idx) {
        hw *= shape->dim[idx];
    }

    const float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);

    float *mean = (float *)mem_data(batchnorm_op->mean);
    float *variance = (float *)mem_data(batchnorm_op->variance);
    float *reduce = (float *)mem_data(batchnorm_op->reduce);

    CUDA_CHECK(cudaMemsetAsync(mean, 0, sizeof(float) * ch, CUDA_WORKSPACE_STREAM(op->workspace)));
    CUDA_CHECK(
        cudaMemsetAsync(variance, 0, sizeof(float) * ch, CUDA_WORKSPACE_STREAM(op->workspace)));

    int hw_less_near_2_n = caculate_2_n_less(hw);

    bn_sum_mean_2<<<
        (bs * ch * hw + 1024 - 1) / 1024, 1024, 1024 * sizeof(float),
        CUDA_WORKSPACE_STREAM(op->workspace)>>>(reduce, input_0, bs, ch, hw, hw_less_near_2_n);
    bn_reduece_2<<<bs * ch, (hw + 1023) / 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        mean, reduce, bs, ch, hw);
    bn_get_mean_2<<<(bs * ch + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        mean, bs * ch, hw);
    CUDA_CHECK(
        cudaMemsetAsync(reduce, 0, sizeof(float) * bs * hw, CUDA_WORKSPACE_STREAM(op->workspace)));
    bn_sum_variance_2<<<
        (bs * ch * hw + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        reduce, input_0, mean, bs, ch, hw, hw_less_near_2_n);
    bn_reduece_2<<<bs * ch, (hw + 1023) / 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        variance, reduce, bs, ch, hw);
    bn_get_variance_2<<<
        (bs * ch + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        variance, bs * ch, hw);
    bn_update_output_2<<<
        (bs * ch * hw + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output_0, input_0, mean, variance, bs, ch, hw, batchnorm_op->eps);
}

static void op_cuda_batchnorm_average_run(op_t *op)
{
    const op_batchnorm_t *batchnorm_op = (op_batchnorm_t *)op;
    int bs = op->input_tensors[0]->shape.dim[0];
    int ch = op->input_tensors[0]->shape.dim[1];
    const float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (float *)mem_data(op->input_tensors[1]->mem);
    const float *input_2 = (float *)mem_data(op->input_tensors[2]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    int hw = 1;
    shape_t const *shape = &(op->input_tensors[0]->shape);
    for (int idx = 2; idx < shape->dim_size; ++idx) {
        hw *= shape->dim[idx];
    }

    bn_update_output_2<<<
        (bs * ch * hw + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output_0, input_0, input_1, input_2, bs, ch, hw, batchnorm_op->eps);
}

void op_cuda_batchnorm_tp_prepare(op_t *op)
{
    op_batchnorm_t *batchnorm_op = (op_batchnorm_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (op->input_size == 1) {
            int bs = op->input_tensors[0]->shape.dim[0];
            int ch = op->input_tensors[0]->shape.dim[1];
            int hw = 1;
            size_t i;
            for (i = 2; i < op->input_tensors[0]->shape.dim_size; i++)
                hw *= op->input_tensors[0]->shape.dim[i];
            if (batchnorm_op->mean == NULL) {
                batchnorm_op->mean = mem_new(cuda_mem_tp);
            }
            mem_alloc(batchnorm_op->mean, ch * sizeof(float));
            if (batchnorm_op->variance == NULL) {
                batchnorm_op->variance = mem_new(cuda_mem_tp);
            }
            mem_alloc(batchnorm_op->variance, ch * sizeof(float));
            if (batchnorm_op->reduce == NULL) {
                batchnorm_op->reduce = mem_new(cuda_mem_tp);
            }
            mem_alloc(batchnorm_op->reduce, bs * hw * sizeof(float));
            CUDA_CHECK(cudaMemset(mem_data(batchnorm_op->mean), 0, sizeof(float) * ch));
            CUDA_CHECK(cudaMemset(mem_data(batchnorm_op->variance), 0, sizeof(float) * ch));
            CUDA_CHECK(cudaMemset(mem_data(batchnorm_op->reduce), 0, sizeof(float) * bs * hw));
            op->run_func = op_cuda_batchnorm_run;
        } else {
            op->run_func = op_cuda_batchnorm_average_run;
        }
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
