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
} op_instancenorm_t;

op_instancenorm_t *op_cuda_instancenorm_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_instancenorm_t *res = (op_instancenorm_t *)malloc(sizeof(op_instancenorm_t));
    memset(res, 0, sizeof(op_instancenorm_t));
    res->mean = NULL;
    res->variance = NULL;
    return res;
}

void op_cuda_instancenorm_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_INSTANCENORM_EPS, dtFLOAT32, &((op_instancenorm_t *)op)->eps));
}

void op_cuda_instancenorm_tp_destroy(op_t *op)
{
    op_instancenorm_t *instancenorm_op = (op_instancenorm_t *)op;
    if (instancenorm_op->mean != NULL) {
        mem_delete(instancenorm_op->mean);
        instancenorm_op->mean = NULL;
    }

    if (instancenorm_op->variance != NULL) {
        mem_delete(instancenorm_op->variance);
        instancenorm_op->variance = NULL;
    }

    if (instancenorm_op->reduce != NULL) {
        mem_delete(instancenorm_op->reduce);
        instancenorm_op->reduce = NULL;
    }
}

void op_cuda_instancenorm_tp_dealloc(op_t *op)
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

__global__ void instancenorm_sum_mean(
    float *output, const float *input, size_t bs, size_t ch, size_t hw, int hw_near)
{
    // int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw_division = (hw + 1023) / 1024;
    // int bid = blockIdx.x;
    int tid = threadIdx.x;
    int nci = blockIdx.x / hw_division;
    int hwi = blockIdx.x % hw_division * 1024 + threadIdx.x;

    if (hwi >= hw)
        return;
    __shared__ float reduce[ThreadsPerBlock];
    reduce[tid] = input[nci * hw + hwi];
    __syncthreads();
    if (hwi >= hw - hw % ThreadsPerBlock) {
        atomicAdd(output + nci * hw + hwi / ThreadsPerBlock, reduce[tid]);
        return;
    }
    for (int i = ThreadsPerBlock / 2; i > 0; i /= 2) {
        if (tid < i) {
            reduce[tid] += reduce[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output + nci * hw + hwi / ThreadsPerBlock, reduce[0]);
    }
}

__global__ void instancenorm_get_mean(float *input, size_t bsch, size_t HW)
{

    CUDA_KERNEL_LOOP(i, bsch) { input[i] /= HW; }
}

__global__ void instancenorm_sum_variance(
    float *output, const float *input, float *mean, size_t bs, size_t ch, size_t hw, int hw_near)
{
    // int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hw_division = (hw + 1023) / 1024;
    // int bid = blockIdx.x;
    int tid = threadIdx.x;
    int nci = blockIdx.x / hw_division;
    int hwi = blockIdx.x % hw_division * 1024 + threadIdx.x;

    if (hwi >= hw)
        return;
    __shared__ double reduce[ThreadsPerBlock];
    reduce[tid] = pow(input[nci * hw + hwi] - mean[nci], 2);
    __syncthreads();
    if (hwi >= hw - hw % 1024) {
        atomicAdd(output + nci * hw + hwi / 1024, reduce[tid]);
        return;
    }
    for (int i = ThreadsPerBlock / 2; i > 0; i /= 2) {
        if (tid < i) {
            reduce[tid] += reduce[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output + nci * hw + hwi / 1024, reduce[0]);
    }
}

__global__ void instancenorm_get_variance(float *input, size_t bsch, size_t HW)
{
    CUDA_KERNEL_LOOP(i, bsch) { input[i] /= HW; }
}

__global__ void instancenorm_reduece(float *output, float *input, size_t bs, size_t ch, size_t hw)
{
    // int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    atomicAdd(output + bid, input[bid * hw + tid]);
}

__global__ void instancenorm_update_output(
    float *output, const float *input_0, const float *mean, const float *variance,
    const float *input_1, const float *input_2, size_t bs, size_t ch, size_t hw, const float eps)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < bs * ch * hw) {
        int channel_id = (global_idx / hw);
        float std_ch = sqrt(variance[channel_id] + eps);
        float mean_ch = mean[channel_id];
        output[global_idx] = (input_0[global_idx] - mean_ch) / std_ch * input_1[channel_id % ch]
            + input_2[channel_id % ch];
    }
}

void op_cuda_instancenorm_run(op_t *op)
{
    const op_instancenorm_t *instancenorm_op = (op_instancenorm_t *)op;
    int bs = op->input_tensors[0]->shape.dim[0];
    int ch = op->input_tensors[0]->shape.dim[1];
    int hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];

    const float *input_0 = (float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (float *)mem_data(op->input_tensors[1]->mem);
    const float *input_2 = (float *)mem_data(op->input_tensors[2]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);

    float *mean = (float *)mem_data(instancenorm_op->mean);
    float *variance = (float *)mem_data(instancenorm_op->variance);
    float *reduce = (float *)mem_data(instancenorm_op->reduce);

    CUDA_CHECK(
        cudaMemsetAsync(mean, 0, sizeof(float) * ch * bs, CUDA_WORKSPACE_STREAM(op->workspace)));
    CUDA_CHECK(cudaMemsetAsync(
        variance, 0, sizeof(float) * ch * bs, CUDA_WORKSPACE_STREAM(op->workspace)));
    CUDA_CHECK(cudaMemsetAsync(
        reduce, 0, sizeof(float) * ch * bs * hw, CUDA_WORKSPACE_STREAM(op->workspace)));

    int hw_less_near_2_n = caculate_2_n_less(hw);
    instancenorm_sum_mean<<<
        bs * ch *((hw + 1024 - 1) / 1024), 1024, 1024 * sizeof(float),
        CUDA_WORKSPACE_STREAM(op->workspace)>>>(reduce, input_0, bs, ch, hw, hw_less_near_2_n);
    instancenorm_reduece<<<bs * ch, (hw + 1023) / 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        mean, reduce, bs, ch, hw);
    instancenorm_get_mean<<<
        (bs * ch + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        mean, bs * ch, hw);
    CUDA_CHECK(cudaMemsetAsync(
        reduce, 0, sizeof(float) * ch * bs * hw, CUDA_WORKSPACE_STREAM(op->workspace)));
    instancenorm_sum_variance<<<
        bs * ch *((hw + 1024 - 1) / 1024), 1024, 1024 * sizeof(float),
        CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        reduce, input_0, mean, bs, ch, hw, hw_less_near_2_n);
    instancenorm_reduece<<<bs * ch, (hw + 1023) / 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        variance, reduce, bs, ch, hw);
    instancenorm_get_variance<<<
        (bs * ch + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        variance, bs * ch, hw);
    instancenorm_update_output<<<
        (bs * ch * hw + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output_0, input_0, mean, variance, input_1, input_2, bs, ch, hw, instancenorm_op->eps);
}

void op_cuda_instancenorm_tp_prepare(op_t *op)
{
    op_instancenorm_t *instancenorm_op = (op_instancenorm_t *)op;
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32: {
        int bs = op->input_tensors[0]->shape.dim[0];
        int ch = op->input_tensors[0]->shape.dim[1];
        int hw = op->input_tensors[0]->shape.dim[2] * op->input_tensors[0]->shape.dim[3];
        if (instancenorm_op->mean == NULL) {
            instancenorm_op->mean = mem_new(cuda_mem_tp);
        }
        mem_alloc(instancenorm_op->mean, bs * ch * sizeof(float));
        if (instancenorm_op->variance == NULL) {
            instancenorm_op->variance = mem_new(cuda_mem_tp);
        }
        mem_alloc(instancenorm_op->variance, bs * ch * sizeof(float));
        if (instancenorm_op->reduce == NULL) {
            instancenorm_op->reduce = mem_new(cuda_mem_tp);
        }
        mem_alloc(instancenorm_op->reduce, bs * ch * hw * sizeof(float));
        CUDA_CHECK(cudaMemset(mem_data(instancenorm_op->mean), 0, sizeof(float) * ch * bs));
        CUDA_CHECK(cudaMemset(mem_data(instancenorm_op->variance), 0, sizeof(float) * ch * bs));
        CUDA_CHECK(cudaMemset(mem_data(instancenorm_op->reduce), 0, sizeof(float) * ch * bs * hw));
        op->run_func = op_cuda_instancenorm_run;
    } break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
