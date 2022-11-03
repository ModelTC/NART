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

#include <cuda_runtime.h>
#include <cudnn.h>

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t num_output;
    uint32_t *pad;
    uint32_t *kernel;
    uint32_t *stride;
    uint32_t *hole;
    uint32_t group;
} op_conv_nd_t;

void op_cuda_conv_1d_tp_prepare(op_t *op);

op_conv_nd_t *op_cuda_conv_nd_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_conv_nd_t *res = new op_conv_nd_t;
    memset(res, 0, sizeof(op_t));
    return res;
}

void op_cuda_conv_nd_tp_config(op_t *op)
{
    size_t kernel_length;
    size_t pad_length;
    size_t stride_length;
    size_t hole_length;
    size_t length = op->input_tensors[0]->shape.dim_size - 2;
    CHECK(op_setting_single_get(
        op, SETTING_CONV_ND_NUM_OUTPUT, dtUINT32, &((op_conv_nd_t *)op)->num_output));
    CHECK(op_setting_array_get(
        op, SETTING_CONV_ND_KERNEL, dtUINT32, &kernel_length, &((op_conv_nd_t *)op)->kernel));
    CHECK(op_setting_array_get(
        op, SETTING_CONV_ND_PAD, dtUINT32, &pad_length, &((op_conv_nd_t *)op)->pad));
    CHECK(op_setting_array_get(
        op, SETTING_CONV_ND_STRIDE, dtUINT32, &stride_length, &((op_conv_nd_t *)op)->stride));
    CHECK(op_setting_array_get(
        op, SETTING_CONV_ND_HOLE, dtUINT32, &hole_length, &((op_conv_nd_t *)op)->hole));
    CHECK(op_setting_single_get(op, SETTING_CONV_ND_GROUP, dtUINT32, &((op_conv_nd_t *)op)->group));
    if (kernel_length != length)
        LOG_error("kernel_length don't match input0_length\n");
    if (pad_length != length)
        LOG_error("pad_length don't match input0_length\n");
    if (stride_length != length)
        LOG_error("stride_length don't match input0_length\n");
    if (hole_length != length)
        LOG_error("hole_length don't match input0_length\n");
}

void op_cuda_conv_nd_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void op_cuda_conv_nd_tp_destroy(op_t *op) { (void)op; }

// for i in range(batch_size):
//     for g in range(group):
//         for k in range(size_pergroup_out):
//             for m in range(output_t):
// #you are considering output[i][j *size_pergroup_out + k][m]
//                     for ii in range(size_group_in):
//                         for jj in range(kernel):
//                             jjj=-m*stride+jj*(hole-1)
//                             output[i][g*size_pergroup_out+k][m]+=
//                             input[i][g*size_pergroup_in+ii][jjj]*
//                             input[output_channel][input_channel][jj]

__global__ void conv_without_bias(
    float *output, const float *input1, const float *input2, size_t batch_size, size_t group,
    size_t size_per_group_out, size_t size_per_group_in, size_t output_t, size_t input_t,
    size_t pad, size_t stride, size_t kernel, size_t size_input_channel, size_t size_output_channel,
    int hole)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size * group * size_per_group_out * output_t)
        return;
    int m = global_idx % output_t;
    global_idx /= output_t;
    int k = global_idx % size_per_group_out;
    global_idx /= size_per_group_out;
    int g = global_idx % group;
    global_idx /= group;
    int i = global_idx % batch_size;

    int output_channel = g * size_per_group_out + k;
    int start = m * stride - pad;
    float plus = 0;
    int ii, jj;
    for (ii = 0; ii < size_per_group_in; ii++)
        for (jj = 0; jj < kernel; jj++) {
            if (jj + jj * (hole - 1) + start < 0 || jj + jj * (hole - 1) + start >= input_t) {
                continue;
            }
            plus += input1
                        [i * size_input_channel * input_t + (ii + g * size_per_group_in) * input_t
                         + jj + jj * (hole - 1) + start]
                * input2
                    [output_channel * size_input_channel * kernel
                     + (ii + g * size_per_group_in) * kernel + jj];
        }
    output[i * size_output_channel * output_t + output_channel * output_t + m] = plus;
}

__global__ void conv_with_bias(
    float *output, const float *input1, const float *input2, const float *input3, size_t batch_size,
    size_t group, size_t size_per_group_out, size_t size_per_group_in, size_t output_t,
    size_t input_t, size_t pad, size_t stride, size_t kernel, size_t size_input_channel,
    size_t size_output_channel, int hole)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= batch_size * group * size_per_group_out * output_t)
        return;
    int channel = (global_idx / output_t) % (group * size_per_group_out);
    int m = global_idx % output_t;
    global_idx /= output_t;
    int k = global_idx % size_per_group_out;
    global_idx /= size_per_group_out;
    int g = global_idx % group;
    global_idx /= group;
    int i = global_idx % batch_size;

    int output_channel = g * size_per_group_out + k;
    int start = m * stride - pad;
    float plus = 0;
    int ii, jj;
    for (ii = 0; ii < size_per_group_in; ii++)
        for (jj = 0; jj < kernel; jj++) {
            if (jj + jj * (hole - 1) + start < 0 || jj + jj * (hole - 1) + start >= input_t) {
                continue;
            }
            plus += input1
                        [i * size_input_channel * input_t + (ii + g * size_per_group_in) * input_t
                         + jj + jj * (hole - 1) + start]
                * input2
                    [output_channel * size_input_channel * kernel
                     + (ii + g * size_per_group_in) * kernel + jj];
        }
    plus += input3[channel];
    output[i * size_output_channel * output_t + output_channel * output_t + m] = plus;
}

void op_cuda_conv_nd_tp_prepare(op_t *op)
{
    if (op->input_tensors[0]->shape.dim_size == 3)
        op_cuda_conv_1d_tp_prepare(op);
    else
        LOG_error("conv besides 1d and 2d have not been realized\n");
}

static void op_cuda_conv_1d_group_run(op_t *op)
{
    const op_conv_nd_t *conv_op = (op_conv_nd_t *)op;
    const float *input1 = (float *)mem_data(op->input_tensors[0]->mem);
    const float *input2 = (float *)mem_data(op->input_tensors[1]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    size_t output_channel = op->input_tensors[1]->shape.dim[0];
    size_t input_channel = op->input_tensors[1]->shape.dim[1];
    size_t size_per_group_out = output_channel / conv_op->group;
    size_t size_per_group_in = input_channel / conv_op->group;
    size_t input_t = op->input_tensors[0]->shape.dim[2];
    size_t output_t = op->output_tensors[0].shape.dim[2];
    size_t kernel = conv_op->kernel[0];
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t group = conv_op->group;

    conv_without_bias<<<
        (batch_size * group * size_per_group_out * output_t + 1024 - 1) / 1024, 1024, 0,
        CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output, input1, input2, batch_size, group, size_per_group_out, size_per_group_in, output_t,
        input_t, conv_op->pad[0], conv_op->stride[0], kernel, input_channel, output_channel,
        conv_op->hole[0]);
}

static void op_cuda_conv_1d_group_bias_run(op_t *op)
{
    const op_conv_nd_t *conv_op = (op_conv_nd_t *)op;
    const float *input1 = (float *)mem_data(op->input_tensors[0]->mem);
    const float *input2 = (float *)mem_data(op->input_tensors[1]->mem);
    const float *input3 = (float *)mem_data(op->input_tensors[2]->mem);
    float *output = (float *)mem_data(op->output_tensors[0].mem);

    size_t output_channel = op->input_tensors[1]->shape.dim[0];
    size_t input_channel = op->input_tensors[1]->shape.dim[1];
    size_t size_per_group_out = output_channel / conv_op->group;
    size_t size_per_group_in = input_channel / conv_op->group;
    size_t input_t = op->input_tensors[0]->shape.dim[2];
    size_t output_t = op->output_tensors[0].shape.dim[2];
    size_t kernel = conv_op->kernel[0];
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t group = conv_op->group;
    // size_t output_length = batch_size * output_channel * output_t;

    conv_with_bias<<<
        (batch_size * group * size_per_group_out * output_t + 1024 - 1) / 1024, 1024, 0,
        CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        output, input1, input2, input3, batch_size, group, size_per_group_out, size_per_group_in,
        output_t, input_t, conv_op->pad[0], conv_op->stride[0], kernel, input_channel,
        output_channel, conv_op->hole[0]);
}

void op_cuda_conv_1d_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; i++) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; i++) {
        tensor_alloc(&op->output_tensors[i]);
    }

    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        if (2 == op->input_size) {
            op->run_func = op_cuda_conv_1d_group_run;
        } else
            op->run_func = op_cuda_conv_1d_group_bias_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
