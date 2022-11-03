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

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

// #include <cuda_runtime_api.h>

#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

enum GROUPMODE { CORR, DEPTHWISE };

typedef struct {
    op_t o;
    uint32_t groups;
    enum GROUPMODE group_mode;

    size_t num_scales;
    size_t kernel_spatial_dim;
    size_t input_spatial_dim;
    size_t output_spatial_dim;

    cublasHandle_t cublasHandle;

    mem_t *columns;

} op_correlation_t;

op_correlation_t *op_cuda_correlation_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_correlation_t *res = new op_correlation_t;
    memset(res, 0, sizeof(op_t));
    res->cublasHandle = ((cuda_workspace_t *)ws)->cublasHandle;
    res->columns = mem_new(cuda_mem_tp);
    return res;
}

void op_cuda_correlation_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CORRELATION_GROUPS, dtUINT32, &((op_correlation_t *)op)->groups));
}

void op_cuda_correlation_tp_destroy(op_t *op) { (void)op; }

void op_cuda_correlation_tp_dealloc(op_t *op)
{
    op_correlation_t *corr_op = (op_correlation_t *)op;
    if (NULL != corr_op->columns) {
        mem_delete(corr_op->columns);
    }
    delete corr_op;
}

__global__ void spatial_depthwise_convolution(
    const float *input, const float *weight, float *output, const int input_width,
    const int input_height, const int output_width, const int output_height, const int kernel_width,
    const int kernel_height, const int k_size, const int count, const int output_channels)
{
    const int KW_LIMIT = (k_size != 0) ? k_size : kernel_width;
    const int KH_LIMIT = (k_size != 0) ? k_size : kernel_height;

    CUDA_KERNEL_LOOP(index, count)
    {

        int indtmp1 = index / output_width;
        const int w = index - indtmp1 * output_width;
        int indtmp2 = indtmp1 / output_height;
        const int h = indtmp1 - indtmp2 * output_height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1 / output_channels;
        const int c = indtmp1 - indtmp2 * output_channels;
        const int n = indtmp2;

        int weight_offset = (n * output_channels + c) * kernel_height * kernel_width;

        float value = 0.f;
        const int offset0 = (n * output_channels + c) * input_height * input_width;

        for (int kh = 0; kh < KH_LIMIT; ++kh) {
            const int h_in = h + kh;
            const int weight_offset_h = weight_offset + kh * KW_LIMIT;
            for (int kw = 0; kw < KW_LIMIT; ++kw) {
                const int w_in = w + kw;

                const int offset = offset0 + h_in * input_width + w_in;

                value += weight[weight_offset_h + kw] * input[offset];
            }
        }

        output[index] = value;
    }
}

__global__ void im2col_gpu_kernel(
    const int n, const float *data_im, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int height_col, const int width_col, float *data_col)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * kernel_h * kernel_w;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;
        float *data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float *data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width)
                    ? data_im_ptr[i * width + j]
                    : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void depthwise_conv(
    const op_t *op, const int count, const int output_channels, const int input_width,
    const int input_height, const int output_width, const int output_height, const int kernel_width,
    const int kernel_height, const float *weight, const float *input, float *output)
{
    int kernel_size = 0;

    switch (kernel_height) {
    case 1:
        kernel_size = 1;
        break;
    case 2:
        kernel_size = 2;
        break;
    case 3:
        kernel_size = 3;
        break;
    case 4:
        kernel_size = 4;
        break;
    case 5:
        kernel_size = 5;
        break;
    case 6:
        kernel_size = 6;
        break;
    case 7:
        kernel_size = 7;
        break;
    default:
        kernel_size = 0;
        break;
    }

    spatial_depthwise_convolution<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        input, weight, output, input_width, input_height, output_width, output_height, kernel_width,
        kernel_height, kernel_size, count, output_channels);

    // CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaPeekAtLastError());
}

void im2col_gpu(
    const float *data_im, const int channels, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float *data_col, const op_t *op)
{
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;

    im2col_gpu_kernel<<<
        (num_kernels + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        height_col, width_col, data_col);

    // CUDA_POST_KERNEL_CHECK;
    CUDA_CHECK(cudaPeekAtLastError());

    return;
}

static void op_cuda_correlation_run(op_t *op)
{
    op_correlation_t *corr_op = (op_correlation_t *)op;
    const size_t input_height = op->input_tensors[0]->shape.dim[2];
    const size_t input_width = op->input_tensors[0]->shape.dim[3];
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t kernel_height = op->input_tensors[1]->shape.dim[2];
    const size_t kernel_width = op->input_tensors[1]->shape.dim[3];
    const size_t channel_out = op->input_tensors[1]->shape.dim[1];
    const size_t batch_size = op->input_tensors[1]->shape.dim[0];
    const size_t output_height = op->output_tensors[0].shape.dim[2];
    const size_t output_width = op->output_tensors[0].shape.dim[3];
    const size_t count = shape_count(&op->output_tensors[0].shape);

    // input_0: weight  input_1: data
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    float *columns = (float *)mem_data(corr_op->columns);

    const float *kernel_data = NULL;
    const float *input_data = NULL;
    float *output_data = NULL;

    float alpha = 1.f;
    float beta = 0.f;

    if (corr_op->group_mode == DEPTHWISE) {
        depthwise_conv(
            op, count, 1, // channels
            input_width, input_height, output_width, output_height, kernel_width, kernel_height,
            input_1, input_0, output_0);
        return;
    }

    for (size_t group_index = 0; group_index < corr_op->groups * batch_size; ++group_index) {
        if (corr_op->group_mode == DEPTHWISE) {
            input_data = input_0 + group_index * input_height * kernel_width;
            kernel_data = input_1 + group_index * kernel_height * kernel_width;
            output_data = output_0 + group_index * output_height * output_width;
        } else {
            input_data = input_0 + group_index * corr_op->input_spatial_dim;
            kernel_data = input_1 + group_index * corr_op->kernel_spatial_dim;
            output_data = output_0 + group_index * corr_op->output_spatial_dim;
        }

        // correlate_gpu(kernel_data, input_data, output_data);
        size_t temp_channel_out;
        size_t kernel_dim;
        size_t single_input_size, single_output_size;

        if (corr_op->group_mode == DEPTHWISE) {
            temp_channel_out = channel_out;
        } else {
            temp_channel_out = channel_out / channel_in;
        }

        single_input_size = (channel_in / corr_op->groups) * input_height * input_width;
        single_output_size = (temp_channel_out / corr_op->groups) * output_height * output_width;
        kernel_dim = (channel_in / corr_op->groups) * kernel_height * kernel_width;

        for (size_t scale_index = 0; scale_index < corr_op->num_scales; ++scale_index) {
            const float *input_ptr = input_data + scale_index * single_input_size;
            float *output_ptr = output_data + scale_index * single_output_size;

            im2col_gpu(
                input_ptr, channel_in / corr_op->groups, input_height, input_width, kernel_height,
                kernel_width, 0, 0, 1, 1, columns, op);

            CUBLAS_CHECK(cublasSgemm(
                corr_op->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, output_height * output_width,
                temp_channel_out / corr_op->groups, kernel_dim, &alpha, columns,
                output_height * output_width, kernel_data, kernel_dim, &beta, output_ptr,
                output_height * output_width));
        }
    }
}

void op_cuda_correlation_tp_prepare(op_t *op)
{
    op_correlation_t *corr_op = (op_correlation_t *)op;

    tensor_t *input_tensor = op->input_tensors[0];
    tensor_t *kernel_tensor = op->input_tensors[1];
    tensor_t *output_tensor = &op->output_tensors[0];

    const size_t kernel_height = kernel_tensor->shape.dim[2];
    const size_t kernel_width = kernel_tensor->shape.dim[3];
    const size_t input_height = input_tensor->shape.dim[2];
    const size_t input_width = input_tensor->shape.dim[3];
    const size_t output_height = output_tensor->shape.dim[2];
    const size_t output_width = output_tensor->shape.dim[3];
    const size_t batch_size = kernel_tensor->shape.dim[0];
    const size_t input_num = input_tensor->shape.dim[0];
    const size_t channel_in = input_tensor->shape.dim[1];
    const size_t channel_out = kernel_tensor->shape.dim[1];

    if (corr_op->groups == 1) {
        corr_op->group_mode = CORR;
    } else {
        corr_op->group_mode = DEPTHWISE;
        corr_op->groups = channel_in;
    }

    corr_op->num_scales = input_num / batch_size;
    corr_op->kernel_spatial_dim = channel_out * kernel_height * kernel_width;
    corr_op->input_spatial_dim = channel_in * input_height * input_width;
    corr_op->output_spatial_dim = batch_size * output_height * output_width;

    mem_alloc(
        corr_op->columns,
        sizeof(float) * channel_in / corr_op->groups * kernel_height * kernel_width * output_height
            * output_width);

    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_correlation_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
