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

#include <cuda.h>
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
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t hole_h;
    uint32_t hole_w;
    uint32_t group;
    uint32_t deform_group;

    size_t bottom_offset;
    size_t bottom_deform_offset;
    size_t weight_offset;
    size_t bias_offset;
    size_t top_offset;

    cublasHandle_t cublasHandle;

    mem_t *columns;
} op_deform_conv_2d_t;

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N) { return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS; }

__device__ float cuda_deformable_im2col_bilinear(
    const float *bottom_data, const int data_width, const int height, const int width, float h,
    float w)
{

    int h_low = floor(h);
    int w_low = floor(w);
    int h_high;
    int w_high;
    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h = (float)h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w = (float)w_low;
    } else {
        w_high = w_low + 1;
    }

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = bottom_data[h_low * data_width + w_low];
    float v2 = bottom_data[h_low * data_width + w_high];
    float v3 = bottom_data[h_high * data_width + w_low];
    float v4 = bottom_data[h_high * data_width + w_high];
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

float get_gradient_weight(
    float argmax_h, float argmax_w, const int h, const int w, const int height, const int width)
{

    if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
        // empty
        return 0;
    }

    argmax_h = max(argmax_h, (float)0.0f);
    argmax_w = max(argmax_w, (float)0.0f);

    int argmax_h_low = (int)argmax_h;
    int argmax_w_low = (int)argmax_w;
    int argmax_h_high;
    int argmax_w_high;
    if (argmax_h_low >= height - 1) {
        argmax_h_high = argmax_h_low = height - 1;
        argmax_h = (float)argmax_h_low;
    } else {
        argmax_h_high = argmax_h_low + 1;
    }
    if (argmax_w_low >= width - 1) {
        argmax_w_high = argmax_w_low = width - 1;
        argmax_w = (float)argmax_w_low;
    } else {
        argmax_w_high = argmax_w_low + 1;
    }
    float weight = 0;
    if (h == argmax_h_low) {
        if (w == argmax_w_low) {
            weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
            weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
        }
    } else if (h == argmax_h_high) {
        if (w == argmax_w_low) {
            weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
        } else if (w == argmax_w_high) {
            weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
        }
    }
    return weight;
}

float get_coordinate_weight(
    float argmax_h, float argmax_w, const int height, const int width, const float *im_data,
    const int data_width, const int bp_dir)
{

    if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
        // empty
        return 0;
    }

    if (argmax_h < 0)
        argmax_h = 0;
    if (argmax_w < 0)
        argmax_w = 0;

    int argmax_h_low = (int)argmax_h;
    int argmax_w_low = (int)argmax_w;
    int argmax_h_high;
    int argmax_w_high;
    if (argmax_h_low >= height - 1) {
        argmax_h_high = argmax_h_low = height - 1;
        argmax_h = (float)argmax_h_low;
    } else {
        argmax_h_high = argmax_h_low + 1;
    }
    if (argmax_w_low >= width - 1) {
        argmax_w_high = argmax_w_low = width - 1;
        argmax_w = (float)argmax_w_low;
    } else {
        argmax_w_high = argmax_w_low + 1;
    }
    float weight = 0;

    if (bp_dir == 0) {
        weight += -1 * (argmax_w_low + 1 - argmax_w)
            * im_data[argmax_h_low * data_width + argmax_w_low];
        weight
            += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
        weight
            += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
    } else if (bp_dir == 1) {
        weight += -1 * (argmax_h_low + 1 - argmax_h)
            * im_data[argmax_h_low * data_width + argmax_w_low];
        weight
            += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
        weight
            += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
    }

    return weight;
}

__global__ void cuda_deformable_im2col_gpu_kernel(
    const int n, const float *data_im, const float *data_offset, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int height_col, const int width_col,
    float *data_col)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        // index index of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col) / height_col;
        const int c_col = c_im * kernel_h * kernel_w;

        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        float *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const float *data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
        const float *data_offset_ptr = data_offset
            + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;

        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                const int data_offset_h_ptr
                    = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                const int data_offset_w_ptr
                    = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                const float offset_h = data_offset_ptr[data_offset_h_ptr];
                const float offset_w = data_offset_ptr[data_offset_w_ptr];
                float val = static_cast<float>(0);
                const float h_im = h_in + i * dilation_h + offset_h;
                const float w_im = w_in + j * dilation_w + offset_w;
                if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                    const float map_h = i * dilation_h + offset_h;
                    const float map_w = j * dilation_w + offset_w;
                    const int cur_height = height - h_in;
                    const int cur_width = width - w_in;
                    val = cuda_deformable_im2col_bilinear(
                        data_im_ptr, width, cur_height, cur_width, map_h, map_w);
                }
                *data_col_ptr = val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void cuda_deformable_im2col(
    float *data_im, float *data_offset, const int channels, const int height, const int width,
    const int ksize_h, const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w, const int deformable_group,
    float *data_col)
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col;
    int channel_per_deformable_group = channels / deformable_group;
    // Launch
    cuda_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, data_im, data_offset, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h,
        stride_w, dilation_h, dilation_w, channel_per_deformable_group, height_col, width_col,
        data_col);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
        // TODO(BZ) panic
    }
}

int deform_conv_forward_cuda(
    op_t *op, float *input, float *weight, float *offset, float *output, float *columns, int kH,
    int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW, int groups,
    int deformable_group)
{
    op_deform_conv_2d_t *conv_op = (op_deform_conv_2d_t *)op;
    int batchSize = op->input_tensors[0]->shape.dim[0];
    int nInputPlane = op->input_tensors[0]->shape.dim[1];
    int inputHeight = op->input_tensors[0]->shape.dim[2];
    int inputWidth = op->input_tensors[0]->shape.dim[3];

    int nOutputPlane = op->output_tensors[0].shape.dim[1];

    int outputWidth = op->output_tensors[0].shape.dim[2];
    int outputHeight = op->output_tensors[0].shape.dim[3];

    int columns_off = (int)(nInputPlane / groups) * outputWidth * outputHeight * kW * kH;
    int output_off = (int)(nOutputPlane / groups) * outputWidth * outputHeight;
    int weight_off = (int)(nOutputPlane / groups) * kW * kH * (int)(nInputPlane / groups);
    int elt;
    for (elt = 0; elt < batchSize; elt++) {

        float *input_n = &input[elt * nInputPlane * inputHeight * inputWidth];
        float *offset_n = &offset[elt * kW * kH * nInputPlane * outputHeight * outputWidth];
        float *output_n = &output[elt * nOutputPlane * outputHeight * outputWidth];
        // output_n.zero_();
        cuda_deformable_im2col(
            input_n, offset_n, nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
            dilationH, dilationW, deformable_group, columns);
        int g = 0;
        for (g = 0; g < groups; g++) {
            float *columns_g = &columns[g * columns_off];
            float *weight_g = &weight[g * weight_off];
            float *output_g = &output_n[g * output_off];
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(
                conv_op->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, (int)(nOutputPlane / groups),
                outputHeight * outputWidth, (int)(nInputPlane / groups) * kH * kW, &alpha, weight_g,
                (int)(nInputPlane / groups) * kH * kW, columns_g, outputHeight * outputWidth, &beta,
                output_g, (int)(nOutputPlane / groups));
        }
    }

    return 0;
}

op_deform_conv_2d_t *op_cuda_deform_conv_2d_tp_alloc(workspace_t *ws)
{
    op_deform_conv_2d_t *res = new op_deform_conv_2d_t;
    memset(res, 0, sizeof(op_t));
    res->cublasHandle = ((cuda_workspace_t *)ws)->cublasHandle;
    (void)ws;
    res->columns = mem_new(cuda_mem_tp);
    return res;
}

void op_cuda_deform_conv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_deform_conv_2d_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_deform_conv_2d_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_deform_conv_2d_t *)op)->kernel_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_deform_conv_2d_t *)op)->pad_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_deform_conv_2d_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_deform_conv_2d_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_deform_conv_2d_t *)op)->stride_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_deform_conv_2d_t *)op)->group));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_DEFORM_GROUP, dtUINT32, &((op_deform_conv_2d_t *)op)->deform_group));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_HOLE_H, dtUINT32, &((op_deform_conv_2d_t *)op)->hole_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_HOLE_W, dtUINT32, &((op_deform_conv_2d_t *)op)->hole_w));
}

void op_cuda_deform_conv_2d_tp_destroy(op_t *op) { (void)op; }

void op_cuda_deform_conv_2d_tp_dealloc(op_t *op)
{
    op_deform_conv_2d_t *conv_op = (op_deform_conv_2d_t *)op;
    if (NULL != conv_op->columns) {
        mem_delete(conv_op->columns);
    }
    delete conv_op;
}

static void op_cuda_deform_conv_2d_run(op_t *op)
{
    op_deform_conv_2d_t *conv_op = (op_deform_conv_2d_t *)op;
    deform_conv_forward_cuda(
        op, (float *)mem_data(op->input_tensors[0]->mem),
        (float *)mem_data(op->input_tensors[2]->mem), (float *)mem_data(op->input_tensors[1]->mem),
        (float *)mem_data(op->output_tensors[0].mem), (float *)mem_data(conv_op->columns),
        conv_op->kernel_h, conv_op->kernel_w, conv_op->pad_h, conv_op->pad_w, conv_op->stride_h,
        conv_op->stride_w, conv_op->hole_h, conv_op->hole_w, conv_op->group, conv_op->deform_group);
}

void op_cuda_deform_conv_2d_tp_prepare(op_t *op)
{
    op_deform_conv_2d_t *conv_op = (op_deform_conv_2d_t *)op;

    mem_alloc(
        conv_op->columns,
        sizeof(float) * op->input_tensors[1]->shape.dim[1] * op->input_tensors[1]->shape.dim[2]
            * op->input_tensors[1]->shape.dim[3]);

    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_deform_conv_2d_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
