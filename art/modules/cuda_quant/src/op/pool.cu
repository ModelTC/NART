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

#include "art/cuda_quant/cuda_quant_mem.h"
#include "art/cuda_quant/cuda_quant_op_settings.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "pool_impl.cuh"

#include "../cuda_quant_workspace.h"

#define PROFILE_TIMES 20

#ifdef __cplusplus
extern "C" {
#endif

std::vector<std::function<void(
    int8_t *, int8_t *, int, int, int, int, int, int, int, int, int, int, workspace_t *)>>
    pool2d_table = {
        // maxpool2d_s8
        maxpool2d_s8_cu_v16<1024>,
        maxpool2d_s8_cu_v16<512>,
        maxpool2d_s8_cu_v16<256>,
        maxpool2d_s8_cu_v1<256>,
        maxpool2d_s8_cu_v1<128>,
        maxpool2d_s8_cu_v1<64>,

        // maxpool2d_s4
        maxpool2d_s4_cu_v32<2048>,
        maxpool2d_s4_cu_v32<1024>,
        maxpool2d_s4_cu_v32<512>,
        maxpool2d_s4_cu_v32<256>,
        maxpool2d_s4_cu_v2<256>,
        maxpool2d_s4_cu_v2<128>,
        maxpool2d_s4_cu_v2<64>,

        // avgpool2d_s8
        avgpool2d_s8_cu_v16<1024>,
        avgpool2d_s8_cu_v16<512>,
        avgpool2d_s8_cu_v16<256>,
        avgpool2d_s8_cu_v1<256>,
        avgpool2d_s8_cu_v1<128>,
        avgpool2d_s8_cu_v1<64>,

        // avgpool2d_s4
        avgpool2d_s4_cu_v32<1024>,
        avgpool2d_s4_cu_v32<512>,
        avgpool2d_s4_cu_v32<256>,
        avgpool2d_s4_cu_v2<256>,
        avgpool2d_s4_cu_v2<128>,
        avgpool2d_s4_cu_v2<64>,
    };

std::vector<std::vector<int>> pool2d_config = {
  // maxpool2d_s8
    {16,  1024},
    { 16, 512 },
    { 16, 256 },
    { 1,  256 },
    { 1,  128 },
    { 1,  64  },

 // maxpool2d_s4
    { 32, 2048},
    { 32, 1024},
    { 32, 512 },
    { 32, 256 },
    { 2,  256 },
    { 2,  128 },
    { 2,  64  },

 // avgpool2d_s8
    { 16, 1024},
    { 16, 512 },
    { 16, 256 },
    { 1,  256 },
    { 1,  128 },
    { 1,  64  },

 // avgpool2d_s4
    { 32, 1024},
    { 32, 512 },
    { 32, 256 },
    { 2,  256 },
    { 2,  128 },
    { 2,  64  },
};

int maxpool2d_s8_begin = 0, maxpool2d_s8_num = 6;
int maxpool2d_s4_begin = 6, maxpool2d_s4_num = 7;
int avgpool2d_s8_begin = 13, avgpool2d_s8_num = 6;
int avgpool2d_s4_begin = 19, avgpool2d_s4_num = 6;

typedef struct {
    op_t o;
    uint32_t pool_method; //目前默认使用MAX
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    bool ceil_mode;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;

    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;

    cudnnTensorDescriptor_t pool_in;
    cudnnTensorDescriptor_t pool_out;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnPoolingMode_t pool_mode;

    std::function<void(
        int8_t *, int8_t *, int, int, int, int, int, int, int, int, int, int, workspace_t *)>
        pool_best_func;
    bool if_use_CUDNN;

} op_pool_t;

op_pool_t *op_cuda_quant_pool_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_pool_t *res = new op_pool_t;
    memset(res, 0, sizeof(op_t));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->pool_in));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->pool_out));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&res->pool_desc));

    return res;
}

void op_cuda_quant_pool_tp_config(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    CHECK(op_setting_single_get(op, SETTING_POOL_KERNEL_H, dtUINT32, &((op_pool_t *)op)->kernel_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_KERNEL_W, dtUINT32, &((op_pool_t *)op)->kernel_w));
    CHECK(
        op_setting_single_get(op, SETTING_POOL_METHOD, dtUINT32, &((op_pool_t *)op)->pool_method));
    CHECK(op_setting_single_get(op, SETTING_POOL_PAD_H, dtUINT32, &((op_pool_t *)op)->pad_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_PAD_W, dtUINT32, &((op_pool_t *)op)->pad_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_STRIDE_H, dtUINT32, &((op_pool_t *)op)->stride_h));
    CHECK(op_setting_single_get(op, SETTING_POOL_STRIDE_W, dtUINT32, &((op_pool_t *)op)->stride_w));
    CHECK(op_setting_single_get(op, SETTING_POOL_CEIL_MODE, dtBOOL, &((op_pool_t *)op)->ceil_mode));

    CHECK(pool_op->pool_method == SETTING_POOL_MAX || pool_op->pool_method == SETTING_POOL_AVE);
    switch (pool_op->pool_method) {
    case SETTING_POOL_MAX:
        pool_op->pool_mode = CUDNN_POOLING_MAX;
        break;
    case SETTING_POOL_AVE:
        pool_op->pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
    default:
        CHECK(false);
    }

    size_t len_bits;
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OBITS, dtUINT8, &len_bits, &((op_pool_t *)op)->obits));

    pool_op->if_use_CUDNN = false;
}

void op_cuda_quant_pool_tp_destroy(op_t *op) { (void)op; }

void op_cuda_quant_pool_tp_dealloc(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(pool_op->pool_in));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(pool_op->pool_out));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_op->pool_desc));

    delete pool_op;
}

static void op_cuda_quant_pool_run(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    float alpha = 1.0f;
    float beta = 0.0f;
    int8_t *input = (int8_t *)mem_data(op->input_tensors[0]->mem);
    int8_t *output = (int8_t *)mem_data(op->output_tensors->mem);
    shape_t *input_shape = &op->input_tensors[0]->shape;
    int batch = input_shape->dim[0];
    int channel = input_shape->dim[1];
    int ih = input_shape->dim[2];
    int iw = input_shape->dim[3];

    if (pool_op->obits[0] == 8 && pool_op->if_use_CUDNN) {
        CUDNN_CHECK(cudnnPoolingForward(
            CUDA_WORKSPACE_CUDNNHDL(op->workspace), pool_op->pool_desc, &alpha, pool_op->pool_in,
            mem_data(op->input_tensors[0]->mem), &beta, pool_op->pool_out,
            mem_data(op->output_tensors[0].mem)));
    } else {
        pool_op->pool_best_func(
            input, output, batch, ih, iw, channel, pool_op->kernel_h, pool_op->kernel_w,
            pool_op->pad_h, pool_op->pad_w, pool_op->stride_h, pool_op->stride_w, op->workspace);
    }
}

void profile_pool(op_t *op, int out_bits)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    float alpha = 1.0f;
    float beta = 0.0f;
    int8_t *input = (int8_t *)mem_data(op->input_tensors[0]->mem);
    int8_t *output = (int8_t *)mem_data(op->output_tensors->mem);
    shape_t *input_shape = &op->input_tensors[0]->shape;
    int batch = input_shape->dim[0];
    int channel = input_shape->dim[1];
    int ih = input_shape->dim[2];
    int iw = input_shape->dim[3];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float min_time = 1e10, time;

    // set index for table
    int i_begin, i_end;
    if (out_bits == 8) {
        if (pool_op->pool_method == SETTING_POOL_MAX) {
            i_begin = maxpool2d_s8_begin;
            i_end = maxpool2d_s8_begin + maxpool2d_s8_num - 1;
        } else {
            i_begin = avgpool2d_s8_begin;
            i_end = avgpool2d_s8_begin + avgpool2d_s8_num - 1;
        }

    } else {
        if (pool_op->pool_method == SETTING_POOL_MAX) {
            i_begin = maxpool2d_s4_begin;
            i_end = maxpool2d_s4_begin + maxpool2d_s4_num - 1;
        } else {
            i_begin = avgpool2d_s4_begin;
            i_end = avgpool2d_s4_begin + avgpool2d_s4_num - 1;
        }
    }

    // printf("profiling pool kernel. shape: (%d, %d, %d, %d), k%dp%ds%d\n",
    //         batch, ih, iw, channel, pool_op->kernel_h, pool_op->pad_h, pool_op->stride_h);

    for (int i = i_begin; i <= i_end; i++) {

        // printf("config: [VECTOR-%d, NUM-%d]\t", pool2d_config[i][0], pool2d_config[i][1]);
        cudaEventRecord(start);
        for (int j = 0; j < PROFILE_TIMES; j++) {
            pool2d_table[i](
                input, output, batch, ih, iw, channel, pool_op->kernel_h, pool_op->kernel_w,
                pool_op->pad_h, pool_op->pad_w, pool_op->stride_h, pool_op->stride_w,
                op->workspace);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());
        cudaEventElapsedTime(&time, start, stop);

        time /= PROFILE_TIMES;
        // printf("time: %f\n", time);

        if (time < min_time) {
            min_time = time;
            pool_op->pool_best_func = pool2d_table[i];
        }
    }

    // profile cudnn-int8
    if (out_bits == 8) {
        cudaEventRecord(start);
        for (int j = 0; j < PROFILE_TIMES; j++) {
            CUDNN_CHECK(cudnnPoolingForward(
                CUDA_WORKSPACE_CUDNNHDL(op->workspace), pool_op->pool_desc, &alpha,
                pool_op->pool_in, mem_data(op->input_tensors[0]->mem), &beta, pool_op->pool_out,
                mem_data(op->output_tensors[0].mem)));
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        CUDA_CHECK(cudaPeekAtLastError());
        cudaEventElapsedTime(&time, start, stop);

        time /= PROFILE_TIMES;
        // printf("cudnn time: %f\n", time);

        if (time < min_time) {
            pool_op->if_use_CUDNN = true;
        }
    }
}

void op_cuda_quant_pool_tp_prepare(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    shape_t *input_shape = &op->input_tensors[0]->shape;
    shape_t *output_shape = &op->output_tensors[0].shape;
    int i_b = input_shape->dim[0];
    int i_c = input_shape->dim[1];
    int i_h = input_shape->dim[2];
    int i_w = input_shape->dim[3];

    int o_b = output_shape->dim[0];
    int o_c = output_shape->dim[1];
    int o_h = output_shape->dim[2];
    int o_w = output_shape->dim[3];

    int out_bits = pool_op->obits[0];

    if (out_bits == 8) {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            pool_op->pool_in, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, i_b, i_c, i_h, i_w));

        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            pool_op->pool_out, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, o_b, o_c, o_h, o_w));
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(
            pool_op->pool_desc, pool_op->pool_mode, CUDNN_PROPAGATE_NAN, pool_op->kernel_h,
            pool_op->kernel_w, pool_op->pad_h, pool_op->pad_w, pool_op->stride_h,
            pool_op->stride_w));
    }

    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }

    profile_pool(op, out_bits);
    op->run_func = op_cuda_quant_pool_run;
}

#ifdef __cplusplus
}
#endif
