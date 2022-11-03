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
    uint32_t pool_method; //目前默认使用MAX
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    bool ceil_mode;

    cudnnTensorDescriptor_t pool_in;
    cudnnTensorDescriptor_t pool_out;
    cudnnPoolingDescriptor_t pool_desc;
    cudnnPoolingMode_t pool_mode;
} op_pool_t;

op_pool_t *op_cuda_pool_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_pool_t *res = new op_pool_t;
    memset(res, 0, sizeof(op_t));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->pool_in));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->pool_out));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&res->pool_desc));
    return res;
}

void op_cuda_pool_tp_config(op_t *op)
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
}

void op_cuda_pool_tp_destroy(op_t *op) { (void)op; }

void op_cuda_pool_tp_dealloc(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(pool_op->pool_in));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(pool_op->pool_out));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_op->pool_desc));

    delete pool_op;
}

static void op_cuda_pool_run(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(
        CUDA_WORKSPACE_CUDNNHDL(op->workspace), pool_op->pool_desc, &alpha, pool_op->pool_in,
        mem_data(op->input_tensors[0]->mem), &beta, pool_op->pool_out,
        mem_data(op->output_tensors[0].mem)));
}

void op_cuda_pool_tp_prepare(op_t *op)
{
    op_pool_t *pool_op = (op_pool_t *)op;
    shape_t *shape = &op->input_tensors[0]->shape;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        pool_op->pool_in, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape->dim[0], shape->dim[1],
        shape->dim[2], shape->dim[3]));
    shape = &op->output_tensors[0].shape;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        pool_op->pool_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape->dim[0], shape->dim[1],
        shape->dim[2], shape->dim[3]));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_op->pool_desc, pool_op->pool_mode, CUDNN_PROPAGATE_NAN, pool_op->kernel_h,
        pool_op->kernel_w, pool_op->pad_h, pool_op->pad_w, pool_op->stride_h, pool_op->stride_w));
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_pool_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
