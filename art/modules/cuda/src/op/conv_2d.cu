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
    uint32_t pad_h;
    uint32_t pad_w;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t hole_h;
    uint32_t hole_w;
    uint32_t group;

    size_t bottom_offset;
    size_t weight_offset;
    size_t bias_offset;
    size_t top_offset;

    mem_t *ws;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t convIn;
    cudnnFilterDescriptor_t convW;
    cudnnTensorDescriptor_t convBias;
    cudnnTensorDescriptor_t convOut;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
} op_conv_2d_t;

op_conv_2d_t *op_cuda_conv_2d_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_conv_2d_t *res = new op_conv_2d_t;
    memset(res, 0, sizeof(op_t));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&res->convDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->convIn));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&res->convW));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->convBias));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&res->convOut));
    res->ws = mem_new(cuda_mem_tp);
    return res;
}

void op_cuda_conv_2d_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_NUM_OUTPUT, dtUINT32, &((op_conv_2d_t *)op)->num_output));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_H, dtUINT32, &((op_conv_2d_t *)op)->kernel_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_KERNEL_W, dtUINT32, &((op_conv_2d_t *)op)->kernel_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_H, dtUINT32, &((op_conv_2d_t *)op)->pad_h));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_PAD_W, dtUINT32, &((op_conv_2d_t *)op)->pad_w));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_H, dtUINT32, &((op_conv_2d_t *)op)->stride_h));
    CHECK(op_setting_single_get(
        op, SETTING_CONV_2D_STRIDE_W, dtUINT32, &((op_conv_2d_t *)op)->stride_w));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_HOLE_H, dtUINT32, &((op_conv_2d_t *)op)->hole_h));
    CHECK(
        op_setting_single_get(op, SETTING_CONV_2D_HOLE_W, dtUINT32, &((op_conv_2d_t *)op)->hole_w));
    CHECK(op_setting_single_get(op, SETTING_CONV_2D_GROUP, dtUINT32, &((op_conv_2d_t *)op)->group));
}

void op_cuda_conv_2d_tp_destroy(op_t *op) { (void)op; }

void op_cuda_conv_2d_tp_dealloc(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    if (NULL != conv_op->ws) {
        mem_delete(conv_op->ws);
    }
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(conv_op->convIn));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(conv_op->convW));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(conv_op->convBias));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(conv_op->convOut));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_op->convDesc));
    delete conv_op;
}

static void op_cuda_conv_2d_run(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    float alpha = 1.0;
    float beta = 0.0;

    uint32_t g;
    for (g = 0; g < conv_op->group; ++g) {
        CUDNN_CHECK(cudnnConvolutionForward(
            CUDA_WORKSPACE_CUDNNHDL(op->workspace), &alpha, conv_op->convIn,
            (float *)mem_data(op->input_tensors[0]->mem) + conv_op->bottom_offset * g,
            conv_op->convW,
            (float *)mem_data(op->input_tensors[1]->mem) + conv_op->weight_offset * g,
            conv_op->convDesc, conv_op->perfResults.algo, mem_data(conv_op->ws),
            mem_sizeof(conv_op->ws), &beta, conv_op->convOut,
            (float *)mem_data(op->output_tensors[0].mem) + conv_op->top_offset * g));
        if (op->input_size > 2) {
            CUDNN_CHECK(cudnnAddTensor(
                CUDA_WORKSPACE_CUDNNHDL(op->workspace), &alpha, conv_op->convBias,
                (float *)mem_data(op->input_tensors[2]->mem) + conv_op->bias_offset * g, &alpha,
                conv_op->convOut,
                (float *)mem_data(op->output_tensors[0].mem) + conv_op->top_offset * g));
        }
    }
}

void op_cuda_conv_2d_tp_prepare(op_t *op)
{
    op_conv_2d_t *conv_op = (op_conv_2d_t *)op;
    /* set input tensor desc */
    shape_t *shape = &op->input_tensors[0]->shape;
    conv_op->bottom_offset = shape->dim[1] / conv_op->group * shape->dim[2] * shape->dim[3];
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        conv_op->convIn, CUDNN_DATA_FLOAT, shape->dim[0], shape->dim[1] / conv_op->group,
        shape->dim[2], shape->dim[3], // shape
        shape->dim[1] * shape->dim[2] * shape->dim[3], shape->dim[2] * shape->dim[3], shape->dim[3],
        1)); // stride

    /* set filter desc */
    shape = &op->input_tensors[1]->shape;
    conv_op->weight_offset
        = (shape->dim[0] / conv_op->group) * shape->dim[1] * shape->dim[2] * shape->dim[3];
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        conv_op->convW, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, shape->dim[0] / conv_op->group,
        shape->dim[1], shape->dim[2], shape->dim[3]));

    /* set bias desc */
    if (op->input_size > 2) {
        conv_op->bias_offset = op->input_tensors[1]->shape.dim[0] / conv_op->group;
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            conv_op->convBias, CUDNN_DATA_FLOAT, 1, shape->dim[0] / conv_op->group, 1, 1,
            shape->dim[0] / conv_op->group, 1, 1, 1));
    }

    /* set output tensor desc */
    shape = &op->output_tensors[0].shape;
    conv_op->top_offset = shape->dim[1] / conv_op->group * shape->dim[2] * shape->dim[3];
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        conv_op->convOut, CUDNN_DATA_FLOAT, shape->dim[0], shape->dim[1] / conv_op->group,
        shape->dim[2], shape->dim[3], shape->dim[1] * shape->dim[2] * shape->dim[3],
        shape->dim[2] * shape->dim[3], shape->dim[3], 1));

    /* set conv desc */
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_op->convDesc, conv_op->pad_h, conv_op->pad_w, conv_op->stride_h, conv_op->stride_w,
        conv_op->hole_h, conv_op->hole_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    /* find algo */
    int returnedAlgoCount = 0;

#if CUDNN_VERSION >= 8000
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        CUDA_WORKSPACE_CUDNNHDL(op->workspace), conv_op->convIn, conv_op->convW, conv_op->convDesc,
        conv_op->convOut, 1, &returnedAlgoCount, &conv_op->perfResults));
#else
    conv_op->perfResults.algo = (cudnnConvolutionFwdAlgo_t)0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        CUDA_WORKSPACE_CUDNNHDL(op->workspace), conv_op->convIn, conv_op->convW, conv_op->convDesc,
        conv_op->convOut, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 8 * 1024 * 1024,
        &conv_op->perfResults.algo));
#endif

    /* get workspace size */
    size_t sizeBytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        CUDA_WORKSPACE_CUDNNHDL(op->workspace), conv_op->convIn, conv_op->convW, conv_op->convDesc,
        conv_op->convOut, conv_op->perfResults.algo, &sizeBytes));
    mem_alloc(conv_op->ws, sizeBytes);

    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_conv_2d_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
