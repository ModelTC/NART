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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "art/cuda/cuda_mem.h"
#include "art/cuda/cuda_ops.h"
#include "art/cuda/cuda_ws_settings.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"

#include "./cuda_workspace.h"

static int cuda_ws_num = 0;

const char *cublasGetErrorString(cublasStatus_t error)
{
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "Unknown cublas status";
}

static const char *ws_cuda_name(const cuda_workspace_t *ws) { return ws->name; }

static const mem_tp *ws_cuda_mem_tp(cuda_workspace_t *ws)
{
    (void)ws;
    return cuda_mem_tp;
}

static void ws_cuda_delete(cuda_workspace_t *ws)
{
    CUDA_CHECK(cudaSetDevice(ws->dev_id));
    // CUDA_CHECK(cudaStreamDestroy(ws->stream));
    CUDNN_CHECK(cudnnDestroy(ws->cudnnHandle));
    CUBLAS_CHECK(cublasDestroy(ws->cublasHandle));
    free(ws->name);
    free(ws);
}

static workspace_t *ws_cuda_new(const setting_entry_t *);

static op_tp_entry_t op_entries[] = {
    OP_ENTRY(op_add_tp, cuda, add),
    OP_ENTRY(op_abs_tp, cuda, abs),
    OP_ENTRY(op_conv_nd_tp, cuda, conv_nd),
    OP_ENTRY(op_conv_2d_tp, cuda, conv_2d),
    OP_ENTRY(op_deconv_2d_tp, cuda, deconv_2d),
    OP_ENTRY(op_deform_conv_2d_tp, cuda, deform_conv_2d),
    OP_ENTRY(op_relu_tp, cuda, relu),
    OP_ENTRY(op_relu6_tp, cuda, relu6),
    OP_ENTRY(op_prelu_tp, cuda, prelu),
    OP_ENTRY(op_pool_tp, cuda, pool),
    OP_ENTRY(op_ip_tp, cuda, ip),
    OP_ENTRY(op_eltwise_tp, cuda, eltwise),
    OP_ENTRY(op_softmax_tp, cuda, softmax),
    OP_ENTRY(op_concat_tp, cuda, concat),
    OP_ENTRY(op_interp_tp, cuda, interp),
    OP_ENTRY(op_sigmoid_tp, cuda, sigmoid),
    OP_ENTRY(op_bn_tp, cuda, bn),
    OP_ENTRY(op_batchnorm_tp, cuda, batchnorm),
    OP_ENTRY(op_roialignpooling_tp, cuda, roialignpooling),
    OP_ENTRY(op_roipooling_tp, cuda, roipooling),
    OP_ENTRY(op_psroipooling_tp, cuda, psroipooling),
    OP_ENTRY(op_psroialignpooling_tp, cuda, psroialignpooling),
    OP_ENTRY(op_pad_tp, cuda, pad),
    OP_ENTRY(op_mul_tp, cuda, mul),
    OP_ENTRY(op_podroialignpooling_tp, cuda, podroialignpooling),
    OP_ENTRY(op_correlation_tp, cuda, correlation),
    OP_ENTRY(op_quant_dequant_tp, cuda, quant_dequant),
    OP_ENTRY(op_sub_tp, cuda, sub),
    OP_ENTRY(op_div_tp, cuda, div),
    OP_ENTRY(op_pow_tp, cuda, pow),
    OP_ENTRY(op_log_tp, cuda, log),
    OP_ENTRY(op_exp_tp, cuda, exp),
    OP_ENTRY(op_tanh_tp, cuda, tanh),
    OP_ENTRY(op_instancenorm_tp, cuda, instancenorm),
    OP_ENTRY(op_matmul_tp, cuda, matmul),
    OP_ENTRY(op_podroialignpooling_tp, cuda, podroialignpooling),
    OP_ENTRY(op_correlation_tp, cuda, correlation),
    OP_ENTRY(op_reducemin_tp, cuda, reducemin),
    OP_ENTRY(op_reducemax_tp, cuda, reducemax),
    OP_ENTRY(op_reducemean_tp, cuda, reducemean),
    OP_ENTRY(op_reduceprod_tp, cuda, reduceprod),
    OP_ENTRY(op_reducesum_tp, cuda, reducesum),
    OP_ENTRY(op_reducel2_tp, cuda, reducel2),
    OP_ENTRY(op_reshape_tp, cuda, reshape),
    OP_ENTRY(op_transpose_tp, cuda, transpose),
    OP_ENTRY(op_lpnormalization_tp, cuda, lpnormalization),
    OP_ENTRY(op_slice_tp, cuda, slice),
    OP_ENTRY(op_gridsample_tp, cuda, gridsample),
    OP_ENTRY(op_unfold_tp, cuda, unfold),
    OP_ENTRY(op_subpixel_tp, cuda, subpixel),
    OP_ENTRY(op_lstm_tp, cuda, lstm),
    OP_ENTRY(op_hardsigmoid_tp, cuda, hardsigmoid),
    OP_ENTRY(op_heatmap2coord_tp, cuda, heatmap2coord),
    OP_ENTRY(op_clip_tp, cuda, clip),
    OP_ENTRY(op_cast_tp, cuda, cast),
    OP_ENTRY(op_erf_tp, cuda, erf),
    OP_ENTRY(op_psroimaskpooling_tp, cuda, psroimaskpooling),
    OP_ENTRY(op_hswish_tp, cuda, hswish),
    OP_ENTRY(op_scatternd_tp, cuda, scatternd),
    OP_ENTRY(op_elu_tp, cuda, elu),
    OP_ENTRY(op_clip_cast_tp, cuda, clip_cast),
    OP_ENTRY(op_add_div_clip_cast_tp, cuda, add_div_clip_cast),
    { NULL },
};

const module_t DLL_PUBLIC cuda_module_tp = {
    .op_group = OP_GROUP_CODE_CUDA,
    .name_func = (ws_name_func)ws_cuda_name,
    .memtype_func = (ws_memtp_func)ws_cuda_mem_tp,
    .new_func = (ws_new_func)ws_cuda_new,
    .delete_func = (ws_delete_func)ws_cuda_delete,
    .op_tp_entry = op_entries,
};

workspace_t *ws_cuda_new(const setting_entry_t *settings)
{
    uint32_t dev_id = -1;
    if (settings != NULL) {
        const setting_entry_t *iter = settings;
        for (; iter->item != 0; ++iter) {
            switch (iter->item) {
            case SETTING_CUDA_WORKSPACE_DEVID:
                CHECK_EQ(iter->dtype, dtUINT32);
                CHECK_EQ(iter->tp, ENUM_SETTING_VALUE_SINGLE);
                dev_id = iter->v.single.value.u32;
                break;
            default:
                CHECK(true);
            }
        }
    }

    char tmp[20];
    cuda_workspace_t *res = (cuda_workspace_t *)malloc(sizeof(cuda_workspace_t));
    memset(res, 0, sizeof(*res));
    sprintf(tmp, "cuda:%d", cuda_ws_num++);
    res->name = malloc(strlen(tmp) + 1);
    strcpy(res->name, tmp);
    res->ws.module_type = &cuda_module_tp;

    if (dev_id == -1) {
        CUDA_CHECK(cudaGetDevice(&dev_id));
    }
    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaStream_t stream;
    stream = 0;
    // CUDA_CHECK(cudaStreamCreate(&stream));
    res->stream = stream;

    cudnnHandle_t cudnnHandle;
    CUDNN_CHECK(cudnnCreate(&cudnnHandle));
    CUDNN_CHECK(cudnnSetStream(cudnnHandle, stream));
    res->cudnnHandle = cudnnHandle;

    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    res->cublasHandle = cublasHandle;
    res->dev_id = dev_id;
    return (workspace_t *)res;
}
