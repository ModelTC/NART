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

#include "art/cuda_quant/cuda_quant_mem.h"
#include "art/cuda_quant/cuda_quant_ops.h"
#include "art/cuda_quant/cuda_quant_ws_settings.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"

#include "./cuda_quant_workspace.h"

static int cuda_quant_ws_num = 0;

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

static const char *ws_cuda_quant_name(const cuda_quant_workspace_t *ws) { return ws->name; }

static const mem_tp *ws_cuda_quant_mem_tp(cuda_quant_workspace_t *ws)
{
    (void)ws;
    return cuda_mem_tp;
}

static void ws_cuda_quant_delete(cuda_quant_workspace_t *ws)
{
    CUDA_CHECK(cudaSetDevice(ws->dev_id));
    // CUDA_CHECK(cudaStreamDestroy(ws->stream));
    CUDNN_CHECK(cudnnDestroy(ws->cudnnHandle));
    CUBLAS_CHECK(cublasDestroy(ws->cublasHandle));
    free(ws->name);
    free(ws);
}

static workspace_t *ws_cuda_quant_new(const setting_entry_t *);

static op_tp_entry_t op_entries[] = {
    OP_ENTRY(op_quantize_tp, cuda_quant, quantize),
    OP_ENTRY(op_dequantize_tp, cuda_quant, dequantize),
    OP_ENTRY(op_conv_2d_tp, cuda_quant, conv_2d),
    OP_ENTRY(op_relu_tp, cuda_quant, relu),
    OP_ENTRY(op_eltwise_tp, cuda_quant, eltwise),
    OP_ENTRY(op_pool_tp, cuda_quant, pool),
    { NULL },
};

const module_t DLL_PUBLIC cuda_quant_module_tp = {
    .op_group = OP_GROUP_CODE_CUDA_QUANT,
    .name_func = (ws_name_func)ws_cuda_quant_name,
    .memtype_func = (ws_memtp_func)ws_cuda_quant_mem_tp,
    .new_func = (ws_new_func)ws_cuda_quant_new,
    .delete_func = (ws_delete_func)ws_cuda_quant_delete,
    .op_tp_entry = op_entries,
};

workspace_t *ws_cuda_quant_new(const setting_entry_t *settings)
{
    uint32_t dev_id = 0;
    if (settings != NULL) {
        const setting_entry_t *iter = settings;
        for (; iter->item != 0; ++iter) {
            switch (iter->item) {
            case SETTING_CUDA_QUANT_WORKSPACE_DEVID:
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
    cuda_quant_workspace_t *res = (cuda_quant_workspace_t *)malloc(sizeof(cuda_quant_workspace_t));
    memset(res, 0, sizeof(*res));
    sprintf(tmp, "cuda quant:%d", cuda_quant_ws_num++);
    res->name = malloc(strlen(tmp) + 1);
    strcpy(res->name, tmp);
    res->ws.module_type = &cuda_quant_module_tp;

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
