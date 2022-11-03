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

#ifndef CUDA_WORKSPACE_H
#define CUDA_WORKSPACE_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "art/workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_workspace_t {
    workspace_t ws;
    char *name;
    uint32_t dev_id;
    cudaStream_t stream;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
} cuda_workspace_t;

extern const char *cublasGetErrorString(cublasStatus_t error);

#define CUDA_WORKSPACE_STREAM(ws) ((cuda_workspace_t *)(ws))->stream

#define CUDA_WORKSPACE_CUDNNHDL(ws) ((cuda_workspace_t *)(ws))->cudnnHandle

#define CUDA_WORKSPACE_CUBLASHDL(ws) ((cuda_workspace_t *)(ws))->cublasHandle

#define CUDA_KERNEL_LOOP(i, n)                                           \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (int)(n); \
         i += blockDim.x * gridDim.x)

#ifdef __cplusplus
}
#endif

#endif // CUDA_WORKSPACE_H
