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

#ifndef TENSORRT_WORKSPACE_H
#define TENSORRT_WORKSPACE_H

#include <cuda_runtime.h>

#include "art/workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensorrt_workspace_t {
    workspace_t ws;
    char *name;
    uint32_t dev_id;
    cudaStream_t stream;
} tensorrt_workspace_t;

#define TENSORRT_WORKSPACE_STREAM(ws) ((tensorrt_workspace_t *)(ws))->stream

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_WORKSPACE_H
