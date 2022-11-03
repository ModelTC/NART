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

#ifndef CUDA_MEM_H
#define CUDA_MEM_H

#include "art/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const mem_tp *const trt_mem_tp;

// int trt_mem_set_gpu(mem_t* mem, int gpuid);
// int trt_mem_get_gpu(mem_t* mem);

#ifdef __cplusplus
}
#endif

#endif // CUDA_MEM_H