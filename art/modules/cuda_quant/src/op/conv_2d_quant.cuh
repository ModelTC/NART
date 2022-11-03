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

#ifndef CONV_2D_QUANT_CUH
#define CONV_2D_QUANT_CUH

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_quant_workspace.h"
#include "art/cuda_quant/cuda_quant_mem.h"
#include "art/cuda_quant/cuda_quant_op_settings.h"

#include <functional>

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
    uint32_t group;
    uint32_t relu_flag;

    float *walpha;
    uint8_t *wzero_point;
    uint8_t *wbits;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;

    std::function<void(
        char *, char *, char *, int, int, int, int, int, int, int, int, int, int, int, int, float *,
        bool, bool, uint8_t, float *, void *, workspace_t *)>
        conv_best_func;

    std::function<void(
        char *, char *, char *, int *, int, int, int, int, int, int, int, int, int, int, int, int,
        float *, bool, bool, uint8_t, float *, void *, workspace_t *)>
        conv_best_func_split;

    size_t M_best_pad;
    size_t N_best_pad;
    size_t K_best_pad;

    bool use_conv_split = false;

    mem_t *ws_pre;
    mem_t *ws_alphas;
    mem_t *ws_inttmp;

} op_conv_2d_t;

void op_cuda_quant_conv_2d_i8xi8_run(op_t *op);
void op_cuda_quant_conv_2d_i4xi4_run(op_t *op);
#ifdef __cplusplus
}
#endif

#endif
