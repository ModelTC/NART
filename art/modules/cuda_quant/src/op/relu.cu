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

#include "art/cuda_quant/cuda_quant_op_settings.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

// #include "art/cuda_quant/cuda_quant_helper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../cuda_quant_workspace.h"

typedef struct {
    op_t o;
    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;
    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_relu_t;

#ifdef __cplusplus
extern "C" {
#endif

op_relu_t *op_cuda_quant_relu_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_relu_t *res = (op_relu_t *)malloc(sizeof(op_relu_t));
    memset(res, 0, sizeof(op_relu_t));
    return res;
}

void op_cuda_quant_relu_tp_config(op_t *op)
{
    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;

    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_relu_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_relu_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IBITS, dtUINT8, &len_bits, &((op_relu_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_relu_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_relu_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OBITS, dtUINT8, &len_bits, &((op_relu_t *)op)->obits));
}

void op_cuda_quant_relu_tp_destroy(op_t *op) { (void)op; }

void op_cuda_quant_relu_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void relu_int8_int4_ker(const int8_t *input, int8_t *output, const size_t len)
{
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < len) {
        output[tidx] = input[tidx] < 0 ? 0 : input[tidx];
    }
}
//__global__ void relu_uint8_uint4_ker(const uint8_t * input, uint8_t* output, const size_t len,
// const uint8_t i_zp, const uint8_t o_zp, const float ialpha, const float oalpha)
//{
//    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tidx < len){
//       //	output[tidx]
//        uint8_t tmp = input[tidx] < i_zp ? i_zp : input[tidx];
//
//    }
//    // uncomplete
//}

static void op_cuda_quant_relu_sym_run(op_t *op)
{
    // const op_relu_t * relu_op = (op_relu_t*) op;
    size_t count = shape_count(&op->output_tensors[0].shape);
    // size_t i;
    const int8_t *input_0 = (const int8_t *)mem_data(op->input_tensors[0]->mem);
    int8_t *output_0 = (int8_t *)mem_data(op->output_tensors[0].mem);
    // relu_int8_int4_ker<<<(count + 1023) / 1024, 1024, 0
    // ,CUDA_WORKSPACE_STREAM(op->workspace)>>>(input_0, output_0, count);
}

// static void op_cuda_quant_relu_asym_run(op_t* op)
//{
//     const op_relu_t* relu_op = (op_relu_t*) op;
//     size_t count = shape_count(&op->output_tensors[0].shape);
//     size_t i;
//     const uint8_t* input_0 = mem_data(op->input_tensors[0]->mem);
//     uint8_t* output_0 = mem_data(op->output_tensors[0].mem);
//
//     float ialpha = relu_op->ialpha[0];
//     uint8_t izero_point = relu_op->izero_point[0];
//
//     float oalpha = relu_op->oalpha[0];
//     uint8_t ozero_point = relu_op->ozero_point[0];
//#ifdef USE_FIXED_POINT_ONLY
//     /*ialpha / oalpha * to  result_mult / (1 >> resultshift) */
//     uint32_t result_mult;
//     int8_t result_shift;
//
//     result_mult = 0;
//     result_shift = 0;
//     get_quant_scale(ialpha / oalpha, &result_mult, &result_shift, 24);
//#endif
//     /* oalpha(oq - ozero_point) = of = ialpha (iq - izero_point)*/
//     /* oq = ialpha / oalpha (iq - izero_point) + ozero_point */
//     uint8_t bit = relu_op->obits[0];
//     relu_uint8_uint4_ker<<<(count + 1023) / 1024, 1024, 0 , CUDA_WORKSPACE_STREAM(ws)>>>(input_0,
//     output_0, count, ) for (i = 0; i < count; ++i) {
//         /*do relu*/
//         output_0[i] = input_0[i] >= izero_point ? input_0[i] : izero_point;
//         /*to output scale*/
//#ifdef USE_FIXED_POINT_ONLY
//         output_0[i] = saturate_int_by_bits(rshift_rn((int32_t)((int32_t) output_0[i] -
//         izero_point) * result_mult, result_shift) + ozero_point, bit);
//#else
//         output_0[i] = saturate_int_by_bits((int32_t)((int16_t) output_0[i] - izero_point) *
//         ialpha / oalpha + ozero_point, bit);
//#endif
//     }
// }

void op_cuda_quant_relu_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtINT8:
        // case dtINT4:
        op->run_func = op_cuda_quant_relu_sym_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
