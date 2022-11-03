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

//#include "art/cuda/cuda_quant_helper.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../cuda_quant_workspace.h"

typedef struct {
    op_t o;
    uint32_t operation;
    float *coeff;
    uint32_t relu_flag;
    // uint32_t input_bits;
    // uint32_t output_bits;

    float *ialpha;
    uint8_t *izero_point;
    uint8_t *ibits;

    float *oalpha;
    uint8_t *ozero_point;
    uint8_t *obits;
} op_eltwise_t;

#ifdef __cplusplus
extern "C" {
#endif

op_eltwise_t *op_cuda_quant_eltwise_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_eltwise_t *res = (op_eltwise_t *)malloc(sizeof(op_eltwise_t));
    memset(res, 0, sizeof(op_eltwise_t));
    return res;
}

void op_cuda_quant_eltwise_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_ELTWISE_OPERATION, dtUINT32, &((op_eltwise_t *)op)->operation));
    //    if (!op_setting_if_set(op, SETTING_ELTWISE_COEFF)) {
    //        float *coeff = (float *)malloc(sizeof(float) * op->input_size);
    //        int i;
    //        for (i = 0; i < op->input_size; ++i) {
    //            coeff[i] = 1.0;
    //        }
    //        CHECK(op_setting_array_set(op, SETTING_ELTWISE_COEFF, dtFLOAT32, op->input_size,
    //        coeff)); free(coeff);
    //    }
    size_t len;
    CHECK(op_setting_array_get(
        op, SETTING_ELTWISE_COEFF, dtFLOAT32, &len, &((op_eltwise_t *)op)->coeff));
    CHECK(op_setting_single_get(
        op, SETTING_ELTWISE_RELU_FLAG, dtUINT32, &((op_eltwise_t *)op)->relu_flag));
    // CHECK(op_setting_single_get(op, SETTING_ELTWISE_INPUT_BITS, dtUINT32,
    // &((op_eltwise_t*)op)->input_bits)); CHECK(op_setting_single_get(op,
    // SETTING_ELTWISE_OUTPUT_BITS, dtUINT32, &((op_eltwise_t*)op)->output_bits));
    CHECK_EQ(op->input_size, len);

    size_t len_alpha;
    size_t len_zero_point;
    size_t len_bits;
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IALPHA, dtFLOAT32, &len_alpha, &((op_eltwise_t *)op)->ialpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IZERO_POINT, dtUINT8, &len_zero_point,
        &((op_eltwise_t *)op)->izero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_IBITS, dtUINT8, &len_bits, &((op_eltwise_t *)op)->ibits));

    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OALPHA, dtFLOAT32, &len_alpha, &((op_eltwise_t *)op)->oalpha));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OZERO_POINT, dtUINT8, &len_zero_point,
        &((op_eltwise_t *)op)->ozero_point));
    CHECK(op_setting_array_get(
        op, SETTING_CUDA_QUANT_OBITS, dtUINT8, &len_bits, &((op_eltwise_t *)op)->obits));
}

void op_cuda_quant_eltwise_tp_destroy(op_t *op)
{
    //(void*)op;
}

void op_cuda_quant_eltwise_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void inline __device__ __bfe_4b(int &A, int4 &A0, int4 &A1)
{
    int *A0t = (int *)(&A0);
    int *A1t = (int *)(&A1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        asm volatile("bfe.s32 %0, %1, %2, %3;" : "=r"(A0t[i]) : "r"(A), "r"((i * 4)), "r"(4));
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        asm volatile("bfe.s32 %0, %1, %2, %3;" : "=r"(A1t[i]) : "r"(A), "r"((i * 4) + 16), "r"(4));
    }
}

void inline __device__ __bfe_8b(int &A, int4 &A0)
{
    int *A0t = (int *)(&A0);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        asm volatile("bfe.s32 %0, %1, %2, %3;" : "=r"(A0t[i]) : "r"(A), "r"((i * 8)), "r"(8));
    }
}

static inline int __device__ clip(int v, int max_v, int min_v)
{
    int ret;
    ret = v > max_v ? max_v : v;
    ret = ret < min_v ? min_v : ret;
    return ret;
}

// void inline __device__ __cvt_pack()  sm75

int inline __device__ eltwise_int8(
    float A_ialpha, float B_ialpha, int &A, int &B, bool relu_flag, const size_t out_bits = 8)
{
    int C = 0;
    int4 A_vec, B_vec;
    __bfe_8b(A, A_vec);
    __bfe_8b(B, B_vec);

    float vec_0_fp32 = float(A_vec.x) * A_ialpha + float(B_vec.x) * B_ialpha;
    float vec_1_fp32 = float(A_vec.y) * A_ialpha + float(B_vec.y) * B_ialpha;
    float vec_2_fp32 = float(A_vec.z) * A_ialpha + float(B_vec.z) * B_ialpha;
    float vec_3_fp32 = float(A_vec.w) * A_ialpha + float(B_vec.w) * B_ialpha;

    int max_val = (1 << (out_bits - 1)) - 1;
    int min_val = relu_flag ? 0 : -max_val - 1;

    int vec_0_int32, vec_1_int32, vec_2_int32, vec_3_int32;

    vec_0_int32 = clip(int(rintf(vec_0_fp32)), max_val, min_val);
    vec_1_int32 = clip(int(rintf(vec_1_fp32)), max_val, min_val);
    vec_2_int32 = clip(int(rintf(vec_2_fp32)), max_val, min_val);
    vec_3_int32 = clip(int(rintf(vec_3_fp32)), max_val, min_val);

    C |= int8_t(vec_0_int32) & 0xFF;
    C |= (int8_t(vec_1_int32) << 8) & 0xFF00;
    C |= (int8_t(vec_2_int32) << 16) & 0xFF0000;
    C |= (int8_t(vec_3_int32) << 24) & 0xFF000000;
    return C;
}

int inline __device__ eltwise_int4(
    float A_ialpha, float B_ialpha, int &A, int &B, bool relu_flag, const size_t out_bits = 4)
{
    int C = 0;
    int4 A_vec[2], B_vec[2];
    __bfe_4b(A, A_vec[0], A_vec[1]);
    __bfe_4b(B, B_vec[0], B_vec[1]);

    float4 vec_fp32[2];
    vec_fp32[0].x = float(A_vec[0].x) * A_ialpha + float(B_vec[0].x) * B_ialpha;
    vec_fp32[0].y = float(A_vec[0].y) * A_ialpha + float(B_vec[0].y) * B_ialpha;
    vec_fp32[0].z = float(A_vec[0].z) * A_ialpha + float(B_vec[0].z) * B_ialpha;
    vec_fp32[0].w = float(A_vec[0].w) * A_ialpha + float(B_vec[0].w) * B_ialpha;

    vec_fp32[1].x = float(A_vec[1].x) * A_ialpha + float(B_vec[1].x) * B_ialpha;
    vec_fp32[1].y = float(A_vec[1].y) * A_ialpha + float(B_vec[1].y) * B_ialpha;
    vec_fp32[1].z = float(A_vec[1].z) * A_ialpha + float(B_vec[1].z) * B_ialpha;
    vec_fp32[1].w = float(A_vec[1].w) * A_ialpha + float(B_vec[1].w) * B_ialpha;

    int max_val = (1 << (out_bits - 1)) - 1;
    int min_val = relu_flag ? 0 : -max_val - 1;

    int vec_0_int32[4], vec_1_int32[4];

    vec_0_int32[0] = clip(int(rintf(vec_fp32[0].x)), max_val, min_val);
    vec_0_int32[1] = clip(int(rintf(vec_fp32[0].y)), max_val, min_val);
    vec_0_int32[2] = clip(int(rintf(vec_fp32[0].z)), max_val, min_val);
    vec_0_int32[3] = clip(int(rintf(vec_fp32[0].w)), max_val, min_val);

    vec_1_int32[0] = clip(int(rintf(vec_fp32[1].x)), max_val, min_val);
    vec_1_int32[1] = clip(int(rintf(vec_fp32[1].y)), max_val, min_val);
    vec_1_int32[2] = clip(int(rintf(vec_fp32[1].z)), max_val, min_val);
    vec_1_int32[3] = clip(int(rintf(vec_fp32[1].w)), max_val, min_val);

    C |= int8_t(vec_0_int32[0]) & 0xF;
    C |= (int8_t(vec_0_int32[1]) << 4) & 0xF0;
    C |= (int8_t(vec_0_int32[2]) << 8) & 0xF00;
    C |= (int8_t(vec_0_int32[3]) << 12) & 0xF000;

    C |= (int8_t(vec_1_int32[0]) << 16) & 0xF0000;
    C |= (int8_t(vec_1_int32[1]) << 20) & 0xF00000;
    C |= (int8_t(vec_1_int32[2]) << 24) & 0xF000000;
    C |= (int8_t(vec_1_int32[3]) << 28) & 0xF0000000;
    return C;
}

__global__ void eltwise_int8_sum_ker(
    const int8_t *input_1, const int8_t *input_2, int8_t *output, const size_t len,
    const float alpha_0, const float alpha_1, const bool relu_flag, const size_t output_bits)
{
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx * 16 < len) {
        //	output[tidx]
        int4 A_vec = __ldg(((int4 *)input_1) + tidx);
        int4 B_vec = __ldg(((int4 *)input_2) + tidx);

        float A_ialpha = alpha_0;
        float B_ialpha = alpha_1;

        int4 output_res;
        output_res.x = eltwise_int8(A_ialpha, B_ialpha, A_vec.x, B_vec.x, relu_flag);
        output_res.y = eltwise_int8(A_ialpha, B_ialpha, A_vec.y, B_vec.y, relu_flag);
        output_res.z = eltwise_int8(A_ialpha, B_ialpha, A_vec.z, B_vec.z, relu_flag);
        output_res.w = eltwise_int8(A_ialpha, B_ialpha, A_vec.w, B_vec.w, relu_flag);

        ((int4 *)output)[tidx] = output_res;
    }
}

__global__ void eltwise_int4_sum_ker(
    const int8_t *input_1, const int8_t *input_2, int8_t *output, const size_t len,
    const float alpha_0, const float alpha_1, bool relu_flag, const size_t output_bits)
{
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx * 32 < len) {

        int4 A_vec = __ldg(((int4 *)input_1) + tidx);
        int4 B_vec = __ldg(((int4 *)input_2) + tidx);

        float A_ialpha = alpha_0;
        float B_ialpha = alpha_1;

        int4 output_res;
        output_res.x = eltwise_int4(A_ialpha, B_ialpha, A_vec.x, B_vec.x, relu_flag);
        output_res.y = eltwise_int4(A_ialpha, B_ialpha, A_vec.y, B_vec.y, relu_flag);
        output_res.z = eltwise_int4(A_ialpha, B_ialpha, A_vec.z, B_vec.z, relu_flag);
        output_res.w = eltwise_int4(A_ialpha, B_ialpha, A_vec.w, B_vec.w, relu_flag);

        ((int4 *)output)[tidx] = output_res;
    }
}

static void op_cuda_quant_eltwise_sym_run(op_t *op)
{
    const op_eltwise_t *eltwise_op = (op_eltwise_t *)op;
    size_t count = shape_count(&op->output_tensors[0].shape);

    int in_bit = eltwise_op->ibits[0];

    size_t tnum = in_bit == 8 ? count / 16 : count / 32;
    const int8_t *input_0 = (const int8_t *)mem_data(op->input_tensors[0]->mem);
    const int8_t *input_1 = (const int8_t *)mem_data(op->input_tensors[1]->mem);
    int8_t *output_0 = (int8_t *)mem_data(op->output_tensors[0].mem);

    float norm_alpha_0 = eltwise_op->ialpha[0] / eltwise_op->oalpha[0];
    float norm_alpha_1 = eltwise_op->ialpha[1] / eltwise_op->oalpha[0];

    bool relu_flag = eltwise_op->relu_flag;
    if (in_bit == 8)
        eltwise_int8_sum_ker<<<(tnum + 511) / 512, 512, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            input_0, input_1, output_0, count, norm_alpha_0, norm_alpha_1, relu_flag, 8);
    else
        eltwise_int4_sum_ker<<<(tnum + 511) / 512, 512, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            input_0, input_1, output_0, count, norm_alpha_0, norm_alpha_1, relu_flag, 4);
}

void op_cuda_quant_eltwise_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_quant_eltwise_sym_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
