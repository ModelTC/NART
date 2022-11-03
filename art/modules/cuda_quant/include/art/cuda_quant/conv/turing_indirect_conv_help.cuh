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

#ifndef TURING_INDIRECT_CONV_HELP_CUH
#define TURING_INDIRECT_CONV_HELP_CUH

#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>

#define WARP_SIZE 32

using namespace nvcuda;

typedef struct preComputeMatixA {
    int32_t image_offset;
    uint16_t out_height;
    uint16_t out_width;
    int16_t im_h;
    int16_t im_w;
    uint8_t batch;
    uint8_t fill_zero_flag;
    uint8_t aligned[2];
} preComputeMatixA __attribute__((__aligned__(16)));

template <typename T> struct SharedMemory_t {
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <> struct SharedMemory_t<int> {
    __device__ int *getPointer()
    {
        extern __shared__ int s_int[];
        return s_int;
    }
};

static __device__ int2 mma8816(int2 C, int A, int B)
{
    int2 D;
#ifdef WIN32
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(D.x), "=r"(D.y)
        : "r"(A), "r"(B), "r"(C.x), "r"(C.y));
#else
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(D.x), "=r"(D.y)
        : "r"(A), "r"(B), "r"(C.x), "r"(C.y));
#endif
    return D; // D = C + A * B
}

static __device__ __inline__ int2 mma8832(int2 C, int A, int B)
{
    int2 D;
#ifdef WIN32
    asm("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(D.x), "=r"(D.y)
        : "r"(A), "r"(B), "r"(C.x), "r"(C.y));
#else
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32.satfinite {%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(D.x), "=r"(D.y)
        : "r"(A), "r"(B), "r"(C.x), "r"(C.y));
#endif

    return D; // D = C + A * B
}

__device__ int4 zero_arr[10] = { 0 };

template <bool overlap = true> __global__ void init_zero(int *input)
{
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    int4 zero = { 0 };
    ((int4 *)input)[threadId] = zero;
}

template <bool overlap = true>
__global__ void quantize(
    int *input, char *output, float *bias, float *alpha, int channel, bool bias_flag,
    bool relu_flag, int out_bits)
{

    int channelId = threadIdx.x;
    int oh_ow_index = blockIdx.x;

    int mval = (1 << (out_bits - 1)) - 1;
    int lower = relu_flag ? 0 : -mval - 1;

    if (out_bits == 4) {
        float2 bias_t = { 0 };
        if (bias_flag)
            bias_t = __ldg(((float2 *)bias) + channelId);
        int2 input_t = __ldg(((int2 *)(input + oh_ow_index * channel)) + channelId);
        float2 oalpha = __ldg((float2 *)(alpha) + channelId);
        int32_t quant_res_reg0 = (int)rintf((float)(input_t.x + bias_t.x) * oalpha.x);
        int32_t quant_res_reg1 = (int)rintf((float)(input_t.y + bias_t.y) * oalpha.y);

        int8_t quant_res_regarr[2];

        quant_res_reg0 = quant_res_reg0 < lower ? lower : quant_res_reg0;
        quant_res_reg0 = quant_res_reg0 > mval ? mval : quant_res_reg0;
        quant_res_reg1 = quant_res_reg1 < lower ? lower : quant_res_reg1;
        quant_res_reg1 = quant_res_reg1 > mval ? mval : quant_res_reg1;

        quant_res_regarr[0] = quant_res_reg0;
        quant_res_regarr[1] = quant_res_reg1;

        int8_t quant_res_4bit_0 = (quant_res_regarr[0] & 0xF) | (quant_res_regarr[1] << 4);
        output[oh_ow_index * channel / 2 + channelId] = quant_res_4bit_0;

    } else {
        float bias_t = bias_flag ? __ldg(bias + channelId) : 0;
        int input_t = __ldg(input + oh_ow_index * channel + channelId);
        float oalpha = __ldg(alpha + channelId);

        int32_t quant_res_reg0 = (int)rintf((float)(input_t + bias_t) * oalpha);

        int8_t quant_res_regarr;

        quant_res_reg0 = quant_res_reg0 < lower ? lower : quant_res_reg0;
        quant_res_reg0 = quant_res_reg0 > mval ? mval : quant_res_reg0;
        quant_res_regarr = quant_res_reg0;

        output[oh_ow_index * channel + channelId] = quant_res_regarr;
    }
}

#endif
