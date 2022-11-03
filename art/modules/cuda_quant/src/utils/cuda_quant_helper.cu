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

#include <assert.h>
#include <stdio.h>

#include "art/log.h"

#include "cuda_quant_helper.cuh"

__global__ void round_clip_ker(const float *x, int8_t *o, size_t num, int l, int h, float alpha)
{
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        float t = x[tidx];
        int t_int = rintf(
            t * alpha); // cuda document recommend to use this instead of roudf(), see  appendix E
        o[tidx] = t_int > h ? h : (t_int < l ? l : t_int);
    }
}

void round_clip_cu(
    const float *input, int8_t *output, size_t num, int l, int h, float alpha, workspace_t *ws)
{
    round_clip_ker<<<(num + 511) / 512, 512, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, l, h, alpha);
}

__global__ void saturate_float_by_bits_ker(
    const float *x, uint8_t *o, uint8_t bits, size_t num, float alpha, float beta)
{
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        float t = (x[tidx] - beta) / alpha;
        o[tidx] = t > ((1 << bits) - 1) ? ((1 << bits) - 1) : (t < 0 ? 0 : (uint8_t)(t + 0.5));
    }
}
void saturate_float_by_bits_cu(
    const float *input, uint8_t *output, uint8_t bits, size_t num, float alpha, float beta,
    workspace_t *ws)
{
    saturate_float_by_bits_ker<<<(num + 1023) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, bits, num, alpha, beta);
}

__global__ void
dequantize_u8_ker(const uint8_t *input, float *output, size_t num, uint8_t zero_point, float alpha)
{
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        output[tidx] = ((int16_t)input[tidx] - zero_point) * alpha;
    }
}
void dequantize_u8_cu(
    const uint8_t *input, float *output, size_t num, uint8_t zero_point, float alpha,
    workspace_t *ws)
{
    dequantize_u8_ker<<<(num + 1023) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, zero_point, alpha);
}

__global__ void dequantize_i8_ker(const int8_t *input, float *output, size_t num, float alpha)
{
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        output[tidx] = input[tidx] * alpha;
    }
}
void dequantize_i8_cu(const int8_t *input, float *output, size_t num, float alpha, workspace_t *ws)
{
    dequantize_i8_ker<<<(num + 1023) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, alpha);
}

__global__ void dequantize_i4_ker(const int8_t *input, float *output, size_t num, float alpha)
{
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx * 2 < num) {
        output[tidx * 2 + 1] = ((float)(input[tidx] >> 4)) * alpha;
        int8_t tmp = input[tidx] << 4;
        output[tidx * 2] = ((float)(tmp >> 4)) * alpha;
    }
}
void dequantize_i4_cu(const int8_t *input, float *output, size_t num, float alpha, workspace_t *ws)
{
    dequantize_i4_ker<<<(num + 1023) / 1024, 512, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, alpha);
}

__device__ int clip(int x, int l, int h) { return x > h ? h : (x < l ? l : x); }

__device__ int64_t rshift_rn(int64_t x, uint8_t bits)
{
    return ((x >> (bits - 1)) + (((x >> (bits - 1)) & 1) << 1)) >> 1;
}

__global__ void saturate_int_by_bits_ker(
    int32_t *input, uint8_t *output, size_t num, uint32_t res_mult, uint32_t res_shift,
    uint8_t bits)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        int64_t res = rshift_rn((int64_t)input[tidx] * (int64_t)res_mult, res_shift);
        output[tidx] = res > ((1 << bits) - 1) ? ((1 << bits) - 1) : (res < 0 ? 0 : (uint8_t)(res));
    }
}

void saturate_int_by_bits_cu(
    int32_t *input, uint8_t *output, size_t num, uint32_t res_mult, uint32_t res_shift,
    uint8_t bits, workspace_t *ws)
{
    saturate_int_by_bits_ker<<<(num + 1023) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, res_mult, res_shift, bits);
}

__global__ void ssaturate_int_by_bits_ker(
    int32_t *input, int8_t *output, size_t num, uint32_t res_mult, uint32_t res_shift, uint8_t bits)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        int64_t res = rshift_rn((int64_t)input[tidx] * (int64_t)res_mult, res_shift);
        int mval = (1 << bits) - 1;
        output[tidx] = clip(res, -mval, mval);
    }
}

void ssaturate_int_by_bits_cu(
    int32_t *input, int8_t *output, size_t num, uint32_t res_mult, uint32_t res_shift, uint8_t bits,
    workspace_t *ws)
{
    ssaturate_int_by_bits_ker<<<(num + 511) / 512, 512, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, res_mult, res_shift, bits);
}

__global__ void ssaturate_int_by_bits_ker_temp(
    int32_t *input, int8_t *output, size_t num, size_t row_pad, size_t row_actual, size_t col_pad,
    size_t col_actual, uint32_t res_mult, uint32_t res_shift, uint8_t bits)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num) {
        int tidx_row = tidx / col_pad;
        int tidx_col = tidx % col_pad;

        if (tidx_row < row_actual) {
            if (tidx_col < col_actual) {
                int64_t res = rshift_rn(
                    (int64_t)input[tidx_row * col_pad + tidx_col] * (int64_t)res_mult, res_shift);
                int mval = (1 << bits) - 1;
                output[tidx_row * col_actual + tidx_col] = clip(res, -mval, mval);
            }
        }
    }
}

void ssaturate_int_by_bits_cu_temp(
    int32_t *input, int8_t *output, size_t num, size_t row_pad, size_t row_actual, size_t col_pad,
    size_t col_actual, uint32_t res_mult, uint32_t res_shift, uint8_t bits, workspace_t *ws)
{
    ssaturate_int_by_bits_ker_temp<<<(num + 511) / 512, 512, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, num, row_pad, row_actual, col_pad, col_actual, res_mult, res_shift, bits);
}

void get_quant_scale_int32(float x, uint32_t *multer, int8_t *shift)
{
    *shift = 0;
    if (fabs(x - 0) < 1e-10) {
        *multer = 0;
        return;
    }
    while (x >= 1.0f) {
        x /= 2;
        *shift -= 1;
    }
    while (x < 0.5f) {
        x *= 2;
        *shift += 1;
    }
    *shift += 31;
    *multer = x * (1ll << 31);
    return;
}

__global__ void transpose_NCHW_2_NHWC_quantize_i8_ker(
    float *input, int8_t *output, int n, int c, int hw, float alpha)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int block_n_tile = blockIdx.x;
    int block_hw_tile = blockIdx.y;

    int line_off = laneId; // % 8;
    int line_num = warpId; // laneId / 8 + warpId * 4;

    int batch_offset = block_n_tile * c * hw;
    int hw_offset = block_hw_tile * 32;

    int threshold = 127;
    __shared__ float smem[32][32 + 1];

    {
        for (int i = 0; i < c; i += 32) {
            float fragment[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                fragment[j]
                    = input[batch_offset + hw_offset + line_num * hw + line_off + (i + j * 4) * hw];
            }

#pragma unroll
            for (int j = 0; j < 8; j++) {
                smem[j * 4 + line_num][line_off] = fragment[j];
            }
            __syncthreads();

            float trans_fragment[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                trans_fragment[j] = smem[line_off][j * 4 + line_num];
            }
            int8_t t_int[8];
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int t = rintf(trans_fragment[j] * alpha);
                t_int[j] = t > threshold ? threshold : (t < -threshold - 1 ? -threshold - 1 : t);
            }
#pragma unroll
            for (int j = 0; j < 8; j++) {
                int hw_idx = hw_offset + ((j * 4) + line_num);
                if (hw_idx < hw)
                    output[batch_offset + line_off + i + hw_idx * c] = t_int[j];
            }
        }
    }
}

// NCHW to NHWC
void transpose_NCHW_2_NHWC_quantize_i8_cu(
    float *input, int8_t *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    assert(c >= 32 && c % 32 == 0);
    dim3 grid(n, (h * w + 31) / 32);
    dim3 block(32, 4);
    transpose_NCHW_2_NHWC_quantize_i8_ker<<<grid, block, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, n, c, h * w, alpha);
}

__global__ void transpose_NHWC_2_NCHW_dequantize_i8_split_c_ker(
    int8_t *input, float *output, int n, int c, int hw, float alpha)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int block_n_tile = blockIdx.x;
    int block_c_tile = blockIdx.y;

    int line_off = laneId;
    int line_num = warpId;

    int batch_offset = block_n_tile * hw * c;
    int c_offset = block_c_tile * 32;

    __shared__ int8_t smem[32][32 + 1];

    {
        for (int i = 0; i < hw; i += 32) {
            for (int j = 0; j < 8; j++) {
                smem[j * 4 + line_num][line_off]
                    = input[batch_offset + c_offset + line_num * c + line_off + (i + j * 4) * c];
            }
            __syncthreads();

            float trans_fragment[8];
            for (int j = 0; j < 8; j++) {
                trans_fragment[j] = smem[line_off][j * 4 + line_num] * alpha;
            }

            if (i + line_off < hw) {
                for (int j = 0; j < 8; j++) {
                    output[batch_offset + (c_offset + j * 4 + line_num) * hw + i + line_off]
                        = trans_fragment[j];
                }
            }
        }
    }
}

void transpose_NHWC_2_NCHW_dequantize_i8_split_c_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    assert(c >= 32 && c % 32 == 0);
    dim3 grid(n, (c + 31) / 32);
    dim3 block(32, 4);
    transpose_NHWC_2_NCHW_dequantize_i8_split_c_ker<<<grid, block, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, n, c, h * w, alpha);
}

__global__ void transpose_NHWC_2_NCHW_dequantize_i8_split_hw_ker(
    int8_t *input, float *output, int n, int c, int hw, float alpha)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int block_n_tile = blockIdx.x;
    int block_hw_tile = blockIdx.y;

    int line_off = laneId;
    int line_num = warpId;

    int batch_offset = block_n_tile * hw * c;
    int hw_offset = block_hw_tile * 32;

    __shared__ int8_t smem[32][32];

    {
        for (int i = 0; i < c; i += 32) {

            for (int j = 0; j < 8; j++) {
                smem[j * 4 + line_num][line_off]
                    = input[batch_offset + (hw_offset + j * 4 + line_num) * c + i + line_off];
            }
            __syncthreads();

            if (hw_offset + line_off < hw) {
                float trans_fragment[8];
                for (int j = 0; j < 8; j++) {
                    trans_fragment[j] = smem[line_off][j * 4 + line_num] * alpha;
                }

                for (int j = 0; j < 8; j++) {
                    output[batch_offset + (i + j * 4 + line_num) * hw + hw_offset + line_off]
                        = trans_fragment[j];
                }
            }
        }
    }
}

void transpose_NHWC_2_NCHW_dequantize_i8_split_hw_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    assert(c >= 32 && c % 32 == 0);
    dim3 grid(n, (h * w + 31) / 32);
    dim3 block(32, 4);
    transpose_NHWC_2_NCHW_dequantize_i8_split_hw_ker<<<grid, block, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, n, c, h * w, alpha);
}

// transpose from NHWC to NCHW
void transpose_NHWC_2_NCHW_dequantize_i8_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    if ((h == 1 && w == 1) || c % 32 != 0) {
        dequantize_i8_cu(input, output, n * c * h * w, alpha, ws);
    } else if (c > h * w) {
        transpose_NHWC_2_NCHW_dequantize_i8_split_c_cu(input, output, n, c, h, w, alpha, ws);
    } else {
        transpose_NHWC_2_NCHW_dequantize_i8_split_hw_cu(input, output, n, c, h, w, alpha, ws);
    }
}

__global__ void transpose_NCHW_2_NHWC_quantize_i4_ker(
    float *input, int8_t *output, int n, int c, int hw, float alpha)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int block_n_tile = blockIdx.x;
    int block_hw_tile = blockIdx.y;

    int line_off = laneId;
    int line_num = warpId;

    int batch_offset = block_n_tile * c * hw;
    int hw_offset = block_hw_tile * 32;
    int hw_idx = hw_offset + line_off;

    if (hw_idx < hw) {
        for (int i = 0; i < c; i += 32) {

            float trans_fragment[8];
            for (int j = 0; j < 4; j++) {
                trans_fragment[j * 2]
                    = input[batch_offset + hw_offset + line_off + (i + j * 8 + line_num * 2) * hw];
                trans_fragment[j * 2 + 1] = input
                    [batch_offset + hw_offset + line_off + (i + j * 8 + line_num * 2 + 1) * hw];
            }
            int8_t t_int[4];
            for (int j = 0; j < 4; j++) {
                int t1 = rintf(trans_fragment[j * 2] * alpha);
                int t2 = rintf(trans_fragment[j * 2 + 1] * alpha);
                t1 = t1 > 7 ? 7 : (t1 < -8 ? -8 : t1);
                t2 = t2 > 7 ? 7 : (t2 < -8 ? -8 : t2);
                t_int[j] = (t1 & 0xf) | (t2 << 4);
            }
            for (int j = 0; j < 4; j++) {
                output[(batch_offset + hw_idx * c + i + j * 8 + line_num * 2) / 2] = t_int[j];
            }
        }
    }
}

void transpose_NCHW_2_NHWC_quantize_i4_cu(
    float *input, int8_t *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    assert(c >= 32 && c % 32 == 0);
    dim3 grid(n, (h * w + 31) / 32);
    dim3 block(32, 4);
    transpose_NCHW_2_NHWC_quantize_i4_ker<<<grid, block, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, n, c, h * w, alpha);
}

__global__ void transpose_NHWC_2_NCHW_dequantize_i4_split_c_ker(
    int8_t *input, float *output, int n, int c, int hw, float alpha)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int block_n_tile = blockIdx.x;
    int block_c_tile = blockIdx.y;

    int line_off = laneId % 16;
    int line_num = warpId * 2 + laneId / 16;

    int batch_offset = block_n_tile * hw * c;
    int c_offset = block_c_tile * 32;

    __shared__ int8_t smem[32][16 + 1];

    {
        for (int i = 0; i < hw; i += 32) {

            for (int j = 0; j < 4; j++) {
                smem[j * 8 + line_num][line_off]
                    = input[(batch_offset + (i + j * 8 + line_num) * c + c_offset) / 2 + line_off];
            }
            __syncthreads();

            // row to col
            if (i + laneId < hw) {
                float trans_fragment[8];
                for (int j = 0; j < 4; j++) {
                    int8_t t_int8 = smem[laneId][j * 4 + warpId] << 4;
                    t_int8 = t_int8 >> 4;
                    trans_fragment[j * 2] = t_int8 * alpha;
                    t_int8 = smem[laneId][j * 4 + warpId] >> 4;
                    trans_fragment[j * 2 + 1] = t_int8 * alpha;
                }
                for (int j = 0; j < 4; j++) {
                    output[batch_offset + (c_offset + j * 8 + warpId * 2) * hw + i + laneId]
                        = trans_fragment[j * 2];
                    output[batch_offset + (c_offset + j * 8 + warpId * 2 + 1) * hw + i + laneId]
                        = trans_fragment[j * 2 + 1];
                }
            }
        }
    }
}

void transpose_NHWC_2_NCHW_dequantize_i4_split_c_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    assert(c >= 32 && c % 32 == 0);
    dim3 grid(n, (c + 31) / 32);
    dim3 block(32, 4);
    transpose_NHWC_2_NCHW_dequantize_i4_split_c_ker<<<grid, block, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, n, c, h * w, alpha);
}

__global__ void transpose_NHWC_2_NCHW_dequantize_i4_split_hw_ker(
    int8_t *input, float *output, int n, int c, int hw, float alpha)
{
    int laneId = threadIdx.x;
    int warpId = threadIdx.y;

    int block_n_tile = blockIdx.x;
    int block_hw_tile = blockIdx.y;

    int line_off = laneId % 16;
    int line_num = warpId * 2 + laneId / 16;

    int batch_offset = block_n_tile * hw * c;
    int hw_offset = block_hw_tile * 32;

    __shared__ int8_t smem[32][16];

    {
        for (int i = 0; i < c; i += 32) {

            for (int j = 0; j < 4; j++) {
                smem[j * 8 + line_num][line_off]
                    = input[(batch_offset + (hw_offset + j * 8 + line_num) * c + i) / 2 + line_off];
            }
            __syncthreads();

            // row to col
            if (hw_offset + laneId < hw) {
                float trans_fragment[8];
                for (int j = 0; j < 4; j++) {
                    int8_t t_int8 = smem[laneId][j * 4 + warpId] << 4;
                    t_int8 = t_int8 >> 4;
                    trans_fragment[j * 2] = t_int8 * alpha;
                    t_int8 = smem[laneId][j * 4 + warpId] >> 4;
                    trans_fragment[j * 2 + 1] = t_int8 * alpha;
                }
                for (int j = 0; j < 4; j++) {
                    output[batch_offset + (i + j * 8 + warpId * 2) * hw + hw_offset + laneId]
                        = trans_fragment[j * 2];
                    output[batch_offset + (i + j * 8 + warpId * 2 + 1) * hw + hw_offset + laneId]
                        = trans_fragment[j * 2 + 1];
                }
            }

            // // col to row
            // float trans_fragment[8];
            // for (int j = 0; j < 4; j++) {
            //     int8_t t_int8 = smem[j * 8 + line_num][line_off] << 4;
            //     trans_fragment[j*2] = (t_int8 >> 4) * alpha;
            //     trans_fragment[j*2 + 1] = (smem[j * 8 + line_num][line_off] >> 4) * alpha;
            // }
            // for (int j = 0; j < 4; j++) {
            //     if (hw_offset + j*8 + line_num < hw) {
            //         output[batch_offset + (i + line_off*2) * hw + hw_offset + j*8 + line_num] =
            //         trans_fragment[j*2]; output[batch_offset + (i + line_off*2 + 1) * hw +
            //         hw_offset + j*8 + line_num] = trans_fragment[j*2 + 1];
            //     }
            // }
        }
    }
}

void transpose_NHWC_2_NCHW_dequantize_i4_split_hw_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    assert(c >= 32 && c % 32 == 0);
    dim3 grid(n, (h * w + 31) / 32);
    dim3 block(32, 4);
    transpose_NHWC_2_NCHW_dequantize_i4_split_hw_ker<<<grid, block, 0, CUDA_WORKSPACE_STREAM(ws)>>>(
        input, output, n, c, h * w, alpha);
}

// transpose from NHWC to NCHW
void transpose_NHWC_2_NCHW_dequantize_i4_cu(
    int8_t *input, float *output, int n, int c, int h, int w, float alpha, workspace_t *ws)
{
    if ((h == 1 && w == 1) || c % 32 != 0) {
        dequantize_i4_cu(input, output, n * c * h * w, alpha, ws);
    } else if (c > h * w) {
        transpose_NHWC_2_NCHW_dequantize_i4_split_c_cu(input, output, n, c, h, w, alpha, ws);
    } else {
        transpose_NHWC_2_NCHW_dequantize_i4_split_hw_cu(input, output, n, c, h, w, alpha, ws);
    }
}
