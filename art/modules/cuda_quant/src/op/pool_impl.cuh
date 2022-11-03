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

#ifndef POOL_IMPL_CUH
#define POOL_IMPL_CUH

#include "../cuda_quant_workspace.h"
#include <cuda_runtime.h>

#include <assert.h>
#include <functional>
#include <vector>

#define MAX_VAL_INT8 127
#define MIN_VAL_INT8 -128
#define MAX_VAL_INT4 7
#define MIN_VAL_INT4 -8

template <int NUM_PER_BLOCK>
__global__ void maxpool2d_s8_ker_v16(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x * 16 + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        int8_t res[16];

#pragma unroll
        for (int i = 0; i < 16; i++) {
            res[i] = MIN_VAL_INT8;
        }

        int8_t buf[16];

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    ((int4 *)buf)[0] = __ldg(
                        (int4 *)&input[input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel]);
#pragma unroll
                    for (int k = 0; k < 16; k++) {
                        res[k] = max(res[k], buf[k]);
                    }
                }
            }
        }

        ((int4 *)(output + idx))[0] = ((int4 *)res)[0];
    }
}

template <int NUM_PER_BLOCK>
void maxpool2d_s8_cu_v16(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    assert(channel % 16 == 0);

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    maxpool2d_s8_ker_v16<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK / 16, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void maxpool2d_s8_ker_v1(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        int8_t res = MIN_VAL_INT8;
        int8_t buf;

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    buf = __ldg(
                        (int8_t *)&input[input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel]);
                    res = max(res, buf);
                }
            }
        }

        ((int8_t *)(output + idx))[0] = res;
    }
}

template <int NUM_PER_BLOCK>
void maxpool2d_s8_cu_v1(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    maxpool2d_s8_ker_v1<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void maxpool2d_s4_ker_v32(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x * 32 + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        register int8_t res_int8[32];
        register int8_t buf_int4[16];

#pragma unroll
        for (int i = 0; i < 32; i++)
            res_int8[i] = MIN_VAL_INT4;

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    ((int4 *)buf_int4)[0] = __ldg(
                        (int4 *)&input
                            [(input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel) / 2]);

#pragma unroll
                    for (int k = 0; k < 16; k++) {
                        res_int8[k * 2] = max(res_int8[k * 2], ((int8_t)(buf_int4[k] << 4)) >> 4);
                        res_int8[k * 2 + 1] = max(res_int8[k * 2 + 1], buf_int4[k] >> 4);
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 16; i++) {
            buf_int4[i] = (res_int8[i * 2] & 0xf) | (res_int8[i * 2 + 1] << 4);
        }

        ((int4 *)(output + idx / 2))[0] = ((int4 *)buf_int4)[0];
    }
}

template <int NUM_PER_BLOCK>
void maxpool2d_s4_cu_v32(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    assert(channel % 32 == 0);

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    maxpool2d_s4_ker_v32<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK / 32, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void maxpool2d_s4_ker_v2(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x * 2 + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        register int8_t res_int8[2];
        register int8_t buf_int4;

        res_int8[0] = MIN_VAL_INT4;
        res_int8[1] = MIN_VAL_INT4;

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    buf_int4 = __ldg(
                        (int8_t *)&input
                            [(input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel) / 2]);

                    res_int8[0] = max(res_int8[0], ((int8_t)(buf_int4 << 4)) >> 4);
                    res_int8[1] = max(res_int8[1], buf_int4 >> 4);
                }
            }
        }

        buf_int4 = (res_int8[0] & 0xf) | (res_int8[1] << 4);
        ((int8_t *)(output + idx / 2))[0] = buf_int4;
    }
}

template <int NUM_PER_BLOCK>
void maxpool2d_s4_cu_v2(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    assert(channel % 2 == 0);

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    maxpool2d_s4_ker_v2<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK / 2, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void avgpool2d_s8_ker_v16(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x * 16 + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        int res[16] = { 0 };
        int8_t buf[16];
        float ff = 1.0f / (kh * kw);

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    ((int4 *)buf)[0] = __ldg(
                        (int4 *)&input[input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel]);

#pragma unroll
                    for (int k = 0; k < 16; k++) {
                        res[k] += buf[k];
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 16; i++) {
            res[i] = rintf(res[i] * ff);
            buf[i] = res[i] > MAX_VAL_INT8
                ? MAX_VAL_INT8
                : (res[i] < MIN_VAL_INT8 ? MIN_VAL_INT8 : (int8_t)res[i]);
        }

        ((int4 *)(output + idx))[0] = ((int4 *)buf)[0];
    }
}

template <int NUM_PER_BLOCK>
void avgpool2d_s8_cu_v16(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    assert(channel % 16 == 0);

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    avgpool2d_s8_ker_v16<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK / 16, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void avgpool2d_s8_ker_v1(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        int res;
        int8_t buf;
        float ff = 1.0f / (kh * kw);

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    buf = __ldg(
                        (int8_t *)&input[input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel]);

                    res += buf;
                }
            }
        }

        res = rintf(res * ff);
        buf = res > MAX_VAL_INT8 ? MAX_VAL_INT8 : (res < MIN_VAL_INT8 ? MIN_VAL_INT8 : (int8_t)res);

        ((int8_t *)(output + idx))[0] = buf;
    }
}

template <int NUM_PER_BLOCK>
void avgpool2d_s8_cu_v1(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    avgpool2d_s8_ker_v1<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void avgpool2d_s4_ker_v32(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x * 32 + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        register int res_int[32] = { 0 };
        register int8_t buf_int4[16];
        float ff = 1.0f / (kh * kw);

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    ((int4 *)buf_int4)[0] = __ldg(
                        (int4 *)&input
                            [(input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel) / 2]);

#pragma unroll
                    for (int k = 0; k < 16; k++) {
                        res_int[k * 2] += ((int8_t)(buf_int4[k] << 4)) >> 4;
                        res_int[k * 2 + 1] += buf_int4[k] >> 4;
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 16; i++) {
            res_int[i * 2] = rintf(res_int[i * 2] * ff);
            int8_t tmp1 = res_int[i * 2] > MAX_VAL_INT4
                ? MAX_VAL_INT4
                : (res_int[i * 2] < MIN_VAL_INT4 ? MIN_VAL_INT4 : res_int[i * 2]);

            res_int[i * 2 + 1] = rintf(res_int[i * 2 + 1] * ff);
            int8_t tmp2 = res_int[i * 2 + 1] > MAX_VAL_INT4
                ? MAX_VAL_INT4
                : (res_int[i * 2 + 1] < MIN_VAL_INT4 ? MIN_VAL_INT4 : res_int[i * 2 + 1]);

            buf_int4[i] = (tmp1 & 0xf) | (tmp2 << 4);
        }

        ((int4 *)(output + idx / 2))[0] = ((int4 *)buf_int4)[0];
    }
}

template <int NUM_PER_BLOCK>
void avgpool2d_s4_cu_v32(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    assert(channel % 32 == 0);

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    avgpool2d_s4_ker_v32<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK / 32, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

template <int NUM_PER_BLOCK>
__global__ void avgpool2d_s4_ker_v2(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int oh, int ow, int kh,
    int kw, int pad_h, int pad_w, int str_h, int str_w, int num)
{

    int idx = threadIdx.x * 2 + blockIdx.x * NUM_PER_BLOCK;

    if (idx < num) {
        int batch_idx = idx / (oh * ow * channel);
        int oh_idx = (idx / (ow * channel)) % oh;
        int ow_idx = (idx / channel) % ow;
        int channel_idx = idx % channel;

        register int res_int[2] = { 0 };
        register int8_t buf_int4;
        float ff = 1.0f / (kh * kw);

        int ih_idx = oh_idx * str_h - pad_h;
        int iw_idx = ow_idx * str_w - pad_w;
        int input_idx = batch_idx * ih * iw * channel + channel_idx;
        for (int i = 0; i < kh; i++) {
            for (int j = 0; j < kw; j++) {
                if (!(ih_idx + i < 0 || ih_idx + i >= ih || iw_idx + j < 0 || iw_idx + j >= iw)) {

                    buf_int4 = __ldg(
                        (int8_t *)&input
                            [(input_idx + ((ih_idx + i) * iw + (iw_idx + j)) * channel) / 2]);

                    res_int[0] += ((int8_t)(buf_int4 << 4)) >> 4;
                    res_int[1] += buf_int4 >> 4;
                }
            }
        }

        res_int[0] = rintf(res_int[0] * ff);
        int8_t tmp1 = res_int[0] > MAX_VAL_INT4
            ? MAX_VAL_INT4
            : (res_int[0] < MIN_VAL_INT4 ? MIN_VAL_INT4 : res_int[0]);
        res_int[1] = rintf(res_int[1] * ff);
        int8_t tmp2 = res_int[1] > MAX_VAL_INT4
            ? MAX_VAL_INT4
            : (res_int[1] < MIN_VAL_INT4 ? MIN_VAL_INT4 : res_int[1]);

        buf_int4 = (tmp1 & 0xf) | (tmp2 << 4);

        ((int8_t *)(output + idx / 2))[0] = buf_int4;
    }
}

template <int NUM_PER_BLOCK>
void avgpool2d_s4_cu_v2(
    int8_t *input, int8_t *output, int batch, int ih, int iw, int channel, int kh, int kw,
    int pad_h, int pad_w, int str_h, int str_w, workspace_t *ws)
{

    assert(channel % 2 == 0);

    int oh = (ih - kh + 2 * pad_h) / str_h + 1;
    int ow = (iw - kw + 2 * pad_w) / str_w + 1;
    int num = batch * oh * ow * channel;

    avgpool2d_s4_ker_v2<NUM_PER_BLOCK>
        <<<(num + NUM_PER_BLOCK - 1) / NUM_PER_BLOCK, NUM_PER_BLOCK / 2, 0,
           CUDA_WORKSPACE_STREAM(ws)>>>(
            input, output, batch, ih, iw, channel, oh, ow, kh, kw, pad_h, pad_w, str_h, str_w, num);
}

#endif
