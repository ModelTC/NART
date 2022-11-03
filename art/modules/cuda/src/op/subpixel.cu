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

#include <stdlib.h>
#include <string.h>

#include "art/cuda/cuda_mem.h"
#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint8_t method; // 0: down 1: up
    uint32_t sample;
} op_subpixel_t;

op_subpixel_t *op_cuda_subpixel_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_subpixel_t *res = (op_subpixel_t *)malloc(sizeof(op_subpixel_t));
    memset(res, 0, sizeof(op_subpixel_t));
    return res;
}

void op_cuda_subpixel_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_SUBPIXEL_METHOD, dtUINT8, &((op_subpixel_t *)op)->method));
    CHECK(op_setting_single_get(
        op, SETTING_SUBPIXEL_SAMPLE, dtUINT32, &((op_subpixel_t *)op)->sample));
}

void op_cuda_subpixel_tp_destroy(op_t *op) { (void)op; }

void op_cuda_subpixel_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_subpixelup_kernel(
    size_t count, float *output, const float *input, uint32_t sample, size_t batch_size,
    size_t channel, size_t height, size_t width)
{
    CUDA_KERNEL_LOOP(i, count)
    {
        size_t n = i / (channel * height * width);
        size_t c = i / (height * width) % channel;
        size_t h = i / width % height;
        size_t w = i % width;

        size_t bc = channel * sample * sample;
        size_t bh = height / sample;
        size_t bw = width / sample;

        size_t bot_h = h / sample;
        size_t bot_w = w / sample;
        size_t sub_h = h % sample;
        size_t sub_w = w % sample;

        size_t bot_c = (c * sample + sub_h) * sample + sub_w;
        size_t old_idx = ((n * bc + bot_c) * bh + bot_h) * bw + bot_w;
        output[i] = input[old_idx];
    }
}

__global__ void op_cuda_subpixeldown_kernel(
    size_t count, float *output, const float *input, uint32_t sample, size_t batch_size,
    size_t channel, size_t height, size_t width)
{
    CUDA_KERNEL_LOOP(i, count)
    {
        size_t n = i / (channel * height * width);
        size_t c = i / (height * width) % channel;
        size_t h = i / width % height;
        size_t w = i % width;

        size_t bc = channel / (sample * sample);
        size_t bh = height * sample;
        size_t bw = width * sample;

        size_t bot_c = c / (sample * sample);
        size_t sub_idx = c % (sample * sample);
        size_t sub_h = sub_idx / sample;
        size_t sub_w = sub_idx % sample;
        size_t bot_h = h * sample + sub_h;
        size_t bot_w = w * sample + sub_w;

        size_t old_idx = ((n * bc + bot_c) * bh + bot_h) * bw + bot_w;
        output[i] = input[old_idx];
    }
}

static void op_cuda_subpixel_run(op_t *op)
{
    op_subpixel_t *subpixel_op = (op_subpixel_t *)op;
    shape_t out_shape = op->output_tensors[0].shape;
    size_t count = shape_count(&out_shape);
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);

    if (subpixel_op->method == 0) {
        op_cuda_subpixeldown_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            count, output_0, input_0, subpixel_op->sample, out_shape.dim[0], out_shape.dim[1],
            out_shape.dim[2], out_shape.dim[3]);
    } else if (subpixel_op->method == 1) {
        op_cuda_subpixelup_kernel<<<
            (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            count, output_0, input_0, subpixel_op->sample, out_shape.dim[0], out_shape.dim[1],
            out_shape.dim[2], out_shape.dim[3]);
    } else {
        LOG(error, "Invalid method for subpixel, should be {0, 1} but got: %d.\n",
            subpixel_op->method);
    }
}

void op_cuda_subpixel_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    // size_t count = op->output_tensors[0].shape.dim_size;
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_subpixel_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
