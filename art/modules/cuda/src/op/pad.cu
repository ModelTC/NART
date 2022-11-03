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

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

typedef struct {
    op_t o;
    uint32_t mode;
    float value;
    int32_t *pads;
} op_pad_t;

op_pad_t *op_cuda_pad_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_pad_t *res = (op_pad_t *)malloc(sizeof(op_pad_t));
    memset(res, 0, sizeof(op_pad_t));
    return res;
}

void op_cuda_pad_tp_config(op_t *op)
{
    size_t len;
    CHECK(op_setting_single_get(op, SETTING_PAD_MODE, dtUINT32, &((op_pad_t *)op)->mode));
    CHECK(op_setting_single_get(op, SETTING_PAD_VALUE, dtFLOAT32, &((op_pad_t *)op)->value));
    CHECK(op_setting_array_get(op, SETTING_PAD_PADS, dtINT32, &len, &((op_pad_t *)op)->pads));
    CHECK(len == 8);
    /* TODO:
    Only supprt NCHW now;
    High dimensions need to be added later?
    */
}

void op_cuda_pad_tp_destroy(op_t *op) { (void)op; }

void op_cuda_pad_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void PadConst(
    const int nthreads, const float *bottom_data, const int channel, const int height,
    const int width, const int padded_channel, const int padded_height, const int padded_width,
    const int pad_c, const int pad_h, const int pad_w, const float value, float *top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int n = index / (padded_channel * padded_height * padded_width);
        const int pw = index % padded_width;
        const int ph = (index / padded_width) % padded_height;
        const int pc = (index / (padded_width * padded_height)) % padded_channel;
        const int h = ph - pad_h;
        const int w = pw - pad_w;
        const int c = pc - pad_c;
        top_data[index] = (h < 0 || w < 0 || c < 0 || h >= height || w >= width || c >= channel)
            ? value
            : bottom_data[((n * channel + c) * height + h) * width + w];
    }
}

__global__ void PadReflect(
    const int nthreads, const float *const bottom_data, const int height, const int width,
    const int padded_height, const int padded_width, const int pad_t, const int pad_l,
    float *top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int nc = index / padded_width;
        const int pw = index % padded_width;
        const int ph = nc % padded_height;
        nc /= padded_height;
        int h = ph - pad_t;
        int w = pw - pad_l;
        h = MAX(h, -h);
        w = MAX(w, -w);
        h = MIN(h, 2 * height - h - 2);
        w = MIN(w, 2 * width - w - 2);
        top_data[index] = bottom_data[(nc * height + h) * width + w];
    }
}

__global__ void PadEdge(
    const int nthreads, const float *const bottom_data, const int height, const int width,
    const int padded_height, const int padded_width, const int pad_t, const int pad_l,
    float *top_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        int nc = index / padded_width;
        const int pw = index % padded_width;
        const int ph = nc % padded_height;
        nc /= padded_height;
        const int h = MIN(height - 1, MAX(ph - pad_t, 0));
        const int w = MIN(width - 1, MAX(pw - pad_l, 0));
        top_data[index] = bottom_data[(nc * height + h) * width + w];
    }
}

static void op_cuda_pad_run(op_t *op)
{
    uint32_t mode = ((op_pad_t *)op)->mode;
    float val = ((op_pad_t *)op)->value;
    int32_t *pads;
    size_t len;
    CHECK(op_setting_array_get(op, SETTING_PAD_PADS, dtINT32, &len, &pads));

    int o_c = op->output_tensors[0].shape.dim[1];
    int o_h = op->output_tensors[0].shape.dim[2];
    int o_w = op->output_tensors[0].shape.dim[3];
    int i_c = op->input_tensors[0]->shape.dim[1];
    int i_h = op->input_tensors[0]->shape.dim[2];
    int i_w = op->input_tensors[0]->shape.dim[3];
    // int batch_size = op->input_tensors[0]->shape.dim[0];
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    size_t count = shape_count(&op->output_tensors[0].shape);
#define CONSTANT 1
#define REFLECT  2
#define EDGE     3
    switch (mode) {
    case CONSTANT:
        PadConst<<<(count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            count, input_0, i_c, i_h, i_w, o_c, o_h, o_w, pads[1], pads[2], pads[3], val, output_0);
        break;
    case REFLECT:
        PadReflect<<<(count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            count, input_0, i_h, i_w, o_h, o_w, pads[2], pads[3], output_0);
        break;
    case EDGE:
        PadEdge<<<(count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
            count, input_0, i_h, i_w, o_h, o_w, pads[2], pads[3], output_0);
        break;
    }
#undef CONSTANT
#undef REFLECT
#undef EDGE
    (void)op;
}

void op_cuda_pad_tp_prepare(op_t *op)
{
    int i;
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_cuda_pad_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
