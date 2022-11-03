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

// author : zenghaolun
// create date : 2021-7-5
// purpose : implement the heatmap2coord in cuda (reference to caffeinfer's heatmap2coord.cu)

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    uint32_t coord_h;
    uint32_t coord_w;
    bool reposition;
} op_heatmap2coord_t;

op_heatmap2coord_t *op_cuda_heatmap2coord_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_heatmap2coord_t *res = (op_heatmap2coord_t *)malloc(sizeof(op_heatmap2coord_t));
    memset(res, 0, sizeof(op_heatmap2coord_t));
    return res;
}

void op_cuda_heatmap2coord_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_HEATMAP2COORD_COORD_H, dtUINT32, &((op_heatmap2coord_t *)op)->coord_h));
    CHECK(op_setting_single_get(
        op, SETTING_HEATMAP2COORD_COORD_W, dtUINT32, &((op_heatmap2coord_t *)op)->coord_w));
    CHECK(op_setting_single_get(
        op, SETTING_HEATMAP2COORD_REPOSITION, dtBOOL, &((op_heatmap2coord_t *)op)->reposition));
}

void op_cuda_heatmap2coord_tp_destroy(op_t *op) { (void)op; }

void op_cuda_heatmap2coord_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_heatmap2coord_kernel(
    const float *bottom_data, float *top_data, const size_t count, const int channels,
    const int height, const int width, const int coord_h, const int coord_w,
    const bool coord_reposition)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        int n = index / channels;
        int c = index % channels;
        int max_h = -1e9, max_w = -1e9;
        float score_max = -1e9;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                if (score_max < bottom_data[((n * channels + c) * height + h) * width + w]) {
                    max_h = h;
                    max_w = w;
                    score_max = bottom_data[((n * channels + c) * height + h) * width + w];
                }
            }
        }
        float fw = max_w, fh = max_h;
        if (coord_reposition && max_w > 0 && max_w < width - 1) {
            float leftscore
                = bottom_data[((n * channels + c) * height + max_h) * width + max_w - 1];
            float rightscore
                = bottom_data[((n * channels + c) * height + max_h) * width + max_w + 1];
            if (leftscore > rightscore)
                fw -= 0.25;
            else if (leftscore < rightscore)
                fw += 0.25;
        }
        if (coord_reposition && max_h > 0 && max_h < height - 1) {
            float upscore = bottom_data[((n * channels + c) * height + max_h - 1) * width + max_w];
            float downscore
                = bottom_data[((n * channels + c) * height + max_h + 1) * width + max_w];
            if (upscore > downscore)
                fh -= 0.25;
            else if (upscore < downscore)
                fh += 0.25;
        }
        top_data[n * (channels * 3) + c * 3 + 0] = (fw + 0.5) * coord_w / width;
        top_data[n * (channels * 3) + c * 3 + 1] = (fh + 0.5) * coord_h / height;
        top_data[n * (channels * 3) + c * 3 + 2] = score_max;
    }
}

static void op_cuda_heatmap2coord_run(op_t *op)
{
    op_heatmap2coord_t *heatmap2coord_op = (op_heatmap2coord_t *)op;
    const size_t num = op->input_tensors[0]->shape.dim[0];
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const uint32_t coord_h = heatmap2coord_op->coord_h;
    const uint32_t coord_w = heatmap2coord_op->coord_w;
    const bool reposition = heatmap2coord_op->reposition;
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    size_t count = num * channel_in;

    op_cuda_heatmap2coord_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        input_0, output_0, count, channel_in, height_in, width_in, coord_h, coord_w, reposition);
}

void op_cuda_heatmap2coord_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_heatmap2coord_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
