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

// author : liyicheng
// create date : 2019-8-20
// purpose : implement the roi pooling layer in cuda (reference to caffeinfer's roipooling.cu)

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

#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

typedef struct {
    op_t o;
    float spatial_scale;
    uint32_t pooled_height;
    uint32_t pooled_width;
} op_roipooling_t;

op_roipooling_t *op_cuda_roipooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_roipooling_t *res = (op_roipooling_t *)malloc(sizeof(op_roipooling_t));
    memset(res, 0, sizeof(op_roipooling_t));
    return res;
}

void op_cuda_roipooling_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_ROIPOOLING_SPATIAL_SCALE, dtFLOAT32, &((op_roipooling_t *)op)->spatial_scale));
    CHECK(op_setting_single_get(
        op, SETTING_ROIPOOLING_POOLED_HEIGHT, dtUINT32, &((op_roipooling_t *)op)->pooled_height));
    CHECK(op_setting_single_get(
        op, SETTING_ROIPOOLING_POOLED_WIDTH, dtUINT32, &((op_roipooling_t *)op)->pooled_width));
}

void op_cuda_roipooling_tp_destroy(op_t *op) { (void)op; }

void op_cuda_roipooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_roipooling_kernel(
    const float *bottom_data, const float *bottom_rois, float *top_data, const size_t count,
    const size_t channels, const size_t height, const size_t width, const float spatial_scale,
    const size_t pooled_height, const size_t pooled_width, const size_t roi_shape)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;

        bottom_rois += n * roi_shape;
        int roi_batch_ind = bottom_rois[0];
        int roi_start_w = round(bottom_rois[1] * spatial_scale);
        int roi_start_h = round(bottom_rois[2] * spatial_scale);
        int roi_end_w = round(bottom_rois[3] * spatial_scale);
        int roi_end_h = round(bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = MAX(roi_end_w - roi_start_w + 1, 1);
        int roi_height = MAX(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = MIN(MAX(hstart + roi_start_h, 0), height);
        hend = MIN(MAX(hend + roi_start_h, 0), height);
        wstart = MIN(MAX(wstart + roi_start_w, 0), width);
        wend = MIN(MAX(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                if (bottom_data[bottom_index] > maxval) {
                    maxval = bottom_data[bottom_index];
                }
            }
        }
        top_data[index] = maxval;
    }
}

static void op_cuda_roipooling_run(op_t *op)
{
    op_roipooling_t *pool_op = (op_roipooling_t *)op;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const float spatial_scale = pool_op->spatial_scale;
    const size_t pooled_height = pool_op->pooled_height;
    const size_t pooled_width = pool_op->pooled_width;
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    size_t count = pooled_height * pooled_width * channel_in * num_rois;

    op_cuda_roipooling_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        input_0, input_1, output_0, count, channel_in, height_in, width_in, spatial_scale,
        pooled_height, pooled_width, roi_shape);
}

void op_cuda_roipooling_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_roipooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
