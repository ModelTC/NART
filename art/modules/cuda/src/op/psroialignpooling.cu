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
// create date : 2019-8-22
// purpose : implement the psroi align pooling layer in cuda (reference to caffeinfer's
// psroi_align_pooling.cu)

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
    uint32_t output_dim;
    uint32_t group_size;
    uint32_t sample_num;
} op_psroialignpooling_t;

op_psroialignpooling_t *op_cuda_psroialignpooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_psroialignpooling_t *res = (op_psroialignpooling_t *)malloc(sizeof(op_psroialignpooling_t));
    memset(res, 0, sizeof(op_psroialignpooling_t));
    return res;
}

void op_cuda_psroialignpooling_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_PSROIALIGNPOOLING_SPATIAL_SCALE, dtFLOAT32,
        &((op_psroialignpooling_t *)op)->spatial_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIALIGNPOOLING_OUTPUT_DIM, dtUINT32,
        &((op_psroialignpooling_t *)op)->output_dim));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIALIGNPOOLING_GROUP_SIZE, dtUINT32,
        &((op_psroialignpooling_t *)op)->group_size));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIALIGNPOOLING_SAMPLE_NUM, dtUINT32,
        &((op_psroialignpooling_t *)op)->sample_num));
}

void op_cuda_psroialignpooling_tp_destroy(op_t *op) { (void)op; }

void op_cuda_psroialignpooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

__global__ void op_cuda_psroialignpooling_kernel(
    const float *bottom_data, const float *bottom_rois, float *top_data, const size_t count,
    const size_t channels, const size_t height, const size_t width, const float spatial_scale,
    const size_t pooled_height, const size_t pooled_width, const int output_dim,
    const int group_size, const size_t roi_shape)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        // liqq 2016/09/25
        bottom_rois += n * roi_shape;

        int roi_batch_ind = (int)bottom_rois[0];
        float roi_start_w = static_cast<float>(round(bottom_rois[1])) * spatial_scale;
        float roi_start_h = static_cast<float>(round(bottom_rois[2])) * spatial_scale;
        float roi_end_w = static_cast<float>(round(bottom_rois[3]) + 1.) * spatial_scale;
        float roi_end_h = static_cast<float>(round(bottom_rois[4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w, 0.1); // avoid 0
        float roi_height = fmaxf(roi_end_h - roi_start_h, 0.1);

        // Compute w and h at bottom
        float bin_size_h = roi_height / static_cast<float>(pooled_height);
        float bin_size_w = roi_width / static_cast<float>(pooled_width);

        int hstart = (int)floor(static_cast<float>(ph) * bin_size_h + roi_start_h);
        int wstart = (int)floor(static_cast<float>(pw) * bin_size_w + roi_start_w);
        int hend = (int)ceil(static_cast<float>(ph + 1) * bin_size_h + roi_start_h);
        int wend = (int)ceil(static_cast<float>(pw + 1) * bin_size_w + roi_start_w);
        // Add roi offsets and clip to input boundaries
        hstart = MIN(MAX(hstart, 0), height);
        hend = MIN(MAX(hend, 0), height);
        wstart = MIN(MAX(wstart, 0), width);
        wend = MIN(MAX(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
        int gh = ph;
        int c = (ctop * group_size + gh) * group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        float out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                out_sum += bottom_data[bottom_index];
            }
        }

        float bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? 0. : out_sum / bin_area;
    }
}

static void op_cuda_psroialignpooling_run(op_t *op)
{
    op_psroialignpooling_t *pool_op = (op_psroialignpooling_t *)op;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const float spatial_scale = pool_op->spatial_scale;
    const size_t output_dim = pool_op->output_dim;
    const size_t group_size = pool_op->group_size;
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    size_t count = group_size * group_size * channel_in * num_rois;

    op_cuda_psroialignpooling_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        input_0, input_1, output_0, count, channel_in, height_in, width_in, spatial_scale,
        group_size, group_size, output_dim, group_size, roi_shape);
}

void op_cuda_psroialignpooling_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_psroialignpooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
