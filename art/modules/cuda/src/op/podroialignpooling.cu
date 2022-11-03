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

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "art/log.h"
#include "art/module.h"
#include "art/op.h"
#include "art/op_settings.h"
#include "art/op_tp.h"

#include "../../../default/src/utils/utils.h"
#include "../cuda_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    op_t o;
    float spatial_scale;
    uint32_t pooled_height;
    uint32_t pooled_width;
    uint32_t sample_num;
} op_podroialignpooling_t;

op_podroialignpooling_t *op_cuda_podroialignpooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_podroialignpooling_t *res
        = (op_podroialignpooling_t *)malloc(sizeof(op_podroialignpooling_t));
    memset(res, 0, sizeof(op_podroialignpooling_t));
    return res;
}

void op_cuda_podroialignpooling_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_PODROIALIGNPOOLING_SPATIAL_SCALE, dtFLOAT32,
        &((op_podroialignpooling_t *)op)->spatial_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PODROIALIGNPOOLING_POOLED_HEIGHT, dtUINT32,
        &((op_podroialignpooling_t *)op)->pooled_height));
    CHECK(op_setting_single_get(
        op, SETTING_PODROIALIGNPOOLING_POOLED_WIDTH, dtUINT32,
        &((op_podroialignpooling_t *)op)->pooled_width));
    CHECK(op_setting_single_get(
        op, SETTING_PODROIALIGNPOOLING_SAMPLE_NUM, dtUINT32,
        &((op_podroialignpooling_t *)op)->sample_num));
}

void op_cuda_podroialignpooling_tp_destroy(op_t *op) { (void)op; }

void op_cuda_podroialignpooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static __device__ float bilinear_interpolate(
    const float *bottom_data, const int height, const int width, float y, float x,
    const int index /* index for debug only*/)
{
    // deal with cases that inverse elements are out of feature map boundary
    // if (y < -1.0 || y > height || x < -1.0 || x > width) {  //old version
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0.f;
    }

    if (y <= 0) {
        y = 0;
    }
    if (x <= 0) {
        x = 0;
    }

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (float)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (float)x_low;
    } else {
        x_high = x_low + 1;
    }

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    float v1 = bottom_data[y_low * width + x_low];
    float v2 = bottom_data[y_low * width + x_high];
    float v3 = bottom_data[y_high * width + x_low];
    float v4 = bottom_data[y_high * width + x_high];
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}

__global__ void op_cuda_podroialignpooling_kernel(
    const float *bottom_data, const float *bottom_rois, float *top_data, const size_t count,
    const size_t channels, const size_t height, const size_t width, const float spatial_scale,
    const size_t aligned_height, const size_t aligned_width, const size_t sample_num,
    size_t roi_shape)
{
    CUDA_KERNEL_LOOP(index, count)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = index % aligned_width;
        int ph = (index / aligned_width) % aligned_height;
        int c = (index / aligned_width / aligned_height) % channels;
        int n = index / aligned_width / aligned_height / channels;

        const float *offset_bottom_rois = bottom_rois + n * roi_shape;
        int roi_batch_ind = offset_bottom_rois[0];

        // Do not using rounding; this implementation detail is critical
        float roi_start_w = offset_bottom_rois[1] * spatial_scale;
        float roi_start_h = offset_bottom_rois[2] * spatial_scale;
        float roi_end_w = offset_bottom_rois[3] * spatial_scale;
        float roi_end_h = offset_bottom_rois[4] * spatial_scale;

        // Force malformed ROIs to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
        float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
        float bin_size_h = roi_height / static_cast<float>(aligned_height);
        float bin_size_w = roi_width / static_cast<float>(aligned_width);

        const float *offset_bottom_data
            = bottom_data + (roi_batch_ind * channels + c) * height * width;

        int roi_bin_grid_h = (sample_num > 0) ? sample_num : ceil(roi_height / aligned_height);
        int roi_bin_grid_w = (sample_num > 0) ? sample_num : ceil(roi_width / aligned_width);
        const float num = roi_bin_grid_h * roi_bin_grid_w;

        float output_val = 0.f;
        for (int iy = 0; iy < roi_bin_grid_h; ++iy) {
            const float y
                = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / roi_bin_grid_h;
            for (int ix = 0; ix < roi_bin_grid_w; ++ix) {
                const float x
                    = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / roi_bin_grid_w;
                float val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
                output_val += val;
            }
        }
        output_val /= num;
        top_data[index] = output_val;
    }
}

static void op_cuda_podroialignpooling_run(op_t *op)
{
    op_podroialignpooling_t *pool_op = (op_podroialignpooling_t *)op;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const float spatial_scale = pool_op->spatial_scale;
    const size_t pooled_height = pool_op->pooled_height;
    const size_t pooled_width = pool_op->pooled_width;
    const size_t sample_ratio = pool_op->sample_num;
    const float *input_0 = (const float *)mem_data(op->input_tensors[0]->mem);
    const float *input_1 = (const float *)mem_data(op->input_tensors[1]->mem);
    float *output_0 = (float *)mem_data(op->output_tensors[0].mem);
    size_t count = pooled_height * pooled_width * channel_in * num_rois;

    op_cuda_podroialignpooling_kernel<<<
        (count + 1024 - 1) / 1024, 1024, 0, CUDA_WORKSPACE_STREAM(op->workspace)>>>(
        input_0, input_1, output_0, count, channel_in, height_in, width_in, spatial_scale,
        pooled_height, pooled_width, sample_ratio, roi_shape);
}

void op_cuda_podroialignpooling_tp_prepare(op_t *op)
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
        op->run_func = op_cuda_podroialignpooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}

#ifdef __cplusplus
}
#endif
