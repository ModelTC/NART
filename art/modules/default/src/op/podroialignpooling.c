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

#include "../utils/utils.h"

typedef struct {
    op_t o;
    float spatial_scale;
    uint32_t pooled_height;
    uint32_t pooled_width;
    uint32_t sample_num;
} op_podroialignpooling_t;

op_podroialignpooling_t *op_default_podroialignpooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_podroialignpooling_t *res
        = (op_podroialignpooling_t *)malloc(sizeof(op_podroialignpooling_t));
    memset(res, 0, sizeof(op_podroialignpooling_t));
    return res;
}

void op_default_podroialignpooling_tp_config(op_t *op)
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

void op_default_podroialignpooling_tp_destroy(op_t *op) { (void)op; }

void op_default_podroialignpooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

// TODO
// c implementation same as roialignpooling
static void op_default_podroialignpooling_run(op_t *op)
{
    op_podroialignpooling_t *pool_op = (op_podroialignpooling_t *)op;
    size_t n, pc, ph, pw;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const float spatial_scale = pool_op->spatial_scale;
    const size_t pooled_height = pool_op->pooled_height;
    const size_t pooled_width = pool_op->pooled_width;
    const size_t sample_ratio = pool_op->sample_num;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    const float *bottom_data;

    for (n = 0; n < num_rois; ++n) {
        size_t roi_batch_ind = input_1[n * roi_shape + 0];
        float roi_start_w = input_1[n * roi_shape + 1] * spatial_scale;
        float roi_start_h = input_1[n * roi_shape + 2] * spatial_scale;
        float roi_end_w = input_1[n * roi_shape + 3] * spatial_scale;
        float roi_end_h = input_1[n * roi_shape + 4] * spatial_scale;

        float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
        float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
        float bin_size_h = roi_height / pooled_height;
        float bin_size_w = roi_width / pooled_width;

        for (pc = 0; pc < channel_in; ++pc) {
            bottom_data = input_0 + (roi_batch_ind * channel_in + pc) * height_in * width_in;
            for (ph = 0; ph < pooled_height; ++ph) {
                for (pw = 0; pw < pooled_width; ++pw) {
                    float hstart = (ph)*bin_size_h;
                    float wstart = (pw)*bin_size_w;
                    float hend = (ph + 1) * bin_size_h;
                    float wend = (pw + 1) * bin_size_w;
                    // Add roi offsets and clip to input boundaries
                    hstart = fminf(fmaxf(hstart + roi_start_h, 0.0f), height_in - 1);
                    hend = fminf(fmaxf(hend + roi_start_h, 0.0f), height_in - 1);
                    wstart = fminf(fmaxf(wstart + roi_start_w, 0.0f), width_in - 1);
                    wend = fminf(fmaxf(wend + roi_start_w, 0.0f), width_in - 1);
                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    // Define an empty pooling region to be zero
                    float maxval = is_empty ? 0 : -FLT_MAX;

                    float sample_h = bin_size_h / (sample_ratio + 1);
                    float sample_w = bin_size_w / (sample_ratio + 1);

                    bool updated = false;
                    size_t i, j;
                    for (i = 1; i <= sample_ratio; ++i) {
                        for (j = 1; j <= sample_ratio; ++j) {
                            float cur_h = hstart + i * sample_h;
                            float cur_w = wstart + j * sample_w;
                            if (cur_h >= hend || cur_w >= wend)
                                continue;
                            maxval = bilinear_interpolate(
                                bottom_data, height_in, width_in, cur_h, cur_w);
                            updated = true;
                        }
                    }
                    *output_0++ = updated ? maxval : 0;
                }
            }
        }
    }
}

void op_default_podroialignpooling_tp_prepare(op_t *op)
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
        op->run_func = op_default_podroialignpooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
