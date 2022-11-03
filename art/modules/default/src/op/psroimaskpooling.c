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
    float roi_scale;
    float bin_scale;
    uint32_t group_size;
    uint32_t output_dim;
} op_psroimaskpooling_t;

op_psroimaskpooling_t *op_default_psroimaskpooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_psroimaskpooling_t *res = (op_psroimaskpooling_t *)malloc(sizeof(op_psroimaskpooling_t));
    memset(res, 0, sizeof(op_psroimaskpooling_t));
    return res;
}

void op_default_psroimaskpooling_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_SPATIAL_SCALE, dtFLOAT32,
        &((op_psroimaskpooling_t *)op)->spatial_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_ROI_SCALE, dtFLOAT32,
        &((op_psroimaskpooling_t *)op)->roi_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_BIN_SCALE, dtFLOAT32,
        &((op_psroimaskpooling_t *)op)->bin_scale));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_GROUP_SIZE, dtUINT32,
        &((op_psroimaskpooling_t *)op)->group_size));
    CHECK(op_setting_single_get(
        op, SETTING_PSROIMASKPOOLING_OUTPUT_DIM, dtUINT32,
        &((op_psroimaskpooling_t *)op)->output_dim));
}

void op_default_psroimaskpooling_tp_destroy(op_t *op) { (void)op; }

void op_default_psroimaskpooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_psroimaskpooling_run(op_t *op)
{
    op_psroimaskpooling_t *pool_op = (op_psroimaskpooling_t *)op;
    size_t n, pc, ph, pw;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const size_t pixels = height_in * width_in;
    const float spatial_scale = pool_op->spatial_scale;
    const float roi_scale = pool_op->roi_scale;
    const float bin_scale = pool_op->bin_scale;
    const size_t group_size = pool_op->group_size;
    const size_t output_dim = pool_op->output_dim;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < num_rois; ++n) {
        const size_t roi_batch_ind = input_1[n * roi_shape + 0];
        const float x1 = input_1[n * roi_shape + 1];
        const float y1 = input_1[n * roi_shape + 2];
        const float x2 = input_1[n * roi_shape + 3];
        const float y2 = input_1[n * roi_shape + 4];

        float roi_height = y2 - y1;
        float roi_width = x2 - x1;
        float xc = (x1 + x2) * 0.5f;
        float yc = (y1 + y2) * 0.5f;

        float xx1 = xc - roi_width * roi_scale * 0.5f;
        float xx2 = xc + roi_width * roi_scale * 0.5f;
        float yy1 = yc - roi_height * roi_scale * 0.5f;
        float yy2 = yc + roi_height * roi_scale * 0.5f;

        float roi_start_w = round(xx1) * spatial_scale;
        float roi_start_h = round(yy1) * spatial_scale;
        float roi_end_w = (round(xx2) + 1.f) * spatial_scale;
        float roi_end_h = (round(yy2) + 1.f) * spatial_scale;

        roi_height = MAX(roi_end_h - roi_start_h, 0.1f);
        roi_width = MAX(roi_end_w - roi_start_w, 0.1f);

        float bin_size_h = roi_height / (float)group_size;
        float bin_size_w = roi_width / (float)group_size;

        float delta_h = (bin_size_h * bin_scale - bin_size_h) * 0.5f;
        float delta_w = (bin_size_w * bin_scale - bin_size_w) * 0.5f;

        for (pc = 0; pc < output_dim; ++pc) {
            for (ph = 0; ph < group_size; ++ph) {
                for (pw = 0; pw < group_size; ++pw) {
                    int hstart = (int)floor((float)ph * bin_size_h + roi_start_h - delta_h);
                    int wstart = (int)floor((float)pw * bin_size_w + roi_start_w - delta_w);
                    int hend = (int)ceil((float)(ph + 1) * bin_size_h + roi_start_h + delta_h);
                    int wend = (int)ceil((float)(pw + 1) * bin_size_w + roi_start_w + delta_w);

                    hstart = MIN(MAX(hstart, 0), height_in);
                    hend = MIN(MAX(hend, 0), height_in);
                    wstart = MIN(MAX(wstart, 0), width_in);
                    wend = MIN(MAX(wend, 0), width_in);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    size_t gw = pw;
                    size_t gh = ph;
                    size_t c = (pc * group_size + gh) * group_size + gw;
                    float out_sum = 0.f;

                    int h, w;
                    size_t bottom_base_offset = (roi_batch_ind * channel_in + c) * pixels;
                    for (h = hstart; h < hend; ++h) {
                        size_t roi_row_offset = h * width_in;
                        for (w = wstart; w < wend; ++w) {
                            size_t bottom_index = roi_row_offset + w;
                            out_sum += input_0[bottom_index + bottom_base_offset];
                        }
                    }
                    float bin_area = (hend - hstart) * (wend - wstart);
                    *output_0++ = is_empty ? 0.f : out_sum / bin_area;
                }
            }
        }
    }
}

void op_default_psroimaskpooling_tp_prepare(op_t *op)
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
        op->run_func = op_default_psroimaskpooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
