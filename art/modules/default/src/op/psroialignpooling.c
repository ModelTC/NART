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
    uint32_t output_dim;
    uint32_t group_size;
    uint32_t sample_num;
} op_psroialignpooling_t;

op_psroialignpooling_t *op_default_psroialignpooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_psroialignpooling_t *res = (op_psroialignpooling_t *)malloc(sizeof(op_psroialignpooling_t));
    memset(res, 0, sizeof(op_psroialignpooling_t));
    return res;
}

void op_default_psroialignpooling_tp_config(op_t *op)
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

void op_default_psroialignpooling_tp_destroy(op_t *op) { (void)op; }

void op_default_psroialignpooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_psroialignpooling_run(op_t *op)
{
    op_psroialignpooling_t *pool_op = (op_psroialignpooling_t *)op;
    size_t n, ctop, ph, pw;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const size_t pixels = height_in * width_in;
    const float spatial_scale = pool_op->spatial_scale;
    const size_t output_dim = pool_op->output_dim;
    const size_t group_size = pool_op->group_size;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < num_rois; ++n) {
        size_t roi_batch_ind = input_1[n * roi_shape + 0];
        float roi_start_w = (float)(round(input_1[n * roi_shape + 1])) * spatial_scale;
        float roi_start_h = (float)(round(input_1[n * roi_shape + 2])) * spatial_scale;
        float roi_end_w = (float)(round(input_1[n * roi_shape + 3]) + 1.f) * spatial_scale;
        float roi_end_h = (float)(round(input_1[n * roi_shape + 4]) + 1.f) * spatial_scale;

        float roi_height = MAX(roi_end_h - roi_start_h, 0.1);
        float roi_width = MAX(roi_end_w - roi_start_w, 0.1);

        float bin_size_h = roi_height / (float)group_size;
        float bin_size_w = roi_width / (float)group_size;

        for (ctop = 0; ctop < output_dim; ++ctop) {
            for (ph = 0; ph < group_size; ++ph) {
                for (pw = 0; pw < group_size; ++pw) {

                    int hstart = floor((float)ph * bin_size_h + roi_start_h);
                    int wstart = floor((float)pw * bin_size_w + roi_start_w);
                    int hend = ceil((float)(ph + 1) * bin_size_h + roi_start_h);
                    int wend = ceil((float)(pw + 1) * bin_size_w + roi_start_w);

                    hstart = MIN(MAX(hstart, 0), height_in);
                    hend = MIN(MAX(hend, 0), height_in);
                    wstart = MIN(MAX(wstart, 0), width_in);
                    wend = MIN(MAX(wend, 0), width_in);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    size_t gw = pw;
                    size_t gh = ph;
                    size_t c = (ctop * group_size + gh) * group_size + gw;

                    int h, w;
                    float out_sum = 0;
                    size_t bottom_base_offset = (roi_batch_ind * channel_in + c) * pixels;
                    const float *current_bottom = input_0 + bottom_base_offset;
                    for (h = hstart; h < hend; ++h) {
                        size_t roi_row_offset = h * width_in;
                        for (w = wstart; w < wend; ++w) {
                            size_t bottom_index = roi_row_offset + w;
                            out_sum += current_bottom[bottom_index];
                        }
                    }

                    float bin_area = (hend - hstart) * (wend - wstart);
                    *output_0++ = is_empty ? 0.f : out_sum / bin_area;
                }
            }
        }
    }
}

void op_default_psroialignpooling_tp_prepare(op_t *op)
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
        op->run_func = op_default_psroialignpooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
