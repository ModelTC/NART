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
} op_roipooling_t;

op_roipooling_t *op_default_roipooling_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_roipooling_t *res = (op_roipooling_t *)malloc(sizeof(op_roipooling_t));
    memset(res, 0, sizeof(op_roipooling_t));
    return res;
}

void op_default_roipooling_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_ROIPOOLING_SPATIAL_SCALE, dtFLOAT32, &((op_roipooling_t *)op)->spatial_scale));
    CHECK(op_setting_single_get(
        op, SETTING_ROIPOOLING_POOLED_HEIGHT, dtUINT32, &((op_roipooling_t *)op)->pooled_height));
    CHECK(op_setting_single_get(
        op, SETTING_ROIPOOLING_POOLED_WIDTH, dtUINT32, &((op_roipooling_t *)op)->pooled_width));
}

void op_default_roipooling_tp_destroy(op_t *op) { (void)op; }

void op_default_roipooling_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_roipooling_run(op_t *op)
{
    op_roipooling_t *pool_op = (op_roipooling_t *)op;
    size_t n, pc, ph, pw;
    const size_t channel_in = op->input_tensors[0]->shape.dim[1];
    const size_t height_in = op->input_tensors[0]->shape.dim[2];
    const size_t width_in = op->input_tensors[0]->shape.dim[3];
    const size_t num_rois = op->input_tensors[1]->shape.dim[0];
    const size_t roi_shape = op->input_tensors[1]->shape.dim[1];
    const size_t pixels = height_in * width_in;
    const float spatial_scale = pool_op->spatial_scale;
    const size_t pooled_height = pool_op->pooled_height;
    const size_t pooled_width = pool_op->pooled_width;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    const float *input_1 = mem_cpu_data(op->input_tensors[1]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);

    for (n = 0; n < num_rois; ++n) {
        size_t roi_batch_ind = input_1[n * roi_shape + 0];
        int roi_start_w = round(input_1[n * roi_shape + 1] * spatial_scale);
        int roi_start_h = round(input_1[n * roi_shape + 2] * spatial_scale);
        int roi_end_w = round(input_1[n * roi_shape + 3] * spatial_scale);
        int roi_end_h = round(input_1[n * roi_shape + 4] * spatial_scale);

        int roi_height = MAX(roi_end_h - roi_start_h + 1, 1);
        int roi_width = MAX(roi_end_w - roi_start_w + 1, 1);
        float bin_size_h = (float)roi_height / (float)pooled_height;
        float bin_size_w = (float)roi_width / (float)pooled_width;

        for (pc = 0; pc < channel_in; ++pc) {
            for (ph = 0; ph < pooled_height; ++ph) {
                for (pw = 0; pw < pooled_width; ++pw) {
                    int hstart = (int)floor((float)ph * bin_size_h);
                    int wstart = (int)floor((float)pw * bin_size_w);
                    int hend = (int)ceil((float)(ph + 1) * bin_size_h);
                    int wend = (int)ceil((float)(pw + 1) * bin_size_w);

                    hstart = MIN(MAX(hstart + roi_start_h, 0), height_in);
                    hend = MIN(MAX(hend + roi_start_h, 0), height_in);
                    wstart = MIN(MAX(wstart + roi_start_w, 0), width_in);
                    wend = MIN(MAX(wend + roi_start_w, 0), width_in);

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    float max_val = is_empty ? 0 : -FLT_MAX;

                    int h, w;
                    size_t bottom_base_offset = (roi_batch_ind * channel_in + pc) * pixels;
                    for (h = hstart; h < hend; ++h) {
                        size_t roi_row_offset = h * width_in;
                        for (w = wstart; w < wend; ++w) {
                            size_t bottom_index = roi_row_offset + w;
                            if (input_0[bottom_base_offset + bottom_index] > max_val) {
                                max_val = input_0[bottom_base_offset + bottom_index];
                            }
                        }
                    }
                    *output_0++ = max_val;
                }
            }
        }
    }
}

void op_default_roipooling_tp_prepare(op_t *op)
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
        op->run_func = op_default_roipooling_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
