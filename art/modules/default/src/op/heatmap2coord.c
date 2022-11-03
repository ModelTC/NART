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
#include "art/op_settings.h"
#include "art/op_tp.h"

typedef struct {
    op_t o;
    uint32_t coord_h;
    uint32_t coord_w;
    bool reposition;
} op_heatmap2coord_t;

op_heatmap2coord_t *op_default_heatmap2coord_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_heatmap2coord_t *res = (op_heatmap2coord_t *)malloc(sizeof(op_heatmap2coord_t));
    memset(res, 0, sizeof(op_heatmap2coord_t));
    return res;
}

void op_default_heatmap2coord_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_HEATMAP2COORD_COORD_H, dtUINT32, &((op_heatmap2coord_t *)op)->coord_h));
    CHECK(op_setting_single_get(
        op, SETTING_HEATMAP2COORD_COORD_W, dtUINT32, &((op_heatmap2coord_t *)op)->coord_w));
    CHECK(op_setting_single_get(
        op, SETTING_HEATMAP2COORD_REPOSITION, dtBOOL, &((op_heatmap2coord_t *)op)->reposition));
}

void op_default_heatmap2coord_tp_destroy(op_t *op) { (void)op; }

void op_default_heatmap2coord_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_heatmap2coord_run(op_t *op)
{
    const op_heatmap2coord_t *hm2c_op = (op_heatmap2coord_t *)op;
    size_t n, c, h, w;
    size_t batch_size = op->input_tensors[0]->shape.dim[0];
    size_t channel_in = op->input_tensors[0]->shape.dim[1];
    size_t height_in = op->input_tensors[0]->shape.dim[2];
    size_t width_in = op->input_tensors[0]->shape.dim[3];
    bool reposition = hm2c_op->reposition;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    const float *temp_input_0 = input_0;

    for (n = 0; n < batch_size; ++n) {
        for (c = 0; c < channel_in; ++c) {
            // int max_h = -1, max_w = -1;
            size_t max_h = 0, max_w = 0;
            float max_score = -1e9;
            for (h = 0; h < height_in; ++h) {
                for (w = 0; w < width_in; ++w) {
                    float score = *input_0++;
                    if (score > max_score) {
                        max_score = score;
                        max_h = h;
                        max_w = w;
                    }
                }
            }
            float fh = max_h, fw = max_w;
            if (reposition && max_h > 0 && max_h < height_in - 1) {
                float up_score = temp_input_0
                    [((n * channel_in + c) * height_in + max_h - 1) * width_in + max_w];
                float down_score = temp_input_0
                    [((n * channel_in + c) * height_in + max_h + 1) * width_in + max_w];
                if (up_score > down_score) {
                    fh -= 0.25;
                } else if (up_score < down_score) {
                    fh += 0.25;
                }
            }
            if (reposition && max_w > 0 && max_w < width_in - 1) {
                float left_score = temp_input_0
                    [((n * channel_in + c) * height_in + max_h) * width_in + max_w - 1];
                float right_score = temp_input_0
                    [((n * channel_in + c) * height_in + max_h) * width_in + max_w + 1];
                if (left_score > right_score) {
                    fw -= 0.25;
                } else if (left_score < right_score) {
                    fw += 0.25;
                }
            }
            *output_0++ = ((fw + 0.5) * hm2c_op->coord_w) / width_in;
            *output_0++ = ((fh + 0.5) * hm2c_op->coord_h) / height_in;
            *output_0++ = max_score;
        }
    }
}

void op_default_heatmap2coord_tp_prepare(op_t *op)
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
        op->run_func = op_default_heatmap2coord_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
