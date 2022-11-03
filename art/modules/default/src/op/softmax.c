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

typedef struct {
    op_t o;
    uint32_t axis;
} op_softmax_t;

op_softmax_t *op_default_softmax_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_softmax_t *res = (op_softmax_t *)malloc(sizeof(op_softmax_t));
    memset(res, 0, sizeof(op_softmax_t));
    return res;
}

void op_default_softmax_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_SOFTMAX_AXIS, dtUINT32, &((op_softmax_t *)op)->axis));
}

void op_default_softmax_tp_destroy(op_t *op) { (void)op; }

void op_default_softmax_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_softmax_run(op_t *op)
{
    const op_softmax_t *softmax_op = (op_softmax_t *)op;
    size_t axis = softmax_op->axis;
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_num = shape_part_count(&op->input_tensors[0]->shape, softmax_op->axis) / ch;
    size_t count = shape_count(&op->output_tensors[0].shape);
    size_t outer_num = count / inner_num / ch;

    size_t inner_idx, outer_idx, ch_idx;
    const float *input_0 = mem_cpu_data(op->input_tensors[0]->mem);
    float *output_0 = mem_cpu_data(op->output_tensors[0].mem);
    for (outer_idx = 0; outer_idx < outer_num; ++outer_idx) {
        for (inner_idx = 0; inner_idx < inner_num; ++inner_idx) {
            float tmp = 0.;
            float max = -FLT_MAX;
            for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
                size_t idx = outer_idx * ch * inner_num + ch_idx * inner_num + inner_idx;
                if (max < input_0[idx]) {
                    max = input_0[idx];
                }
            }
            for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
                size_t idx = outer_idx * ch * inner_num + ch_idx * inner_num + inner_idx;
                tmp += (output_0[idx] = exp(input_0[idx] - max));
            }
            for (ch_idx = 0; ch_idx < ch; ++ch_idx) {
                size_t idx = outer_idx * ch * inner_num + ch_idx * inner_num + inner_idx;
                output_0[idx] = output_0[idx] / tmp;
            }
        }
    }
}

void op_default_softmax_tp_prepare(op_t *op)
{
    int i;
    const op_softmax_t *softmax_op = (op_softmax_t *)op;
    size_t axis = softmax_op->axis;
    CHECK_GT((size_t)op->input_tensors[0]->shape.dim_size, axis);
    for (i = 0; i < op->input_size; ++i) {
        tensor_alloc(op->input_tensors[i]);
    }
    for (i = 0; i < op->output_size; ++i) {
        tensor_alloc(&op->output_tensors[i]);
    }
    switch (op->input_tensors[0]->dtype) {
    case dtFLOAT32:
        op->run_func = op_default_softmax_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
