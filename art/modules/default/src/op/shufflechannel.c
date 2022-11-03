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

typedef struct {
    op_t o;
    uint32_t group;
} op_shufflechannel_t;

op_shufflechannel_t *op_default_shufflechannel_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_shufflechannel_t *res = (op_shufflechannel_t *)malloc(sizeof(op_shufflechannel_t));
    memset(res, 0, sizeof(op_shufflechannel_t));
    return res;
}

void op_default_shufflechannel_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(
        op, SETTING_SHUFFLECHANNEL_GROUP, dtUINT32, &((op_shufflechannel_t *)op)->group));
}

void op_default_shufflechannel_tp_destroy(op_t *op) { (void)op; }

void op_default_shufflechannel_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

void group_resize(float *output, const float *input, int group_row, int group_column, int len)
{
    int i, j;
    for (i = 0; i < group_row; ++i) {
        for (j = 0; j < group_column; ++j) {
            const float *p_i = input + (i * group_column + j) * len;
            float *p_o = output + (j * group_row + i) * len;
            memcpy(p_o, p_i, sizeof(float) * len);
        }
    }
}

static void op_default_shufflechannel_run(op_t *op)
{
    op_shufflechannel_t *shufflechannel_op = (op_shufflechannel_t *)op;
    const float *bottom_data = mem_cpu_data(op->input_tensors[0]->mem);
    float *top_data = mem_cpu_data(op->output_tensors[0].mem);

    const int num = op->input_tensors[0]->shape.dim[0];
    const int chs = op->input_tensors[0]->shape.dim[1];
    const int feature_map_size = shape_part_count(&op->input_tensors[0]->shape, 1);
    const int sp_sz = shape_part_count(&op->input_tensors[0]->shape, 2);

    int group_row = shufflechannel_op->group;
    int group_column = chs / group_row;

    int i;
    for (i = 0; i < num; ++i) {
        group_resize(
            top_data + i * feature_map_size, bottom_data + i * feature_map_size, group_row,
            group_column, sp_sz);
    }
}

void op_default_shufflechannel_tp_prepare(op_t *op)
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
        op->run_func = op_default_shufflechannel_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
