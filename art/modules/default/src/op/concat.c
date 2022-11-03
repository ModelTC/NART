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
    uint32_t axis;
} op_concat_t;

op_concat_t *op_default_concat_tp_alloc(workspace_t *ws)
{
    (void)ws;
    op_concat_t *res = (op_concat_t *)malloc(sizeof(op_concat_t));
    memset(res, 0, sizeof(op_concat_t));
    return res;
}

void op_default_concat_tp_config(op_t *op)
{
    CHECK(op_setting_single_get(op, SETTING_CONCAT_AXIS, dtUINT32, &((op_concat_t *)op)->axis));
}

void op_default_concat_tp_destroy(op_t *op) { (void)op; }

void op_default_concat_tp_dealloc(op_t *op)
{
    if (NULL != op)
        free(op);
}

static void op_default_concat_run(op_t *op)
{
    int i, j;
    op_concat_t *concat_op = (op_concat_t *)op;
    size_t axis = concat_op->axis;
    size_t ch = op->input_tensors[0]->shape.dim[axis];
    size_t inner_num = shape_part_count(&op->input_tensors[0]->shape, concat_op->axis) / ch;
    size_t outer_num = shape_count(&op->input_tensors[0]->shape) / inner_num / ch;

    const float *input = NULL;
    float *output = (float *)mem_cpu_data(op->output_tensors[0].mem);

    for (i = 0; i < op->input_size; ++i) {
        input = (const float *)mem_cpu_data(op->input_tensors[i]->mem);
        size_t width = inner_num * op->input_tensors[i]->shape.dim[axis];
        size_t dpitch = inner_num * op->output_tensors[0].shape.dim[axis];
        float *p_output = output;
        for (j = 0; j < (int)outer_num; ++j) {
            memcpy(p_output, input, width * sizeof(float));
            input += width;
            p_output += dpitch;
        }
        output += inner_num * op->input_tensors[i]->shape.dim[axis];
    }
}

void op_default_concat_tp_prepare(op_t *op)
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
        op->run_func = op_default_concat_run;
        break;
    default:
        CHECK(false);
        break;
    }
}
